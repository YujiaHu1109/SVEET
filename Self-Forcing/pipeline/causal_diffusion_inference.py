from tqdm import tqdm
from typing import List, Optional
from einops import rearrange, repeat
import torch
import numpy as np

from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = 50
        self.sample_solver = 'unipc'
        self.shift = args.timestep_shift

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache_1_pos = None
        self.kv_cache_1_neg = None
        self.kv_cache_2_pos = None
        self.kv_cache_2_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def preprocess_image(self, image, torch_dtype=torch.bfloat16, device="cuda", pattern="B C H W", min_value=-1, max_value=1):
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image


    def preprocess_video(self, video, torch_dtype=torch.bfloat16, device="cuda", pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a list of PIL.Image to torch.Tensor
        video = [self.preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value) for image in video]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video
    
    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        vace_video = None,
        control_video = None,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0,
        vace_reference_image = None, # newly added
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            start_frame_index (int): In long video generation, where does the current window start?
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            # Using a [1, 4, 4, 4, 4, 4] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # vace context
        vace_context = None
        if vace_video is not None:
            vace_video = self.preprocess_video(vace_video)
            
            vace_video_mask = torch.ones_like(vace_video)
     
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = self.vae.encode_to_latent(inactive).to(dtype=noise.dtype, device=noise.device)
            reactive = self.vae.encode_to_latent(reactive).to(dtype=noise.dtype, device=noise.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=2)

            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            vace_mask_latents = vace_mask_latents.permute(0, 2, 1, 3, 4)
            

            if vace_reference_image is not None:
                print("Using reference image for VACE conditioning")
                
                if not isinstance(vace_reference_image, list):
                    vace_reference_image = [vace_reference_image]

                
                vace_reference_image = self.preprocess_video(vace_reference_image)
                # print(vace_reference_image.shape)
                bs, c, f, h, w = vace_reference_image.shape

                
                # ref_frames = []
                # for j in range(f):
                #     ref_frames.append(vace_reference_image[0, :, j:j+1])

                ref_frames = torch.stack([vace_reference_image[0, :, j:j+1] for j in range(f)])

                
                vace_reference_latents = self.vae.encode_to_latent(ref_frames).to(dtype=noise.dtype, device=noise.device)
                # print(vace_reference_latents.shape)
                
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=2)
                # print(vace_reference_latents.shape)
                # print(vace_video_latents.shape)

                
                vace_video_latents = torch.concat((vace_reference_latents, vace_video_latents), dim=1)

                
                pad_mask = torch.zeros(
                    (1, f, vace_mask_latents.shape[2], vace_mask_latents.shape[3], vace_mask_latents.shape[4]),
                    dtype=vace_mask_latents.dtype,
                    device=vace_mask_latents.device
                )
                vace_mask_latents = torch.concat((pad_mask, vace_mask_latents), dim=1)

            # print(vace_mask_latents.shape, vace_video_latents.shape)
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=2)
            print(vace_context.shape)
        
        clip_feature = None
        y = None
        y_input = None

        if control_video is not None:
            control_video = self.preprocess_video(control_video)
            control_latents = self.vae.encode_to_latent(control_video).to(dtype=noise.dtype, device=noise.device)
            control_latents = control_latents.permute(0, 2, 1, 3, 4)

            y_dim = 16 # pipe.dit.in_dim-control_latents.shape[1]-latents.shape[1]
            if clip_feature is None or y is None:
                clip_feature = torch.zeros((1, 257, 1280), dtype=noise.dtype, device=noise.device)
                y = torch.zeros((1, y_dim, num_frames, height, width), dtype=noise.dtype, device=noise.device)
            else:
                y = y[:, -y_dim:]
            print(control_latents.shape, y.shape)
            y = torch.concat([control_latents, y], dim=1)

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_1_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            if vace_context is not None:
                # self._initialize_vace_kv_cache(
                #     batch_size=batch_size,
                #     dtype=noise.dtype,
                #     device=noise.device
                # )
                pass
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache_1_pos)):
                self.kv_cache_1_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_1_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_1_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_1_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
            if self.kv_cache_2_pos is not None:
                for block_index in range(len(self.kv_cache_2_pos)):
                    self.kv_cache_2_pos[block_index]["global_end_index"].fill_(0)
                    self.kv_cache_2_pos[block_index]["local_end_index"].fill_(0)
                    self.kv_cache_2_neg[block_index]["global_end_index"].fill_(0)
                    self.kv_cache_2_neg[block_index]["local_end_index"].fill_(0)



        # Step 2: Cache context feature
        current_start_frame = start_frame_index
        cache_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                current_vace_input = vace_context[:, :1] if vace_context is not None else None
                current_y = y[:, :, :1] if y is not None else None
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    vace_context=current_vace_input,
                    clip_feature = clip_feature,
                    y = current_y,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_1_pos,
                    vace_kv_cache=self.kv_cache_2_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=unconditional_dict,
                    vace_context=current_vace_input,
                    clip_feature = clip_feature,
                    y = current_y,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_1_neg,
                    vace_kv_cache=self.kv_cache_2_pos,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents =\
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                current_vace_input = vace_context[:, cache_start_frame : cache_start_frame + self.num_frame_per_block] if vace_context is not None else None
                current_y = y[:, :, cache_start_frame : cache_start_frame + self.num_frame_per_block] if y is not None else None

                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    vace_context=current_vace_input,
                    clip_feature=clip_feature,
                    y=current_y,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_1_pos,
                    vace_kv_cache=self.kv_cache_2_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    vace_context=current_vace_input,
                    clip_feature=clip_feature,
                    y=current_y,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_1_neg,
                    vace_kv_cache=self.kv_cache_2_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input
            if vace_context is not None:
                vace_context_input = vace_context[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
                # print(noisy_input.shape, vace_context_input.shape)
            else:
                vace_context_input = None
            if y is not None:
                y_input = y[:, :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            else:
                y = None

            # Step 3.1: Spatial denoising loop
            sample_scheduler = self._initialize_sample_scheduler(noise)
            for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
                print(f"current_timestep: {t}")
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                )

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    vace_context=vace_context_input,
                    clip_feature=clip_feature,
                    y=y_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_1_pos,
                    vace_kv_cache=self.kv_cache_2_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    vace_context=vace_context_input,
                    clip_feature=clip_feature,
                    y=y_input,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_1_neg,
                    vace_kv_cache=self.kv_cache_2_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0
                # print(f"kv_cache['local_end_index']: {self.kv_cache_pos[0]['local_end_index']}")
                # print(f"kv_cache['global_end_index']: {self.kv_cache_pos[0]['global_end_index']}")

            # Step 3.2: record the model's output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            self.generator(
                noisy_image_or_video=latents,
                vace_context=vace_context_input,
                clip_feature=clip_feature,
                y=y_input,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_1_pos,
                vace_kv_cache=self.kv_cache_2_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )
            self.generator(
                noisy_image_or_video=latents,
                vace_context=vace_context_input,
                clip_feature=clip_feature,
                y=y_input,
                conditional_dict=unconditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_1_neg,
                vace_kv_cache=self.kv_cache_2_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_1_pos = []
        kv_cache_1_neg = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache_1_pos.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_1_neg.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache_1_pos = kv_cache_1_pos  # always store the clean cache
        self.kv_cache_1_neg = kv_cache_1_neg  # always store the clean cache

    def _initialize_vace_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the VACE blocks.
        """
        vace_layers = getattr(self.generator.model, "vace_layers", [])
        num_vace_blocks = len(vace_layers)
        
        kv_cache_2_pos = []
        kv_cache_2_neg = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 32760

        for _ in range(num_vace_blocks):
            kv_cache_2_pos.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_2_neg.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache_2_pos = kv_cache_2_pos
        self.kv_cache_2_neg = kv_cache_2_neg
    
    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache_pos = []
        crossattn_cache_neg = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
            crossattn_cache_neg.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache_pos = crossattn_cache_pos  # always store the clean cache
        self.crossattn_cache_neg = crossattn_cache_neg  # always store the clean cache

    def _initialize_sample_scheduler(self, noise):
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                self.sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(self.sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler
