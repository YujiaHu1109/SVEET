from typing import List, Optional
from einops import rearrange, repeat
import torch
import time
import numpy as np

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation


class CausalInferencePipeline(torch.nn.Module):
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

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.kv_cache2 = None
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
        profile: bool = False,
        low_memory: bool = False,
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
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        # print(noise.shape)
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

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
            # print(vace_context.shape)
        
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
            # print(control_latents.shape, y.shape)
            y = torch.concat([control_latents, y], dim=1)


        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
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
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
            if self.kv_cache2 is not None:
                for block_index in range(len(self.kv_cache2)):
                    self.kv_cache2[block_index]["global_end_index"].fill_(0)
                    self.kv_cache2[block_index]["local_end_index"].fill_(0)

        # Step 2: Cache context feature
        current_start_frame = 0
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
                    kv_cache=self.kv_cache1,
                    vace_kv_cache=self.kv_cache2,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents =\
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                current_vace_input = vace_context[:, current_start_frame : current_start_frame + self.num_frame_per_block] if vace_context is not None else None
                current_y = y[:, :, current_start_frame : current_start_frame + self.num_frame_per_block] if y is not None else None

                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    vace_context=current_vace_input,
                    clip_feature=clip_feature,
                    y=current_y,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    vace_kv_cache=self.kv_cache2,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()
        # torch.cuda.synchronize()
        pure_inference_start_time = time.perf_counter()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            noisy_input = noise[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
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
            for index, current_timestep in enumerate(self.denoising_step_list):
                # print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        vace_context=vace_context_input,
                        clip_feature=clip_feature,
                        y=y_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        vace_kv_cache=self.kv_cache2,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        vace_context=vace_context_input,
                        clip_feature=clip_feature,
                        y=y_input,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        vace_kv_cache=self.kv_cache2,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                vace_context=vace_context_input,
                clip_feature=clip_feature,
                y=y_input,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                vace_kv_cache=self.kv_cache2,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # torch.cuda.synchronize()
        pure_inference_elapsed = time.perf_counter() - pure_inference_start_time
        print(pure_inference_elapsed)
        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_vace_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the VACE blocks.
        """
        vace_layers = getattr(self.generator.model, "vace_layers", [])
        num_vace_blocks = len(vace_layers)
        
        kv_cache2 = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 32760

        for _ in range(num_vace_blocks):
            kv_cache2.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache2 = kv_cache2
    
    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
