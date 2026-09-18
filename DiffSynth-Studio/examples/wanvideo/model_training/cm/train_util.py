import copy
import functools
import os
import torch

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

from .fp16_util import (
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)
import numpy as np

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model, # 待训练扩散模型
        diffusion, # 扩散过程管理器
        data, # batch, cond
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None, # 扩散模型时间步采样器
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        # for param in self.mp_trainer.master_params:
        #     param.data = param.data.to(torch.bfloat16)

        self.opt = RAdam(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        # if self.resume_step:
        #     self._load_optimizer_state()
        #     # Model was resumed, either due to a restart or a checkpoint
        #     # being specified at the command line.
        #     self.ema_params = [
        #         self._load_ema_parameters(rate) for rate in self.ema_rate
        #     ]
        # else:
        #     # self.ema_params = [
        #     #     copy.deepcopy(self.mp_trainer.master_params)
        #     #     for _ in range(len(self.ema_rate))
        #     # ]
        from collections import defaultdict
        import torch
        
        def get_gpu_free_memory(gpu_id=0):
            """获取指定GPU的剩余显存（返回：字节、MB、GB）"""
            if not torch.cuda.is_available():
                return 0, 0, 0
            # 获取GPU总显存和剩余显存（字节）
            mem_info = torch.cuda.mem_get_info(gpu_id)
            free_bytes = mem_info[0]
            total_bytes = mem_info[1]
            free_mb = free_bytes / (1024 * 1024)
            free_gb = free_bytes / (1024 * 1024 * 1024)
            total_gb = total_bytes / (1024 * 1024 * 1024)
            used_gb = total_gb - free_gb
            return free_bytes, free_mb, free_gb, used_gb, total_gb
        
        def calculate_param_memory_and_dtype(params):
            """计算参数列表的显存占用 + 统计数据类型分布"""
            total_bytes = 0
            dtype_count = defaultdict(int)
            dtype_memory = defaultdict(int)
            
            print("\n===== Master Params 显存+数据类型明细 =====")
            for i, param in enumerate(params):
                if param is None:
                    continue
                param_bytes = param.numel() * param.element_size()
                param_mb = param_bytes / (1024 * 1024)
                param_gb = param_bytes / (1024 * 1024 * 1024)
                total_bytes += param_bytes
                
                dtype = str(param.dtype)
                dtype_count[dtype] += 1
                dtype_memory[dtype] += param_bytes
                
                shape_str = str(list(param.shape))
                if i < 10 or i == len(params)-1:
                    print(f"参数{i:4d} | 形状：{shape_str:<20} | 类型：{dtype:<15} | 占用：{param_mb:.2f} MB ({param_gb:.4f} GB)")
            
            # 汇总参数信息
            total_mb = total_bytes / (1024 * 1024)
            total_gb = total_bytes / (1024 * 1024 * 1024)
            print("\n" + "="*60)
            print(f"【参数总览】Master Params 总显存：{total_mb:.2f} MB ({total_gb:.4f} GB) | 总参数数：{len(params)} 个")
            print(f"【参数设备】参数所在GPU：{params[0].device if len(params)>0 else '无'}")
            print("\n【数据类型分布】")
            for dtype in dtype_count:
                dtype_mb = dtype_memory[dtype] / (1024 * 1024)
                dtype_gb = dtype_memory[dtype] / (1024 * 1024 * 1024)
                ratio = (dtype_count[dtype] / len(params)) * 100 if len(params) > 0 else 0
                mem_ratio = (dtype_memory[dtype] / total_bytes) * 100 if total_bytes > 0 else 0
                print(f"- {dtype:<15}：{dtype_count[dtype]} 个 ({ratio:.2f}%) | 显存：{dtype_mb:.2f} MB ({dtype_gb:.4f} GB) ({mem_ratio:.2f}%)")
            print("="*60 + "\n")
            
            return total_bytes, total_mb, total_gb
        
        # 1. 打印参数信息
        total_bytes, total_mb, total_gb = calculate_param_memory_and_dtype(self.mp_trainer.master_params)
        
        # 2. 打印复制前 GPU 剩余显存（核心新增）
        free_bytes, free_mb, free_gb, used_gb, total_gb_gpu = get_gpu_free_memory(gpu_id=0)
        print("===== 复制前 GPU 显存状态 =====")
        print(f"GPU 0 总显存：{total_gb_gpu:.4f} GB")
        print(f"GPU 0 已用显存：{used_gb:.4f} GB")
        print(f"GPU 0 剩余显存：{free_mb:.2f} MB ({free_gb:.4f} GB)")
        print(f"【关键对比】需要克隆的参数显存：{total_gb:.4f} GB | 剩余显存：{free_gb:.4f} GB")
        if free_gb < total_gb:
            print(f"⚠️ 警告：剩余显存 ({free_gb:.4f} GB) < 需要克隆的参数显存 ({total_gb:.4f} GB)，大概率触发 OOM！")
        else:
            print(f"✅ 剩余显存充足，但仍可能因碎片化触发 OOM！")
        print("="*60 + "\n")
        
        self.ema_params = []
        # for _ in range(len(self.ema_rate)):
        #     ema_param = []
        #     for param in self.mp_trainer.master_params:
        #         # 直接在 CPU 上 clone
        #         p = param.detach().cpu().clone()
        #         p.requires_grad = False
        #         ema_param.append(p)
        #     self.ema_params.append(ema_param)
        


        # if th.cuda.is_available():
        #     self.use_ddp = True
        #     self.ddp_model = DDP(
        #         self.model,
        #         device_ids=[dist_util.dev()],
        #         output_device=dist_util.dev(),
        #         broadcast_buffers=False,
        #         bucket_cap_mb=128,
        #         find_unused_parameters=False,
        #     )
        # else:
        #     if dist.get_world_size() > 1:
        #         logger.warn(
        #             "Distributed training requires CUDA. "
        #             "Gradients will not be synchronized properly!"
        #         )
        #     self.use_ddp = False
        #     self.ddp_model = self.model
        self.use_ddp = False
        self.ddp_model = self.model

        self.step = self.resume_step

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    ),
                )

        dist_util.sync_params(self.model.parameters())
        dist_util.sync_params(self.model.buffers())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while not self.lr_anneal_steps or self.step < self.lr_anneal_steps:
            # batch, cond = next(self.data)
            video = next(self.data)['video']
            vace_video = next(self.data)['vace_video']
            prompt = next(self.data)['prompt']
            # self.run_step(batch, cond)
            self.run_step(video, vace_video, prompt)

            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    # def run_step(self, batch, cond):
    #     self.forward_backward(batch, cond) #
    def run_step(self, video, vace_video, prompt):
        self.forward_backward(video, vace_video, prompt) 
        torch.cuda.empty_cache()
        took_step = self.mp_trainer.optimize(self.opt) # 优化器更新参数
        if took_step: 
            self.step += 1
            self._update_ema() # 更新EMA
        self._anneal_lr() # 学习率退火
        self.log_step() # 记录日志

    # def forward_backward(self, batch, cond):
    def forward_backward(self, video, vace_video, prompt):
        self.mp_trainer.zero_grad()
        # for i in range(0, batch.shape[0], self.microbatch):
        for i in range(0, video.shape[0], self.microbatch):
            # micro = batch[i : i + self.microbatch].to(dist_util.dev())
            # micro_cond = {
            #     k: v[i : i + self.microbatch].to(dist_util.dev())
            #     for k, v in cond.items()
            # }
            # last_batch = (i + self.microbatch) >= batch.shape[0]
            micro = video
            micro_cond = {
                "prompt": prompt,
                "vace_video": vace_video
            }
            last_batch = (i + self.microbatch) >= video.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, self.mp_trainer.master_params)
        dist.barrier()


class CMTrainLoop(TrainLoop):
    def __init__(
        self,
        *,
        target_model,
        teacher_model,
        teacher_diffusion,
        training_mode,
        ema_scale_fn,
        total_training_steps,
        model_logger=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.training_mode = training_mode
        self.ema_scale_fn = ema_scale_fn
        self.target_model = target_model
        self.teacher_model = teacher_model
        self.teacher_diffusion = teacher_diffusion
        self.total_training_steps = total_training_steps
        self.model_logger = model_logger

        if target_model:
            self._load_and_sync_target_parameters()
            self.target_model.requires_grad_(False)
            self.target_model.train()

            self.target_model_param_groups_and_shapes = get_param_groups_and_shapes(
                self.target_model.named_parameters()
            )
            # self.target_model_master_params = make_master_params(
            #     self.target_model_param_groups_and_shapes
            # )
            self.target_model_master_params = list(self.target_model.parameters())

        if teacher_model:
            # self._load_and_sync_teacher_parameters()
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()

        self.global_step = self.step
        if training_mode == "progdist":
            self.target_model.eval()
            _, scale = ema_scale_fn(self.global_step)
            if scale == 1 or scale == 2:
                _, start_scale = ema_scale_fn(0)
                n_normal_steps = int(np.log2(start_scale // 2)) * self.lr_anneal_steps
                step = self.global_step - n_normal_steps
                if step != 0:
                    self.lr_anneal_steps *= 2
                    self.step = step % self.lr_anneal_steps
                else:
                    self.step = 0
            else:
                self.step = self.global_step % self.lr_anneal_steps

    def _load_and_sync_target_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            target_name = name.replace("model", "target_model")
            resume_target_checkpoint = os.path.join(path, target_name)
            if bf.exists(resume_target_checkpoint) and dist.get_rank() == 0:
                logger.log(
                    "loading model from checkpoint: {resume_target_checkpoint}..."
                )
                self.target_model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_target_checkpoint, map_location=dist_util.dev()
                    ),
                )

        dist_util.sync_params(self.target_model.parameters())
        dist_util.sync_params(self.target_model.buffers())

    def _load_and_sync_teacher_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            teacher_name = name.replace("model", "teacher_model")
            resume_teacher_checkpoint = os.path.join(path, teacher_name)

            if bf.exists(resume_teacher_checkpoint) and dist.get_rank() == 0:
                logger.log(
                    "loading model from checkpoint: {resume_teacher_checkpoint}..."
                )
                self.teacher_model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_teacher_checkpoint, map_location=dist_util.dev()
                    ),
                )

        dist_util.sync_params(self.teacher_model.parameters())
        dist_util.sync_params(self.teacher_model.buffers())

    def run_loop(self):
        saved = False
        while (
            not self.lr_anneal_steps
            or self.step < self.lr_anneal_steps
            or self.global_step < self.total_training_steps
        ):
            import gc
            gc.collect()
            th.cuda.empty_cache()
            # print(next(self.data))
            # batch, cond = next(self.data)\
            data = next(self.data)
            inputs = self.model.forward_preprocess(data)
            if isinstance(inputs, dict):
                print("\n===== inputs 详细信息（key: 类型/形状） =====")
                for k, v in inputs.items():
                    # 打印key + 数据类型
                    info = f"key: {k:<20} | 类型: {type(v).__name__}"
                    # 如果是tensor，补充形状和设备信息
                    if hasattr(v, 'shape'):
                        info += f" | 形状: {v.shape}"
                        if hasattr(v, 'device'):
                            info += f" | 设备: {v.device}"
                    # 如果是列表/字符串，补充长度/内容
                    elif isinstance(v, (list, str)):
                        info += f" | 长度: {len(v)}"
                        if isinstance(v, str):
                            info += f" | 内容: {v[:50]}..."  # 只打印前50个字符，避免过长
                    print(info)

            # self.run_step(batch, cond)
            self.run_step(inputs)
            print(self.global_step, self.save_interval)
            saved = False
            if (
                self.global_step
                and self.save_interval != -1
                and self.global_step % self.save_interval == 0
            ):
            # if True:
                self.save()
                saved = True
                th.cuda.empty_cache()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            if self.global_step % self.log_interval == 0:
                logger.dumpkvs()

        # Save the last checkpoint if it wasn't already saved.
        if not saved:
            self.save()

    # def run_step(self, batch, cond):
    #     self.forward_backward(batch, cond)
    def run_step(self, inputs):
        self.forward_backward(inputs)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
            if self.target_model:
                self._update_target_ema()
            if self.training_mode == "progdist":
                self.reset_training_for_progdist()
            self.step += 1
            self.global_step += 1

        self._anneal_lr()
        self.log_step()

    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with th.no_grad():
            update_ema(
                self.target_model_master_params,
                self.mp_trainer.master_params,
                rate=target_ema,
            )
            # master_params_to_model_params(
            #     self.target_model_param_groups_and_shapes,
            #     self.target_model_master_params,
            # )

    def reset_training_for_progdist(self):
        assert self.training_mode == "progdist", "Training mode must be progdist"
        if self.global_step > 0:
            scales = self.ema_scale_fn(self.global_step)[1]
            scales2 = self.ema_scale_fn(self.global_step - 1)[1]
            if scales != scales2:
                with th.no_grad():
                    update_ema(
                        self.teacher_model.parameters(),
                        self.model.parameters(),
                        0.0,
                    )
                # reset optimizer
                self.opt = RAdam(
                    self.mp_trainer.master_params,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )

                self.ema_params = [
                    copy.deepcopy(self.mp_trainer.master_params)
                    for _ in range(len(self.ema_rate))
                ]
                if scales == 2:
                    self.lr_anneal_steps *= 2
                self.teacher_model.eval()
                self.step = 0

    # def forward_backward(self, batch, cond):
    def forward_backward(self, inputs):
        self.mp_trainer.zero_grad()
        # for i in range(0, batch.shape[0], self.microbatch):
        for i in range(0, 1):
            # micro = batch[i : i + self.microbatch].to(dist_util.dev())
            # micro_cond = {
            #     k: v[i : i + self.microbatch].to(dist_util.dev())
            #     for k, v in cond.items()
            # }
            # last_batch = (i + self.microbatch) >= batch.shape[0]
            micro = inputs["input_latents"]
            micro_cond = inputs
            # last_batch = (i + self.microbatch) >= video.shape[0]
            last_batch = False
            print(self.use_ddp)
            # t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            t, weights = self.schedule_sampler.sample(1, dist_util.dev())
            print(t, weights)

            ema, num_scales = self.ema_scale_fn(self.global_step)
            if self.training_mode == "progdist":
                if num_scales == self.ema_scale_fn(0)[1]:
                    compute_losses = functools.partial(
                        self.diffusion.progdist_losses,
                        self.ddp_model,
                        micro,
                        num_scales,
                        target_model=self.teacher_model,
                        target_diffusion=self.teacher_diffusion,
                        model_kwargs=micro_cond,
                    )
                else:
                    compute_losses = functools.partial(
                        self.diffusion.progdist_losses,
                        self.ddp_model,
                        micro,
                        num_scales,
                        target_model=self.target_model,
                        target_diffusion=self.diffusion,
                        model_kwargs=micro_cond,
                    )
            elif self.training_mode == "consistency_distillation":
                self.model.pipe.scheduler.set_timesteps(num_scales, denoising_strength=1.0, training=True, shift=5.0)
                compute_losses = functools.partial(
                    self.diffusion.consistency_losses,
                    self.ddp_model,
                    micro,
                    num_scales,
                    target_model=self.target_model,
                    teacher_model=self.teacher_model,
                    teacher_diffusion=self.teacher_diffusion,
                    model_kwargs=micro_cond,
                )
            elif self.training_mode == "consistency_training":
                compute_losses = functools.partial(
                    self.diffusion.consistency_losses,
                    self.ddp_model,
                    micro,
                    num_scales,
                    target_model=self.target_model,
                    model_kwargs=micro_cond,
                )
            else:
                raise ValueError(f"Unknown training mode {self.training_mode}")

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            weights = weights.to(losses["loss"].device)
            loss = (losses["loss"] * weights).mean()

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
            print(loss)

    # def save(self):
    #     import blobfile as bf

    #     step = self.global_step

    #     def save_checkpoint(rate, params):
    #         state_dict = self.mp_trainer.master_params_to_state_dict(params)
    #         if dist.get_rank() == 0:
    #             logger.log(f"saving model {rate}...")
    #             if not rate:
    #                 filename = f"model{step:06d}.pt"
    #             else:
    #                 filename = f"ema_{rate}_{step:06d}.pt"
    #             with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
    #                 th.save(state_dict, f)

    #     for rate, params in zip(self.ema_rate, self.ema_params):
    #         save_checkpoint(rate, params)

    #     logger.log("saving optimizer state...")
    #     if dist.get_rank() == 0:
    #         with bf.BlobFile(
    #             bf.join(get_blob_logdir(), f"opt{step:06d}.pt"),
    #             "wb",
    #         ) as f:
    #             th.save(self.opt.state_dict(), f)

    #     if dist.get_rank() == 0:
    #         if self.target_model:
    #             logger.log("saving target model state")
    #             filename = f"target_model{step:06d}.pt"
    #             with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
    #                 th.save(self.target_model.state_dict(), f)
    #         if self.teacher_model and self.training_mode == "progdist":
    #             logger.log("saving teacher model state")
    #             filename = f"teacher_model{step:06d}.pt"
    #             with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
    #                 th.save(self.teacher_model.state_dict(), f)

    #     # Save model parameters last to prevent race conditions where a restart
    #     # loads model at step N, but opt/ema state isn't saved for step N.
    #     save_checkpoint(0, self.mp_trainer.master_params)
    #     dist.barrier()
    def save(self):
        # 1. 准备数据：从 mp_trainer 获取最新的 master_params
        # 这一步是必须要做的，因为 master_params 才是最新的权重
        state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
        
        # 2. 调用 ModelLogger 的核心逻辑
        if self.model_logger is not None:
            import torch.distributed as dist
            
            # 确保所有卡都跑到了这里
            dist.barrier()
            
            # 只有 Rank 0 负责写入磁盘
            if dist.get_rank() == 0:
                logger.log(f"Using ModelLogger to save step {self.global_step}...")
                
                # 模拟你 ModelLogger.save_model 里的逻辑
                # 注意：这里我们直接用 self.model (WanTrainingModule)，不需要 accelerator
                trainable_state_dict = self.model.export_trainable_state_dict(
                    state_dict, 
                    remove_prefix=self.model_logger.remove_prefix_in_ckpt
                )
                
                # 如果有转换器就执行
                trainable_state_dict = self.model_logger.state_dict_converter(trainable_state_dict)
                
                # 执行保存
                import os
                from safetensors.torch import save_file
                os.makedirs(self.model_logger.output_path, exist_ok=True)
                
                # 保存 Student
                save_file(
                    trainable_state_dict, 
                    os.path.join(self.model_logger.output_path, f"student_step-{self.global_step}.safetensors")
                )
                
                # 如果有 target_model，顺便也存一份 Target
                if self.target_model:
                    target_state = self.target_model.state_dict()
                    target_trainable = self.model.export_trainable_state_dict(
                        target_state, 
                        remove_prefix=self.model_logger.remove_prefix_in_ckpt
                    )
                    save_file(
                        target_trainable, 
                        os.path.join(self.model_logger.output_path, f"target_step-{self.global_step}.safetensors")
                    )
                    
                # 3. 别忘了保存优化器 (Resume 需要)，这部分 Logger 通常不管，我们单独存
                th.save(self.opt.state_dict(), os.path.join(self.model_logger.output_path, f"opt_step-{self.global_step}.pt"))

            dist.barrier()

    def log_step(self):
        step = self.global_step
        logger.logkv("step", step)
        logger.logkv("samples", (step + 1) * self.global_batch)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
