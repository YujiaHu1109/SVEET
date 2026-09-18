"""Run SVEET streaming video editing on one video."""

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model_dir", required=True,
                        help="Merged causal Wan/VACE model directory")
    parser.add_argument("--checkpoint_path", required=True,
                        help="Causal-Forcing generator checkpoint")
    parser.add_argument("--config_path", default="configs/causal_forcing_dmd_chunkwise.yaml")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--output_name", default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_output_frames", type=int, default=21,
                        help="Latent output frames; 21 corresponds to 81 RGB frames")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--text_encoder_path", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--vae_path", default=None)
    return parser.parse_args()


def safe_name(text):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text).strip("_")


def main():
    args = parse_args()
    input_path = Path(args.input_video).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint_path).expanduser().resolve()
    for path, label in ((input_path, "input video"), (model_dir, "model directory"),
                        (checkpoint, "checkpoint")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    os.environ["SVEET_MODEL_DIR"] = str(model_dir)
    if args.text_encoder_path:
        os.environ["SVEET_TEXT_ENCODER_PATH"] = str(Path(args.text_encoder_path).expanduser())
    if args.tokenizer_path:
        os.environ["SVEET_TOKENIZER_PATH"] = str(Path(args.tokenizer_path).expanduser())
    if args.vae_path:
        os.environ["SVEET_VAE_PATH"] = str(Path(args.vae_path).expanduser())

    import torch
    from einops import rearrange
    from omegaconf import OmegaConf
    from torchvision.io import write_video
    from pipeline import CausalDiffusionInferencePipeline, CausalInferencePipeline
    from utils.misc import set_seed
    from utils.video import VideoData
    from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

    if not torch.cuda.is_available():
        raise RuntimeError("SVEET inference requires a CUDA GPU.")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    set_seed(args.seed)
    torch.set_grad_enabled(False)

    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config_path),
    )
    pipeline_cls = (CausalInferencePipeline if hasattr(config, "denoising_step_list")
                    else CausalDiffusionInferencePipeline)
    pipeline = pipeline_cls(config, device=device).to(dtype=torch.bfloat16)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    key = "generator_ema" if args.use_ema else "generator"
    if key not in state:
        raise KeyError(f"Checkpoint does not contain '{key}'. Available: {list(state)[:10]}")
    pipeline.generator.load_state_dict(state[key], strict=False)

    low_memory = get_cuda_free_memory_gb(gpu) < 50
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)

    expected_frames = (args.num_output_frames - 1) * 4 + 1
    video_data = VideoData(str(input_path), height=args.height, width=args.width)
    if len(video_data) < expected_frames:
        raise ValueError(f"Input has {len(video_data)} frames; at least {expected_frames} are required.")
    video = [video_data[i] for i in range(expected_frames)]
    noise = torch.randn(
        [args.num_samples, args.num_output_frames, 16, args.height // 8, args.width // 8],
        device=device, dtype=torch.bfloat16,
    )
    video_out, _ = pipeline.inference(
        noise=noise,
        vace_video=video,
        text_prompts=args.prompt,
        return_latents=True,
        initial_latent=None,
        low_memory=low_memory,
    )
    frames = (255.0 * rearrange(video_out, "b t c h w -> b t h w c").cpu()).clamp(0, 255).to(torch.uint8)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_name or f"{input_path.stem}_{safe_name(args.prompt)[:48]}"
    for index in range(args.num_samples):
        output_path = output_dir / f"{stem}_{index}.mp4"
        write_video(str(output_path), frames[index], fps=args.fps)
        print(f"Saved: {output_path}")

    if hasattr(pipeline.vae, "model") and hasattr(pipeline.vae.model, "clear_cache"):
        pipeline.vae.model.clear_cache()


if __name__ == "__main__":
    main()
