"""Merge a trained SVEET VACE LoRA into a full Wan2.1-VACE checkpoint."""

import argparse
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_dir", required=True,
                        help="Directory containing the original Wan2.1-VACE files")
    parser.add_argument("--lora_path", required=True,
                        help="LoRA safetensors produced by training")
    parser.add_argument("--output_model_dir", required=True,
                        help="Destination directory for the merged model")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--diffusion_file", default="diffusion_pytorch_model.safetensors")
    parser.add_argument("--text_encoder_file", default="models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--vae_file", default="Wan2.1_VAE.pth")
    parser.add_argument("--assets_mode", choices=("symlink", "copy", "none"), default="symlink",
                        help="How to place the text encoder, VAE, and tokenizer in the output")
    return parser.parse_args()


def lora_pairs(state_dict):
    pairs = {}
    for key in state_dict:
        if ".lora_B." not in key:
            continue
        parts = key.split(".")
        index = parts.index("lora_B")
        if len(parts) > index + 2:
            parts.pop(index + 1)
        parts.pop(index)
        parts.pop(-1)
        target = ".".join(parts)
        a_key = key.replace(".lora_B.", ".lora_A.")
        if a_key not in state_dict:
            raise KeyError(f"Missing LoRA A tensor for {key}")
        pairs[target] = (key, a_key)
    return pairs


def place_asset(source: Path, destination: Path, mode: str):
    if not source.exists() or mode == "none":
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    if mode == "copy":
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main():
    args = parse_args()
    base_dir = Path(args.base_model_dir).expanduser().resolve()
    lora_path = Path(args.lora_path).expanduser().resolve()
    output_dir = Path(args.output_model_dir).expanduser().resolve()
    diffusion_path = base_dir / args.diffusion_file
    text_encoder_path = base_dir / args.text_encoder_file
    vae_path = base_dir / args.vae_file
    for path in (diffusion_path, text_encoder_path, vae_path, lora_path):
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"Loading base checkpoint: {diffusion_path}")
    base_state = load_file(str(diffusion_path))
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cpu",
        model_configs=[
            ModelConfig(path=str(diffusion_path)),
            ModelConfig(path=str(text_encoder_path)),
            ModelConfig(path=str(vae_path)),
        ],
        redirect_common_files=False,
    )

    lora_state = load_file(str(lora_path))
    pairs = lora_pairs(lora_state)
    updated = 0
    matched = set()
    for name, module in pipe.vace.named_modules():
        if name not in pairs:
            continue
        b_key, a_key = pairs[name]
        up = lora_state[b_key].float()
        down = lora_state[a_key].float()
        if up.ndim == 4:
            delta = torch.mm(up.squeeze(), down.squeeze()).unsqueeze(-1).unsqueeze(-1)
        else:
            delta = torch.mm(up, down)
        module_state = module.state_dict()
        if "weight" not in module_state:
            raise KeyError(f"Matched module has no weight: {name}")
        module_state["weight"] = module_state["weight"].float() + args.alpha * delta
        module.load_state_dict(module_state)
        matched.add(name)
        updated += 1

    if updated == 0:
        raise RuntimeError("No VACE modules matched the LoRA checkpoint.")
    unmatched = sorted(set(pairs) - matched)
    if unmatched:
        print(f"Warning: {len(unmatched)} LoRA targets were not matched. First entries: {unmatched[:10]}")

    final_state = {
        key: value.cpu()
        for key, value in base_state.items()
        if "vace_blocks" not in key
    }
    for key, value in pipe.vace.state_dict().items():
        final_state[key] = value.to(torch.bfloat16).cpu()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.diffusion_file
    save_file(final_state, str(output_path))

    place_asset(text_encoder_path, output_dir / args.text_encoder_file, args.assets_mode)
    place_asset(vae_path, output_dir / args.vae_file, args.assets_mode)
    place_asset(base_dir / "google", output_dir / "google", args.assets_mode)

    print(f"Merged {updated} VACE modules.")
    print(f"Saved merged checkpoint: {output_path}")
    print(f"Final tensor count: {len(final_state)}")


if __name__ == "__main__":
    main()
