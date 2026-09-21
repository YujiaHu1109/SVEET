"""Export a Causal/Self-Forcing generator checkpoint as Wan safetensors (single file, no sharding)."""
import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

WEIGHT_PREFIXES = ("_orig_mod.", "module.", "generator.", "model.")

def parse_size(value):
    value = value.strip().upper()
    units = {"B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3}
    for unit in ("GB", "MB", "KB", "B"):
        if value.endswith(unit):
            return int(float(value[:-len(unit)]) * units[unit])
    return int(value)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_dir", required=True,
                        help="Downloaded Wan2.1-T2V-1.3B directory")
    parser.add_argument("--checkpoint_path", required=True,
                        help="causal_forcing.pt or self_forcing_dmd.pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--state_key", default="generator",
                        help="Top-level checkpoint key; use generator_ema for EMA weights")
    parser.add_argument("--max_shard_size", default="4GB",
                        help="[UNUSED] kept for CLI compatibility, now single file always")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()

def find_state_dict(obj, state_key):
    if not isinstance(obj, Mapping):
        raise TypeError("The checkpoint root is not a mapping.")
    if state_key not in obj:
        available = ", ".join(map(str, list(obj)[:20]))
        raise KeyError(f"Checkpoint has no '{state_key}' key. Available keys: {available}")
    state = obj[state_key]
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint['{state_key}'] is not a state dict.")
    return state

def normalize_key(key):
    while True:
        for prefix in WEIGHT_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        else:
            return key

def extract_model_state(state):
    output = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized = normalize_key(key)
        if normalized in output:
            raise ValueError(f"Duplicate normalized key: {normalized}")
        output[normalized] = value.detach().cpu().contiguous()
    if not output:
        raise ValueError("No tensor weights were found in the selected checkpoint state.")
    return output

def base_weight_files(base_dir):
    index = base_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text(encoding="utf-8"))
        return [base_dir / name for name in sorted(set(data["weight_map"].values()))]
    files = sorted(base_dir.glob("diffusion_pytorch_model.safetensors"))
    if not files:
        raise FileNotFoundError(f"No diffusion safetensors found in {base_dir}")
    return files

def read_base_keys(base_dir):
    keys = set()
    for path in base_weight_files(base_dir):
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys

def copy_metadata(base_dir, output_dir):
    for name in ("config.json", "configuration.json"):
        source = base_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)

def save_single_shard(state, output_dir):
    """Always save as one single safetensors file, NO index.json"""
    out_path = output_dir / "diffusion_pytorch_model.safetensors"
    save_file(state, out_path, metadata={"format": "pt"})

def main():
    args = parse_args()
    base_dir = Path(args.base_model_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {base_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --force to overwrite")

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = extract_model_state(find_state_dict(checkpoint, args.state_key))
    base_keys = read_base_keys(base_dir)
    state_keys = set(state)

    missing = sorted(base_keys - state_keys)
    unexpected = sorted(state_keys - base_keys)

    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(
            f"The selected state is not a complete Wan generator: {len(missing)} base keys are missing "
            f"(first keys: {preview}). Do not merge a partial checkpoint as a full model."
        )
    if unexpected:
        print(f"Warning: exporting {len(unexpected)} keys not present in the base checkpoint.")

    # Clean old shard / index files
    for old_file in output_dir.glob("diffusion_pytorch_model.safetensors*"):
        old_file.unlink()

    copy_metadata(base_dir, output_dir)
    save_single_shard(state, output_dir)

    print(f"Exported {len(state)} tensors to {output_dir} (single safetensors file)")
    print(f"State source: checkpoint['{args.state_key}']")

if __name__ == "__main__":
    main()
