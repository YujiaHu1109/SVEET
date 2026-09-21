# SVEET: Streaming Video Editing with Easy Adaptation

![teaser](assets/teaser.png)

Official implementation of **SVEET**, a framework that learns video editing on a bidirectional
Wan2.1-VACE backbone and transfers the learned control branch to a causal streaming backbone.

This repository supports two workflows:

- **Inference:** use our released SVEET LoRA without training.
- **Training and method reproduction:** estimate `W`, construct `Pi_perp`, and train a new editing task.

## Repository layout

```text
SVEET-release/
├── DiffSynth-Studio-vaceori/     # VACE training and offline LoRA merging
├── Self-Forcing/                 # causal streaming inference
├── tools/
│   ├── export_causal_checkpoint.py
│   ├── feature_map.py            # estimate W
│   └── feature_or.py             # construct Pi_perp
├── data/dataset_example.csv
├── assets/
├── requirements.txt
└── THIRD_PARTY.md
```

## Installation

Python 3.10, Linux, an NVIDIA GPU, and a compatible CUDA toolkit are recommended. Install the
PyTorch build matching your driver before installing the remaining dependencies. The command below
uses the CUDA 12.8 wheels as an example.

```bash
conda create -n sveet python=3.10 -y
conda activate sveet

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e ./DiffSynth-Studio-vaceori
pip install -e ./Self-Forcing
```

Install the Hugging Face CLI used by the download commands:

```bash
pip install -U "huggingface_hub[cli]"
```

FlexAttention is compiled by TorchInductor/Triton on first use. Ensure `gcc`, Python development
headers, the CUDA driver library, and sufficient `/tmp` space are available.

# Inference with the released SVEET checkpoint

No training, `W` estimation, or `Pi_perp` construction is required for this workflow.

## Required checkpoints for inference

Only the entries marked **required** below must be downloaded. The `merged` directory is generated
locally by the merge command and must not be downloaded separately.

```text
checkpoints/
├── bidirectional/Wan-AI/Wan2.1-VACE-1.3B/   # required: downloaded base VACE model
│   ├── diffusion_pytorch_model.safetensors
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   ├── Wan2.1_VAE.pth
│   ├── google/umt5-xxl/...
│   └── config.json
├── sveet-control/                            # required: released SVEET LoRA
│   └── style.safetensors
├── chunkwise/                                # required: Causal Forcing generator
│   └── causal_forcing.pt
└── merged/Wan2.1-VACE-1.3B/                 # generated locally; do not download
    ├── diffusion_pytorch_model.safetensors
    ├── models_t5_umt5-xxl-enc-bf16.pth
    ├── Wan2.1_VAE.pth
    ├── google/umt5-xxl/...
    └── config.json
```

### 1. Download Wan2.1-VACE-1.3B

Model page: [Wan-AI/Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B)

```bash
hf download Wan-AI/Wan2.1-VACE-1.3B \
  --local-dir checkpoints/bidirectional/Wan-AI/Wan2.1-VACE-1.3B
```

### 2. Download the released SVEET LoRA

Checkpoint: [Cicici1109/SVEET `style.safetensors`](https://huggingface.co/Cicici1109/SVEET/resolve/main/style.safetensors)

Using the Hugging Face CLI:

```bash
hf download Cicici1109/SVEET style.safetensors \
  --local-dir checkpoints/sveet-control
```

Alternatively, download the same file directly:

```bash
mkdir -p checkpoints/sveet-control
wget -O checkpoints/sveet-control/style.safetensors \
  https://huggingface.co/Cicici1109/SVEET/resolve/main/style.safetensors
```

### 3. Download Causal Forcing

Model page: [zhuhz22/Causal-Forcing](https://huggingface.co/zhuhz22/Causal-Forcing)

The provided inference configuration uses the chunk-wise checkpoint:

```bash
hf download zhuhz22/Causal-Forcing chunkwise/causal_forcing.pt \
  --local-dir checkpoints
```

This creates `checkpoints/chunkwise/causal_forcing.pt`.

### 4. Merge the released SVEET LoRA

The released `style.safetensors` is a VACE LoRA. Merge it into the full VACE checkpoint once before
inference:

```bash
python DiffSynth-Studio-vaceori/merge_lora.py \
  --base_model_dir checkpoints/bidirectional/Wan-AI/Wan2.1-VACE-1.3B \
  --lora_path checkpoints/sveet-control/style.safetensors \
  --output_model_dir checkpoints/merged/Wan2.1-VACE-1.3B \
  --alpha 1.0 \
  --assets_mode symlink
```

`--assets_mode symlink` avoids duplicating the text encoder, tokenizer, and VAE. Use `copy` when a
portable standalone output directory is preferred. The script always copies `config.json`.

### 5. Run streaming inference

```bash
cd Self-Forcing

python infer_single.py \
  --input_video /path/to/source.mp4 \
  --prompt "Make it a Japanese anime style, cel shading." \
  --model_dir ../checkpoints/merged/Wan2.1-VACE-1.3B \
  --checkpoint_path ../checkpoints/chunkwise/causal_forcing.pt \
  --config_path configs/causal_forcing_dmd_chunkwise.yaml \
  --output_dir outputs \
  --num_output_frames 21 \
  --seed 123
```

`num_output_frames=21` corresponds to 81 RGB frames because the Wan VAE has temporal stride 4.
The input must contain at least 81 frames. The default output frame rate is 15 FPS.

# Training and method reproduction

Everything in this section is optional for users who only want to run inference with the released
`style.safetensors`.

## Additional checkpoints required only for training/reproduction

```text
checkpoints/
├── bidirectional/Wan-AI/Wan2.1-T2V-1.3B/   # training/reproduction only
│   ├── diffusion_pytorch_model.safetensors
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   ├── Wan2.1_VAE.pth
│   ├── google/umt5-xxl/...
│   └── config.json
└── causal/Wan-AI/Wan2.1-T2V-1.3B/          # generated locally for W estimation
    ├── diffusion_pytorch_model.safetensors
    └── config.json

artifacts/                                   # generated locally
├── W_matrices.pt
└── pi_perp_0.8.pt
```

Wan2.1-T2V-1.3B model page:
[Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir checkpoints/bidirectional/Wan-AI/Wan2.1-T2V-1.3B
```

The same `checkpoints/chunkwise/causal_forcing.pt` downloaded for inference is used to construct the
causal model directory. If it has not been downloaded yet, run:

```bash
hf download zhuhz22/Causal-Forcing chunkwise/causal_forcing.pt \
  --local-dir checkpoints
```

Export the complete causal generator as a single safetensors file:

```bash
python tools/export_causal_checkpoint.py \
  --base_model_dir checkpoints/bidirectional/Wan-AI/Wan2.1-T2V-1.3B \
  --checkpoint_path checkpoints/chunkwise/causal_forcing.pt \
  --state_key generator \
  --output_dir checkpoints/causal/Wan-AI/Wan2.1-T2V-1.3B
```

The exporter validates the generator state against the base Wan model, writes
`diffusion_pytorch_model.safetensors`, and copies `config.json` from the base model directory.

## Dataset format

Paths in the metadata CSV are relative to `--dataset_base_path`:

```csv
video,prompt,vace_video,vace_reference_image
target/example.mp4,"Make it a watercolor painting.",source/example.mp4,target/example.mp4
```

- `video`: target video.
- `prompt`: editing instruction.
- `vace_video`: source/control video.
- `vace_reference_image`: optional reference input.

See `data/dataset_example.csv` for the metadata format. The example includes paths only, not the
underlying videos.

## 1. Estimate feature maps W

Run from the repository root:

```bash
PYTHONPATH=DiffSynth-Studio-vaceori python tools/feature_map.py \
  --dataset_csv data/dataset_example.csv \
  --dataset_root /path/to/dataset \
  --video_column vace_video \
  --model_root checkpoints/bidirectional \
  --causal_model_root checkpoints/causal \
  --bidirectional_model_id Wan-AI/Wan2.1-T2V-1.3B \
  --causal_model_id Wan-AI/Wan2.1-T2V-1.3B \
  --num_train 500 \
  --num_holdout 20 \
  --ridge 1e-3 \
  --output artifacts/W_matrices.pt
```

The script accumulates `X^T X` and `X^T Y` online and solves the regularized least-squares problem
for each injection layer.

## 2. Construct Pi_perp

```bash
python tools/feature_or.py \
  --w_path artifacts/W_matrices.pt \
  --energy_threshold 0.8 \
  --save_path artifacts/pi_perp_0.8.pt
```

For every layer, the script computes the SVD of `W - I`, selects the smallest rank retaining the
requested spectral energy, and saves `Pi_perp = I - V_k V_k^T`.

## 3. Train the control branch

The VACE model needed here is the same
[Wan-AI/Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B) downloaded for inference.
If it is not available locally, download it with:

```bash
hf download Wan-AI/Wan2.1-VACE-1.3B \
  --local-dir checkpoints/bidirectional/Wan-AI/Wan2.1-VACE-1.3B
```

Then launch training:

```bash
cd DiffSynth-Studio-vaceori

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /path/to/dataset \
  --dataset_metadata_path ../data/dataset_example.csv \
  --data_file_keys "video,vace_video" \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --pi_perp_path ../artifacts/pi_perp_0.8.pt \
  --learning_rate 1e-4 \
  --weight_decay 1e-2 \
  --save_steps 1000 \
  --num_epochs 10 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path ../checkpoints/sveet-control \
  --lora_base_model vace \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 128 \
  --extra_inputs vace_video \
  --use_gradient_checkpointing_offload
```


## 4. Merge a newly trained LoRA

Replace `style.safetensors` with the checkpoint produced by training:

```bash
python DiffSynth-Studio-vaceori/merge_lora.py \
  --base_model_dir checkpoints/bidirectional/Wan-AI/Wan2.1-VACE-1.3B \
  --lora_path checkpoints/sveet-control/epoch-9.safetensors \
  --output_model_dir checkpoints/merged/Wan2.1-VACE-1.3B \
  --alpha 1.0 \
  --assets_mode symlink
```

## Reproducibility notes

- `feature_map.py` uses seed `42` when shuffling calibration videos.
- `W` and `Pi_perp` are specific to the target causal generator. Recompute both when switching
  between Causal Forcing and Self-Forcing checkpoints.
- Keep generated `W_matrices.pt` and `pi_perp_*.pt` under `artifacts/`; both are ignored by Git.
- Multi-GPU training uses one process per GPU through Accelerate. Run `accelerate config` before
  launching distributed training.

## Acknowledgements and licenses

This code builds on
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio),
[Wan2.1-VACE](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B), and
[Causal Forcing](https://github.com/thu-ml/Causal-Forcing). Retained license files are included in
the corresponding subdirectories. See [THIRD_PARTY.md](THIRD_PARTY.md) before redistribution.
