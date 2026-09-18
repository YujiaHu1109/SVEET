# SVEET: Streaming Video Editing with Easy Adaptation
![teaser](assets/teaser.png)

Official implementation of **SVEET**, a framework that learns video editing on a bidirectional
Wan2.1-VACE backbone and transfers the learned control branch to a causal streaming backbone.

The release implements the complete workflow described in the paper:

1. collect paired bidirectional/causal hidden features and estimate layer-wise maps `W`;
2. construct orthogonal-complement projectors `Pi_perp` from `W - I`;
3. train the VACE control branch with temporally independent attention and orthogonal constraints;
4. merge the trained LoRA into a full VACE checkpoint;
5. run causal streaming inference with the transferred control branch.

> The repository does not redistribute Wan, VACE, Causal Forcing checkpoints, datasets, or
> generated videos. Download them from their original providers and comply with their licenses.

## Released Checkpoints
You can skip the training stage and directly use our released SVEET control LoRA checkpoint:
- HuggingFace repo: [Cicici1109/SVEET](https://huggingface.co/Cicici1109/SVEET/tree/main)

Place the downloaded `style.safetensors` under `checkpoints/sveet-control/`. You **still need to run the LoRA merge script** before streaming inference.

## Repository layout

```text
SVEET-release/
├── DiffSynth-Studio/     # bidirectional feature extraction and VACE training
├── Self-Forcing/                 # causal streaming inference
├── tools/
│   ├── feature_map.py            # ridge regression for W
│   └── feature_or.py             # SVD and Pi_perp construction
├── data/dataset_example.csv
├── requirements.txt
└── THIRD_PARTY.md
```

## Installation

Python 3.10, Linux, CUDA-capable NVIDIA GPUs, and a recent CUDA toolkit are recommended.
Install PyTorch for your CUDA version first, then install the remaining dependencies:

```bash
conda create -n sveet python=3.10 -y
conda activate sveet

# Example only. Select the PyTorch command matching your CUDA driver.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e ./DiffSynth-Studio
pip install -e ./Self-Forcing
```

FlexAttention is compiled by TorchInductor/Triton on first use. Ensure `gcc`, Python development
headers, the CUDA driver library, and enough `/tmp` space are available.

## Checkpoints

Prepare the following assets without committing them to Git:

```text
checkpoints/
├── bidirectional/Wan-AI/Wan2.1-VACE-1.3B/
│   ├── diffusion_pytorch_model*.safetensors
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   ├── Wan2.1_VAE.pth
│   └── google/umt5-xxl/...
├── causal/Wan-AI/Wan2.1-VACE-1.3B/...
└── causal_forcing.pt
```

The causal model directory used for final inference should contain the transferred/merged SVEET
control weights alongside the text encoder, tokenizer, and VAE, or those three paths can be
overridden independently in `infer_single.py`.

## Dataset format

Paths in the metadata CSV are relative to `--dataset_base_path`:

```csv
video,prompt,vace_video,vace_reference_image
target/example.mp4,"Make it a watercolor painting.",source/example.mp4,target/example.mp4
```

- `video`: training target video;
- `prompt`: edit instruction;
- `vace_video`: source/control video;
- `vace_reference_image`: optional reference input.

See `data/dataset_example.csv` for the original organization pattern. The example contains paths
only, not the underlying videos.

## 1. Estimate feature maps W

Run from the repository root:

```bash
PYTHONPATH=DiffSynth-Studio python tools/feature_map.py \
  --dataset_csv data/dataset_example.csv \
  --dataset_root /path/to/dataset \
  --video_column vace_video \
  --model_root /path/to/checkpoints/bidirectional \
  --causal_model_root /path/to/checkpoints/causal \
  --num_train 500 \
  --num_holdout 20 \
  --ridge 1e-3 \
  --output artifacts/W_matrices.pt
```

The script accumulates `X^T X` and `X^T Y` online and solves the regularized least-squares
problem independently for each injected VACE layer.

## 2. Construct Pi_perp

```bash
python tools/feature_or.py \
  --w_path artifacts/W_matrices.pt \
  --energy_threshold 0.8 \
  --save_path artifacts/pi_perp_0.8.pt
```

For every layer, the script computes the SVD of `W - I`, selects the smallest rank that retains
the requested spectral energy, and saves `Pi_perp = I - V_k V_k^T`.

## 3. Train the control branch

```bash
cd DiffSynth-Studio

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


## 4. Merge the trained LoRA

Training produces a LoRA `.safetensors` file. Merge it into the full VACE checkpoint once before
streaming inference:

```bash
cd DiffSynth-Studio

python merge_lora.py \
  --base_model_dir /path/to/checkpoints/bidirectional/Wan-AI/Wan2.1-VACE-1.3B \
  --lora_path ../checkpoints/sveet-control/style.safetensors \
  --output_model_dir ../checkpoints/merged/Wan2.1-VACE-1.3B \
  --alpha 1.0 \
  --assets_mode symlink
```

The script reconstructs the full VACE branch, applies each `lora_B @ lora_A` update, retains the
base DiT weights, and writes a merged `diffusion_pytorch_model.safetensors`. With
`--assets_mode symlink`, the text encoder, VAE, and tokenizer are linked into the output directory;
use `copy` for a portable standalone directory or `none` if those assets are managed separately.

This is intentionally an offline step. Performing it inside `infer_single.py` would initialize the
DiffSynth and Self-Forcing model stacks in the same process, substantially increasing peak host/GPU
memory and repeating an invariant merge for every inference run.

## 5. Streaming inference

After merging/transferring the trained VACE control weights into the causal model directory:

```bash
cd Self-Forcing

python infer_single.py \
  --input_video /path/to/source.mp4 \
  --prompt "Make it a Japanese anime style, cel shading." \
  --model_dir ../checkpoints/merged/Wan2.1-VACE-1.3B \
  --checkpoint_path /path/to/causal_forcing.pt \
  --config_path configs/causal_forcing_dmd_chunkwise.yaml \
  --output_dir outputs \
  --num_output_frames 21 \
  --seed 123
```

`num_output_frames=21` expects 81 decoded input frames because the Wan VAE has temporal stride 4.
The script consumes the first 81 frames and writes 15 FPS MP4 output by default.

## Reproducibility notes

- `feature_map.py` uses a deterministic seed (`42`) when shuffling calibration videos.
- Store generated `W_matrices.pt` and `pi_perp_*.pt` under `artifacts/`; both are ignored by Git.
- Model and dataset paths are CLI arguments or environment variables; no user-specific absolute
  paths are required.
- Multi-GPU training uses one process per GPU through Accelerate. Configure it with
  `accelerate config` before launching.

## Acknowledgements and licenses

This code builds on DiffSynth-Studio, Wan2.1-VACE, and Self-Forcing/Causal Forcing. Their retained
license files are included in the corresponding subdirectories. See [THIRD_PARTY.md](THIRD_PARTY.md)
before redistribution. 
