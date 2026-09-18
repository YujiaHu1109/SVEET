import torch
import os
import csv
import imageio
import gc
import argparse
from PIL import Image
from tqdm import tqdm
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, feature_extraction_fn



def load_video_as_pil_list(path, num_frames=81, height=480, width=832):

    reader = imageio.get_reader(path)
    frames = []
    for frame in reader:
        frames.append(Image.fromarray(frame))
    reader.close()
    
    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1])
    elif len(frames) > num_frames:
        idx = torch.linspace(0, len(frames) - 1, num_frames).long().tolist()
        frames = [frames[i] for i in idx]
    
    frames = [f.resize((width, height)) for f in frames]
    return frames



def load_video_paths_from_csv(csv_path, dataset_root, video_column="video_path"):

    paths = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row[video_column].strip()
            # print(p)
            p = p if os.path.isabs(p) else os.path.join(dataset_root, p)
            if p and os.path.exists(p):
                paths.append(p)
            else:
                print(f"[Warning] 跳过不存在的视频: {p}")
    import random
    random.Random(42).shuffle(paths)
    return paths



def extract_feature_pair(video_path, prompt, pipe_wan, pipe_cf, vace_inject_layers,
                         num_frames=81, height=480, width=832):

    
    video_pil_list = load_video_as_pil_list(video_path, num_frames=num_frames,
                                             height=height, width=width)
    video_tensor = pipe_wan.preprocess_video(video_pil_list)
    
    pipe_wan.load_models_to_device(["vae"])
    z0 = pipe_wan.vae.encode(
        video_tensor, device=pipe_wan.device,
        tiled=True, tile_size=(30, 52), tile_stride=(15, 26),
    ).to(dtype=pipe_wan.torch_dtype, device=pipe_wan.device)
    
    # 2. prompt → text embedding
    pipe_wan.load_models_to_device(["text_encoder"])
    text_embed = pipe_wan.prompter.encode_prompt(prompt, positive=True, device=pipe_wan.device)
    
    
    pipe_wan.load_models_to_device(["dit"])
    feats_X = feature_extraction_fn(
        dit=pipe_wan.dit, latents=z0, context=text_embed,
        target_layers=vace_inject_layers, timestep_value=0,
    )
    
    
    pipe_cf.load_models_to_device(["dit"])
    feats_Y = feature_extraction_fn(
        dit=pipe_cf.dit, latents=z0, context=text_embed,
        target_layers=vace_inject_layers, timestep_value=0,
    )
    
    
    del z0, text_embed, video_tensor, video_pil_list
    
    return feats_X, feats_Y



def main():
    parser = argparse.ArgumentParser(description="Estimate SVEET feature maps W by ridge regression.")
    parser.add_argument("--dataset_csv", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--video_column", default="vace_video")
    parser.add_argument("--prompt", default="a high quality video")
    parser.add_argument("--num_train", type=int, default=500)
    parser.add_argument("--num_holdout", type=int, default=20)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--model_root", required=True,
                        help="Root containing Wan-AI/Wan2.1-VACE-1.3B")
    parser.add_argument("--causal_model_root", required=True,
                        help="Root containing the causal model's Wan-AI/Wan2.1-VACE-1.3B")
    parser.add_argument("--output", default="artifacts/W_matrices.pt")
    args = parser.parse_args()

    CSV_PATH, VIDEO_COLUMN, PROMPT = args.dataset_csv, args.video_column, args.prompt
    NUM_TRAIN, NUM_HOLDOUT = args.num_train, args.num_holdout
    NUM_FRAMES, HEIGHT, WIDTH, RIDGE = args.num_frames, args.height, args.width, args.ridge
    SAVE_PATH = args.output
    os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)
    LOCAL_MODEL_ROOT = args.model_root
    CAUSAL_MODEL_ROOT = args.causal_model_root
    MODEL_SUB_DIR = "Wan-AI/Wan2.1-VACE-1.3B"
    DIFFUSION_FILE = "diffusion_pytorch_model.safetensors"
    TEXT_ENCODER_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
    VAE_FILE = "Wan2.1_VAE.pth"
    
    vace_inject_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    
    
    print("Loading Wan pipeline...")
    pipe_wan = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(model_id=MODEL_SUB_DIR, origin_file_pattern=DIFFUSION_FILE,
                        local_model_path=LOCAL_MODEL_ROOT, skip_download=True),
            ModelConfig(model_id=MODEL_SUB_DIR, origin_file_pattern=TEXT_ENCODER_FILE,
                        local_model_path=LOCAL_MODEL_ROOT, skip_download=True),
            ModelConfig(model_id=MODEL_SUB_DIR, origin_file_pattern=VAE_FILE,
                        local_model_path=LOCAL_MODEL_ROOT, skip_download=True),
        ],
        redirect_common_files=False,
    )
    
    print("Loading CF pipeline...")
    pipe_cf = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(model_id=MODEL_SUB_DIR, origin_file_pattern=DIFFUSION_FILE,
                        local_model_path=CAUSAL_MODEL_ROOT, skip_download=True),
            ModelConfig(model_id=MODEL_SUB_DIR, origin_file_pattern=TEXT_ENCODER_FILE,
                        local_model_path=LOCAL_MODEL_ROOT, skip_download=True),
            ModelConfig(model_id=MODEL_SUB_DIR, origin_file_pattern=VAE_FILE,
                        local_model_path=LOCAL_MODEL_ROOT, skip_download=True),
        ],
        redirect_common_files=False,
    )
    
    print(f"Wan dit: {type(pipe_wan.dit).__name__}, dim={pipe_wan.dit.dim}, "
          f"num_blocks={len(pipe_wan.dit.blocks)}")
    print(f"CF  dit: {type(pipe_cf.dit).__name__}, dim={pipe_cf.dit.dim}, "
          f"num_blocks={len(pipe_cf.dit.blocks)}")
    
    C = pipe_wan.dit.dim
    
    
    all_paths = load_video_paths_from_csv(CSV_PATH, args.dataset_root, VIDEO_COLUMN)
    print(f"CSV 总视频数: {len(all_paths)}")
    
    train_paths = all_paths[:NUM_TRAIN]
    holdout_paths = all_paths[NUM_TRAIN:NUM_TRAIN + NUM_HOLDOUT]
    print(f"训练集: {len(train_paths)}, holdout: {len(holdout_paths)}")
    
    
    A = {ell: torch.zeros(C, C, dtype=torch.float64) for ell in vace_inject_layers}
    B = {ell: torch.zeros(C, C, dtype=torch.float64) for ell in vace_inject_layers}
    total_samples = 0
    
    
    print("\n=== 累加 X^T X 和 X^T Y ===")
    for video_path in tqdm(train_paths):
        try:
            feats_X, feats_Y = extract_feature_pair(
                video_path, PROMPT, pipe_wan, pipe_cf, vace_inject_layers,
                num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
            )
        except Exception as e:
            print(f"[Error] {video_path}: {e}")
            continue
        
        
        for ell in vace_inject_layers:
            Xl = feats_X[ell].reshape(-1, C).double()    # [B*T, C]
            Yl = feats_Y[ell].reshape(-1, C).double()
            A[ell] += Xl.T @ Xl
            B[ell] += Xl.T @ Yl
        
        
        total_samples += feats_X[vace_inject_layers[0]].reshape(-1, C).shape[0]
        
        
        del feats_X, feats_Y
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"\n累计 token 总数: {total_samples}")
    
    
    print("\n=== 闭式求解 W ===")
    W = {}
    for ell in vace_inject_layers:
        Cdim = A[ell].shape[0]
        A_reg = A[ell] + RIDGE * torch.eye(Cdim, dtype=torch.float64)
        W[ell] = torch.linalg.solve(A_reg, B[ell]).float()    
        
        
        sv = torch.linalg.svdvals(W[ell].double())
        identity_dist = (W[ell].double() - torch.eye(Cdim, dtype=torch.float64)).norm().item()
        print(f"Layer {ell:2d}: ||W|| = {W[ell].norm().item():.3f}, "
              f"||W - I|| = {identity_dist:.3f}, "
              f"sv range [{sv.min().item():.3f}, {sv.max().item():.3f}]")
    
    
    torch.save({"W": W, "vace_inject_layers": vace_inject_layers, "C": C,
                "total_samples": total_samples, "ridge": RIDGE},
               SAVE_PATH)
    print(f"\n已保存到 {SAVE_PATH}")
    
    
    print("\n=== Holdout 残差评估 ===")
    residual_sum = {ell: 0.0 for ell in vace_inject_layers}
    baseline_sum = {ell: 0.0 for ell in vace_inject_layers}
    holdout_count = 0
    
    for video_path in tqdm(holdout_paths):
        try:
            feats_X, feats_Y = extract_feature_pair(
                video_path, PROMPT, pipe_wan, pipe_cf, vace_inject_layers,
                num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
            )
        except Exception as e:
            print(f"[Error] {video_path}: {e}")
            continue
        
        for ell in vace_inject_layers:
            Xl = feats_X[ell].reshape(-1, C).double()
            Yl = feats_Y[ell].reshape(-1, C).double()
            
            Yl_pred = Xl @ W[ell].double()
            
            
            residual_sum[ell] += ((Yl_pred - Yl) ** 2).sum().item()
            baseline_sum[ell] += ((Xl - Yl) ** 2).sum().item()
        
        holdout_count += 1
        del feats_X, feats_Y
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"\n=== Layer-wise residual report (over {holdout_count} holdout videos) ===")
    print(f"{'Layer':>6} | {'Residual':>10} | {'Baseline':>10} | {'Improvement':>12}")
    print("-" * 50)
    for ell in vace_inject_layers:
        r = residual_sum[ell]
        b = baseline_sum[ell]
        improvement = (1 - r / b) * 100 if b > 0 else 0
        print(f"{ell:>6} | {r:>10.2f} | {b:>10.2f} | {improvement:>11.1f}%")
    
    print("\n说明: Improvement = 1 - residual/baseline。")
    print("> 80%:W 学得非常好,推理可以放心用")
    print("50%~80%:还行,可以试推理但效果可能打折")
    print("< 30%:线性映射不够,考虑换 MLP 或重新设计")


if __name__ == "__main__":
    main()
