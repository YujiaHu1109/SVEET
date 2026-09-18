





import torch
import argparse


def compute_pi_perp_for_layer(W_1, W_b=None, k=None, energy_threshold=0.9, verbose=True):
















    n = W_1.shape[0]
    if W_b is None:
        W_b = torch.eye(n, dtype=W_1.dtype, device=W_1.device)
    
    
    A_1 = W_1 - W_b   # [n, n]
    
    # ---- SVD ----
    U, S, Vt = torch.linalg.svd(A_1, full_matrices=False)
    # U: [n, n], S: [n], Vt: [n, n]
    
    
    if k is None:
        cumsum = torch.cumsum(S ** 2, dim=0)
        total = cumsum[-1]
        k = (cumsum / total < energy_threshold).sum().item() + 1
        
        k = max(1, min(k, n - 1))
    
    
    V_k = Vt[:k].T   # [n, k]
    
    
    
    
    eye_check = (V_k.T @ V_k - torch.eye(k, dtype=V_k.dtype, device=V_k.device)).abs().max()
    assert eye_check < 1e-3, f"V_k 列不正交: max diff = {eye_check}"
    
    
    Pi_perp = torch.eye(n, dtype=V_k.dtype, device=V_k.device) - V_k @ V_k.T
    
    if verbose:
        
        idemp_check = (Pi_perp @ Pi_perp - Pi_perp).abs().max().item()
        trace_val = Pi_perp.diagonal().sum().item()
        print(f"  ||A_1|| = {A_1.norm().item():.3f}")
        print(f"  奇异值前 5: {S[:5].tolist()}")
        print(f"  奇异值前 10 占总能量: {(S[:10]**2).sum() / (S**2).sum() * 100:.1f}%")
        print(f"  选定 k = {k}, V_k shape = {V_k.shape}")
        print(f"  Pi_perp 幂等性误差: {idemp_check:.2e}")
        print(f"  Pi_perp 迹 = {trace_val:.2f}, 期望 ≈ {n - k}")
    
    return Pi_perp, V_k, k, S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w_path", type=str, default="W_matrices.pt",
                        help="包含 {ell: W} 的 .pt 文件")
    parser.add_argument("--save_path", type=str, default="pi_perp_0.8.pt",
                        help="保存 Pi_perp 的路径")
    parser.add_argument("--energy_threshold", type=float, default=0.8,
                        help="按能量阈值自动选 k")
    parser.add_argument("--fixed_k", type=int, default=None,
                        help="如果指定,所有层用同一个 k(覆盖 energy_threshold)")
    args = parser.parse_args()
    
    
    print(f"Loading W from {args.w_path}")
    W_data = torch.load(args.w_path, map_location="cpu", weights_only=False)
    W_dict = W_data["W"]
    
    
    n_list = [W.shape[0] for W in W_dict.values()]
    assert all(n == n_list[0] for n in n_list), "所有层 W 形状不一致"
    n = n_list[0]
    print(f"维度 n = {n}, 共 {len(W_dict)} 层")
    
    
    Pi_perp_dict = {}
    V_k_dict = {}
    k_dict = {}
    S_dict = {}
    
    print(f"\n=== 开始计算 Pi_perp ===")
    print(f"  能量阈值: {args.energy_threshold}")
    if args.fixed_k is not None:
        print(f"  固定 k: {args.fixed_k}")
    
    for ell in sorted(W_dict.keys()):
        print(f"\n--- Layer {ell} ---")
        W_1 = W_dict[ell].double()    
        
        Pi_perp, V_k, k_used, S = compute_pi_perp_for_layer(
            W_1, W_b=None,
            k=args.fixed_k,
            energy_threshold=args.energy_threshold,
            verbose=True,
        )
        
        Pi_perp_dict[ell] = Pi_perp.float()    
        V_k_dict[ell] = V_k.float()
        k_dict[ell] = k_used
        S_dict[ell] = S.float()
    
    
    torch.save({
        "Pi_perp": Pi_perp_dict,
        "V_k": V_k_dict,
        "k": k_dict,
        "singular_values": S_dict,
        "energy_threshold": args.energy_threshold,
        "n": n,
        "method": "weight_orthogonal",
    }, args.save_path)
    
    print(f"\n=== 已保存到 {args.save_path} ===")
    print("各层选定的 k:")
    for ell in sorted(k_dict.keys()):
        print(f"  Layer {ell}: k = {k_dict[ell]}, n - k = {n - k_dict[ell]} (VACE 可用维度)")


if __name__ == "__main__":
    main()