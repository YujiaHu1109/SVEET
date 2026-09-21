





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
    assert eye_check < 1e-3, f"V_k columns are not orthogonal: max diff = {eye_check}"
    
    Pi_perp = torch.eye(n, dtype=V_k.dtype, device=V_k.device) - V_k @ V_k.T
    
    if verbose:    
        idemp_check = (Pi_perp @ Pi_perp - Pi_perp).abs().max().item()
        trace_val = Pi_perp.diagonal().sum().item()
        print(f"  ||A_1|| = {A_1.norm().item():.3f}")
        print(f"  Singular values (first 5): {S[:5].tolist()}")
        print(f"  Top 10 singular values contribute: {(S[:10]**2).sum() / (S**2).sum() * 100:.1f}%")
        print(f"  Selected k = {k}, V_k shape = {V_k.shape}")
        print(f"  Pi_perp idempotency error: {idemp_check:.2e}")
        print(f"  Pi_perp trace = {trace_val:.2f}, expected ≈ {n - k}")
    
    return Pi_perp, V_k, k, S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w_path", type=str, default="W_matrices.pt",
                        help="{ell: W}.pt")
    parser.add_argument("--save_path", type=str, default="pi_perp_0.8.pt",
                        help="Path to save Pi_perp")
    parser.add_argument("--energy_threshold", type=float, default=0.8,
                        help="Automatically select k based on energy threshold")
    parser.add_argument("--fixed_k", type=int, default=None,
                        help="If specified, use the same k for all layers (overrides energy_threshold)")
    args = parser.parse_args()
    
    
    print(f"Loading W from {args.w_path}")
    W_data = torch.load(args.w_path, map_location="cpu", weights_only=False)
    W_dict = W_data["W"]
    
    
    n_list = [W.shape[0] for W in W_dict.values()]
    assert all(n == n_list[0] for n in n_list), "The shapes of W are not consistent"
    n = n_list[0]
    print(f"Dimension n = {n}, total layers: {len(W_dict)}")
    
    
    Pi_perp_dict = {}
    V_k_dict = {}
    k_dict = {}
    S_dict = {}
    
    print(f"\n=== Starting to compute Pi_perp ===")
    print(f"  Energy threshold: {args.energy_threshold}")
    if args.fixed_k is not None:
        print(f"  Fixed k: {args.fixed_k}")
    
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
    
    print(f"\n=== Saved to {args.save_path} ===")
    print("k selected for each layer:")
    for ell in sorted(k_dict.keys()):
        print(f"  Layer {ell}: k = {k_dict[ell]}, n - k = {n - k_dict[ell]} (VACE available dimensions)")


if __name__ == "__main__":
    main()