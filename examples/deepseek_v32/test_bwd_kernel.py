# Test sparse_mla_bwd kernel against reference eager attention backward

import torch
import torch.nn.functional as F
from sparse_mla_fwd import sparse_mla_fwd_interface
from sparse_mla_bwd import sparse_mla_bwd

try:
    import einops
    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False


@torch.no_grad
def eager_attn_fwd(q, k, v, attn_bias, scale):
    """Forward pass for eager attention
    q: [b, sq, h, d]
    k: [b, sk, g, d]  
    v: [b, sk, g, dv]
    attn_bias: [b, sq, g, sk] (True = masked, False = attend)
    """
    b, sq, h, d = q.shape
    _, sk, g, dv = v.shape
    
    # Expand k, v to match heads
    heads_per_group = h // g
    k_exp = k.unsqueeze(3).expand(-1, -1, -1, heads_per_group, -1)  # [b, sk, g, hpg, d]
    k_exp = k_exp.reshape(b, sk, h, d)  # [b, sk, h, d]
    v_exp = v.unsqueeze(3).expand(-1, -1, -1, heads_per_group, -1)  # [b, sk, g, hpg, dv]
    v_exp = v_exp.reshape(b, sk, h, dv)  # [b, sk, h, dv]
    
    # Rearrange to (b, h, s, d)
    _q = einops.rearrange(q, 'b s h d -> b h s d')
    _k = einops.rearrange(k_exp, 'b s h d -> b h d s')
    _v = einops.rearrange(v_exp, 'b s h d -> b h s d')

    # Compute attention weights
    attn_w = torch.matmul(_q, _k) * scale  # [b, h, sq, sk]
    
    # Expand mask: [b, sq, g, sk] -> [b, h, sq, sk]
    # attn_bias: [b, sq, g, sk] -> [b, sq, g, hpg, sk] -> [b, sq, h, sk] -> [b, h, sq, sk]
    attn_bias_expanded = attn_bias.unsqueeze(3).expand(-1, -1, -1, heads_per_group, -1)  # [b, sq, g, hpg, sk]
    attn_bias_expanded = attn_bias_expanded.reshape(b, sq, h, sk)  # [b, sq, h, sk]
    attn_bias_expanded = attn_bias_expanded.permute(0, 2, 1, 3)  # [b, h, sq, sk]
    attn_w = attn_w.masked_fill(attn_bias_expanded, float('-inf'))

    # Compute attention scores
    probs = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(attn_w.dtype)

    # Compute attention output
    attn_output = torch.matmul(probs, _v)
    attn_output = einops.rearrange(attn_output, 'b h s d -> b s h d')

    return attn_output.contiguous(), probs


def eager_attn_bwd(q, k, v, attn_bias, scale, probs, grad_output):
    """Backward pass for eager attention
    q: [b, sq, h, d]
    k: [b, sk, g, d]
    v: [b, sk, g, dv]
    attn_bias: [b, sq, g, sk] (not used in backward, mask is baked into probs)
    probs: [b, h, sq, sk]
    grad_output: [b, sq, h, dv]
    """
    b, sq, h, d = q.shape
    _, sk, g, dv = v.shape
    heads_per_group = h // g
    
    # Expand k, v to match heads
    k_expanded = k.unsqueeze(3).expand(-1, -1, -1, heads_per_group, -1).reshape(b, sk, h, d)
    v_expanded = v.unsqueeze(3).expand(-1, -1, -1, heads_per_group, -1).reshape(b, sk, h, dv)
    
    # Rearrange
    _q_T = einops.rearrange(q, 'b s h d -> b h d s')
    _k_T = einops.rearrange(k_expanded, 'b s h d -> b h s d')
    _v_T = einops.rearrange(v_expanded, 'b s h d -> b h d s')
    grad_output_r = einops.rearrange(grad_output, 'b s h d -> b h s d')
    
    # dV = P^T @ dO
    attn_w_T = einops.rearrange(probs, 'b h sq sk -> b h sk sq')
    grad__v = torch.matmul(attn_w_T, grad_output_r)  # [b, h, sk, dv]
    
    # dP = dO @ V^T
    grad_attn_w = torch.matmul(grad_output_r, _v_T)  # [b, h, sq, sk]
    
    # Backward through softmax: dS = P * (dP - sum(P * dP))
    grad_logits = torch._softmax_backward_data(grad_attn_w, probs, -1, probs.dtype)
    
    # Backward through scale
    grad_logits = grad_logits * scale
    
    # dQ = dS @ K, dK = Q^T @ dS
    grad__q = torch.matmul(grad_logits, _k_T)  # [b, h, sq, d]
    grad__k = torch.matmul(_q_T, grad_logits)  # [b, h, d, sk]
    
    # Rearrange grads
    grad_v = einops.rearrange(grad__v, 'b h s d -> b s h d')  # [b, sk, h, dv]
    grad_k = einops.rearrange(grad__k, 'b h d s -> b s h d')  # [b, sk, h, d]
    grad_q = einops.rearrange(grad__q, 'b h s d -> b s h d')  # [b, sq, h, d]
    
    # Reduce grad_k, grad_v back to kv_groups
    grad_k = grad_k.view(b, sk, g, heads_per_group, d).sum(dim=3)  # [b, sk, g, d]
    grad_v = grad_v.view(b, sk, g, heads_per_group, dv).sum(dim=3)  # [b, sk, g, dv]
    
    return grad_q, grad_k, grad_v


def test_sparse_mla_bwd(
    batch=4,
    seq_len=128,
    seq_len_kv=256,
    dim=192,
    dim_v=128,
    topk=64,
    nheads=32,
    kv_groups=32,
):
    """Test sparse_mla_bwd kernel against reference implementation
    
    The sparse MLA kernel:
    1. Uses indices [b, s, g, topk] to gather K,V at specific positions
    2. Uses masks [b, s, g, skv] to optionally mask positions (True = masked)
    
    For comparison, the reference uses the same masks applied to full attention.
    """
    torch.manual_seed(42)
    
    print(f"Testing sparse_mla_bwd:")
    print(f"  batch={batch}, seq_len={seq_len}, seq_len_kv={seq_len_kv}")
    print(f"  dim={dim}, dim_v={dim_v}, topk={topk}")
    print(f"  nheads={nheads}, kv_groups={kv_groups}")
    
    assert topk % 32 == 0, "topk should be divisible by block_size (32)"
    assert nheads % kv_groups == 0, "nheads should be divisible by kv_groups"
    
    # Create inputs: [b, s, h, d] format
    q = torch.randn((batch, seq_len, nheads, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, seq_len_kv, kv_groups, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((batch, seq_len_kv, kv_groups, dim_v), device="cuda", dtype=torch.bfloat16)
    do = torch.randn((batch, seq_len, nheads, dim_v), device="cuda", dtype=torch.bfloat16)
    
    sm_scale = dim ** -0.5
    
    # Create sparse indices: select topk positions for each query
    # indices: [b, s, g, topk]
    indices = torch.zeros((batch, seq_len, kv_groups, topk), device="cuda", dtype=torch.int32)
    for b_i in range(batch):
        for s_i in range(seq_len):
            for g_i in range(kv_groups):
                perm = torch.randperm(seq_len_kv, device="cuda")[:topk].sort().values
                indices[b_i, s_i, g_i] = perm.to(torch.int32)
    
    # Create mask: True = masked (don't attend), False = attend
    # For sparse attention, we attend to indices positions, mask everything else
    # masks shape: [b, s, g, skv]
    masks = torch.ones((batch, seq_len, kv_groups, seq_len_kv), device="cuda", dtype=torch.bool)
    for b_i in range(batch):
        for s_i in range(seq_len):
            for g_i in range(kv_groups):
                masks[b_i, s_i, g_i].scatter_(0, indices[b_i, s_i, g_i].long(), False)
    
    # ============ Reference Forward + Backward ============
    q_ref = q.detach().clone()
    k_ref = k.detach().clone()
    v_ref = v.detach().clone()
    
    # Reference forward
    out_ref, probs_ref = eager_attn_fwd(q_ref, k_ref, v_ref, masks, sm_scale)
    
    # Reference backward
    dq_ref, dk_ref, dv_ref = eager_attn_bwd(q_ref, k_ref, v_ref, masks, sm_scale, probs_ref, do)
    
    print(f"\nReference output shape: {out_ref.shape}")
    print(f"Reference dq shape: {dq_ref.shape}")
    print(f"Reference dk shape: {dk_ref.shape}")
    print(f"Reference dv shape: {dv_ref.shape}")
    
    # ============ Sparse MLA Forward + Backward ============
    q_tl = q.detach().clone().contiguous()
    k_tl = k.detach().clone().contiguous()
    v_tl = v.detach().clone().contiguous()
    indices_tl = indices.contiguous()
    masks_tl = masks.contiguous()
    do_tl = do.detach().clone().contiguous()
    
    # Sparse MLA forward
    out_tl, lse_tl = sparse_mla_fwd_interface(q_tl, k_tl, v_tl, indices_tl, masks_tl, sm_scale=sm_scale)
    
    # Sparse MLA backward
    dq_tl, dk_tl, dv_tl = sparse_mla_bwd(q_tl, k_tl, v_tl, out_tl, do_tl, indices_tl, masks_tl, lse_tl, sm_scale)
    
    print(f"\nTileLang output shape: {out_tl.shape}")
    print(f"TileLang dq shape: {dq_tl.shape}")
    print(f"TileLang dk shape: {dk_tl.shape}")
    print(f"TileLang dv shape: {dv_tl.shape}")
    
    # ============ Compare Results ============
    print("\n" + "="*50)
    print("Comparing results:")
    
    # Forward comparison
    fwd_diff = (out_ref - out_tl).abs()
    print(f"\nForward output:")
    print(f"  Max diff: {fwd_diff.max().item():.6f}")
    print(f"  Mean diff: {fwd_diff.mean().item():.6f}")
    print(f"  Ref range: [{out_ref.min().item():.4f}, {out_ref.max().item():.4f}]")
    print(f"  TL range:  [{out_tl.min().item():.4f}, {out_tl.max().item():.4f}]")
    
    # dQ comparison
    dq_diff = (dq_ref - dq_tl).abs()
    print(f"\ndQ gradient:")
    print(f"  Max diff: {dq_diff.max().item():.6f}")
    print(f"  Mean diff: {dq_diff.mean().item():.6f}")
    print(f"  Ref range: [{dq_ref.min().item():.4f}, {dq_ref.max().item():.4f}]")
    print(f"  TL range:  [{dq_tl.min().item():.4f}, {dq_tl.max().item():.4f}]")
    
    # dK comparison
    dk_diff = (dk_ref - dk_tl).abs()
    print(f"\ndK gradient:")
    print(f"  Max diff: {dk_diff.max().item():.6f}")
    print(f"  Mean diff: {dk_diff.mean().item():.6f}")
    print(f"  Ref range: [{dk_ref.min().item():.4f}, {dk_ref.max().item():.4f}]")
    print(f"  TL range:  [{dk_tl.min().item():.4f}, {dk_tl.max().item():.4f}]")
    
    # dV comparison
    dv_diff = (dv_ref - dv_tl).abs()
    print(f"\ndV gradient:")
    print(f"  Max diff: {dv_diff.max().item():.6f}")
    print(f"  Mean diff: {dv_diff.mean().item():.6f}")
    print(f"  Ref range: [{dv_ref.min().item():.4f}, {dv_ref.max().item():.4f}]")
    print(f"  TL range:  [{dv_tl.min().item():.4f}, {dv_tl.max().item():.4f}]")
    
    # Check for zeros
    print("\n" + "="*50)
    print("Zero check:")
    print(f"  dq_ref zeros: {(dq_ref.abs() < 1e-6).sum().item()} / {dq_ref.numel()}")
    print(f"  dq_tl zeros:  {(dq_tl.abs() < 1e-6).sum().item()} / {dq_tl.numel()}")
    print(f"  dk_ref zeros: {(dk_ref.abs() < 1e-6).sum().item()} / {dk_ref.numel()}")
    print(f"  dk_tl zeros:  {(dk_tl.abs() < 1e-6).sum().item()} / {dk_tl.numel()}")
    print(f"  dv_ref zeros: {(dv_ref.abs() < 1e-6).sum().item()} / {dv_ref.numel()}")
    print(f"  dv_tl zeros:  {(dv_tl.abs() < 1e-6).sum().item()} / {dv_tl.numel()}")
    
    # Assertions
    rtol, atol = 0.1, 0.1
    print("\n" + "="*50)
    
    try:
        assert torch.allclose(out_ref, out_tl, rtol=rtol, atol=atol), "Forward output mismatch!"
        print("✓ Forward output matches")
    except AssertionError as e:
        print(f"✗ Forward output MISMATCH: {e}")
    
    try:
        assert torch.allclose(dq_ref, dq_tl, rtol=rtol, atol=atol), "dQ mismatch!"
        print("✓ dQ matches")
    except AssertionError as e:
        print(f"✗ dQ MISMATCH: {e}")
    
    try:
        assert torch.allclose(dk_ref, dk_tl, rtol=rtol, atol=atol), "dK mismatch!"
        print("✓ dK matches")
    except AssertionError as e:
        print(f"✗ dK MISMATCH: {e}")
    
    try:
        assert torch.allclose(dv_ref, dv_tl, rtol=rtol, atol=atol), "dV mismatch!"
        print("✓ dV matches")
    except AssertionError as e:
        print(f"✗ dV MISMATCH: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    # Test with different configurations
    # Use kv_groups=1 for simpler testing (nheads all share same KV)
    test_sparse_mla_bwd(
        batch=2,
        seq_len=128,
        seq_len_kv=256,
        dim=192,
        dim_v=128,
        topk=64,
        nheads=16,
        kv_groups=1,
    )
