
import torch
import torch.distributed as dist
from torch.nn import functional as F
import os
from ring_ref import Ref
from ring_wrapper import AttentionFuncionWithContextParallel

try:
    import einops

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

@torch.no_grad
def eager_attn_fwd(q, k, v, attn_bias, sinks, scale, dropout):
    """Forward pass for eager attention"""

    # Rearrange query, key, value to (b, h, s, d)
    b, sq, h, d = q.shape
    sk = k.shape[1]
    _q = einops.rearrange(q, 'b s h d -> b h s d')
    _k = einops.rearrange(k, 'b s h d -> b h d s')
    _v = einops.rearrange(v, 'b s h d -> b h s d')

    # Compute attention weights
    attn_w = torch.matmul(_q, _k) * scale
    attn_w = attn_w + attn_bias

    # Add sinks to attention weights
    if sinks is None:
        logits = attn_w
    else:
        _sinks = sinks.reshape(1, h, 1, 1).expand(b, -1, sq, 1)
        logits = torch.cat([attn_w, _sinks], dim=-1)

    # Compute attention scores
    probs = F.softmax(logits, dim=-1, dtype=logits.dtype)
    if sinks is None:
        attn_w = probs
    else:
        attn_w = probs[..., :-1]  # Drop the sink

    # Compute attention output
    attn_output = torch.matmul(attn_w, _v)
    attn_output = einops.rearrange(attn_output, 'b h s d -> b s h d')
    attn_output = attn_output.contiguous()

    return attn_output, probs


def init_dist():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

def create_cp_pg(cp_size=4):
    """
    Context Parallel (CP) groups:
    world ranks: [0..world_size-1]
    groups: [0,1,2,3], [4,5,6,7], ...
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size % cp_size == 0

    cp_groups = []
    my_cp_pg = None
    my_cp_rank = None

    for start in range(0, world_size, cp_size):
        ranks = list(range(start, start + cp_size))
        pg = dist.new_group(ranks=ranks)
        cp_groups.append(pg)

        if rank in ranks:
            my_cp_pg = pg
            my_cp_rank = ranks.index(rank)

    return my_cp_pg, my_cp_rank

def test_kernel(
    batch,
    seq_len,
    seq_len_kv,
    dim,
    tail_dim,
    topk,
    nheads,
    kv_group,
    cp_size
):
    init_dist()
    cp_pg, curr_rank = create_cp_pg(cp_size=4)

    torch.manual_seed(42)
    # q: [seq_len_shard, batch, nheads, dim + tail_dim]
    # kv: [seq_len_kv_shard, batch, kv_group, dim + tail_dim]
    #   k: [seq_len_kv_shard, batch, kv_group, dim + tail_dim]
    #   v: [seq_len_kv_shard, batch, kv_group, dim]
    # indices: [batch, kv_group, seq_len, topk]
    q_full  = torch.randn((seq_len,    batch, nheads,   dim + tail_dim), device="cuda", dtype=torch.bfloat16)
    kv_full = torch.randn((seq_len_kv, batch, kv_group, dim + tail_dim), device="cuda", dtype=torch.bfloat16)

    q_tmp  = torch.chunk(q_full,  cp_size * 2, dim=0)
    kv_tmp = torch.chunk(kv_full, cp_size * 2, dim=0)

    mirror = cp_size * 2 - curr_rank - 1
    q_local  = torch.cat([q_tmp[curr_rank],  q_tmp[mirror]],  dim=0).contiguous().requires_grad_()
    kv_local = torch.cat([kv_tmp[curr_rank], kv_tmp[mirror]], dim=0).contiguous().requires_grad_()
    k_full, v_full = kv_full.clone().contiguous(), kv_full[..., :dim].clone().contiguous()
    k_local = kv_local.detach().clone().contiguous().requires_grad_(True)
    v_local = kv_local[..., :dim].detach().clone().contiguous().requires_grad_(True)

    # indices: long, no grad
    perm = torch.randperm(seq_len_kv, device="cuda")
    indices_full = perm[:topk]
    indices_full = indices_full.view(1, 1, 1, topk).expand(
        batch, 1, seq_len, topk
    ).contiguous().to(torch.int32)

    # mask: float/bf16, no grad
    attn_mask_full = torch.ones(
        (batch, 1, seq_len, seq_len_kv),
        device="cuda",
        dtype=torch.bool,
    )
    attn_mask_full.scatter_(-1, indices_full, False)

    sparse_mask_tmp   = torch.chunk(attn_mask_full, cp_size * 2, dim=2)
    sparse_mask_local = torch.cat([sparse_mask_tmp[curr_rank], sparse_mask_tmp[mirror]], dim=2).contiguous()

    indices_tmp   = torch.chunk(indices_full, cp_size * 2, dim=2)
    indices_local = torch.cat([indices_tmp[curr_rank], indices_tmp[mirror]], dim=2).contiguous().expand(-1, kv_group, -1, -1).contiguous()

    sm_scale = (dim + tail_dim)**-0.5
    attention_dropout = 0

    do = torch.randn((seq_len//cp_size,  batch, nheads,  dim), device="cuda", dtype=torch.bfloat16).contiguous()

    # [b, 1, sq, skv_global]
    q_local.grad = None
    kv_local.grad = None
    res1 = Ref.apply(q_local, kv_local, v_local, sparse_mask_local, attention_dropout, sm_scale, cp_pg, True)
    res1.backward(do)
    dq_ref = q_local.grad
    dkv_ref = kv_local.grad

    # [batch, kv_group, seq_len, topk]
    q_local.grad = None
    k_local.grad = None
    v_local.grad = None
    kv_local.grad = None
    res2 = AttentionFuncionWithContextParallel.apply(q_local, kv_local, indices_local, dim, topk, attention_dropout, sm_scale, cp_pg)
    res2.backward(do)
    dq = q_local.grad
    dkv = kv_local.grad

    # match the dq
    assert torch.allclose(dq_ref, dq, rtol=1e-1, atol=1e-1)
    assert torch.allclose(dkv_ref, dkv, rtol=1e-1, atol=1e-1)


# run this test: rm -rf /tmp/tilelang_cache_clean && CUDA_VISIBLE_DEVICES=4,5,6,7 TILELANG_CACHE_DIR=/tmp/tilelang_cache_clean torchrun --nproc_per_node=4 /root/tilelang/examples/deepseek_v32/test.py
test_kernel(32, 512, 512, 512 // 4, 64 // 4, 128, 128, 128, 4)