# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Some of this code was adopted from https://github.com/zhuzilin/ring-flash-attention/
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
from torch.nn import functional as F
from sparse_mla_fwd import sparse_mla_fwd_interface
from sparse_mla_bwd import sparse_mla_bwd

try:
    import einops

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False


class AllGatherComm:
    """All gather communication with async operations"""

    def __init__(self, group=None) -> None:
        self.group = group
        self.handles = []

    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        '''All gather the input tensor to the output tensor'''

        if self.group is None:
            output_tensor.copy_(input_tensor)
        else:
            handle = torch.distributed.all_gather_into_tensor(
                output_tensor, input_tensor, group=self.group, async_op=True
            )
            self.handles.append(handle)

    def wait(self):
        '''Wait for all gather operations to complete'''

        if self.group is not None:
            for handle in self.handles:
                handle.wait()
            self.handles = []

class AttentionFuncionWithContextParallel(torch.autograd.Function):
    """Native attention function with context parallelism."""

    # q: [seq_len_shard, batch, nheads, dim + tail_dim]
    # kv: [seq_len_kv_shard, batch, kv_group, dim + tail_dim]
    #   k: [seq_len_kv_shard, batch, kv_group, dim + tail_dim]
    #   v: [seq_len_kv_shard, batch, kv_group, dim]
    # indices: [batch, kv_group, seq_len, topk]
    @staticmethod
    def forward(ctx, q, kv, indices, K, attention_dropout, softmax_scale, pg):
        '''Forward pass for the native attention function with context parallelism'''

        # Assert einops exists
        if not HAVE_EINOPS:
            raise ImportError("einops is required by the attention CP but cannot be imported.")

        # Initialize communication group and constants
        cp_size = 1
        if pg is not None:
            cp_size = torch.distributed.get_world_size(pg)
        comm = AllGatherComm(group=pg)
        nheads = q.shape[2]
        kv_group = kv.shape[2]
        heads_kv_stride = 1
        assert nheads % kv_group == 0 and kv_group % heads_kv_stride == 0
        outs = []
        lses = []

        # Initialize KV buffers
        kv_buffer = torch.empty(
            (kv.shape[0] * cp_size, kv.shape[1], heads_kv_stride, kv.shape[3]),
            dtype=kv.dtype,
            device=kv.device,
        )
        kv_buffer_copy = torch.empty_like(kv_buffer)

        # All-gather first chunk of KV buffers
        kv_0 = kv[:, :, :heads_kv_stride].contiguous()
        comm.all_gather(kv_buffer_copy, kv_0)

        # Prepare topk
        zz_indices = indices.transpose(1, 2)
        
        # Iterate over heads, sequential, i
        for i in range(0, kv_group, heads_kv_stride):
            # Wait for previous all-gather to complete
            comm.wait()
            kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer
            # All-gather the next portion of KV buffers if not the last iteration
            if i < kv_group - heads_kv_stride:
                kvsl = i + heads_kv_stride
                kvsr = kvsl + heads_kv_stride
                send_kv = kv[:, :, kvsl:kvsr].contiguous()
                comm.all_gather(kv_buffer_copy, send_kv)

            # Prepare query, key, value for attention
            q_i = q[:, :, i * nheads // kv_group : (i + heads_kv_stride) * nheads // kv_group]
            kv_i = kv_buffer

            # Rearrange query, key, value to (b, s, h, d)
            q_i = einops.rearrange(q_i, 's b h d -> b s h d')
            kv_i = einops.rearrange(kv_i, 's b h d -> b s h d')
            zz_indices_i = zz_indices[:, :, i:(i+heads_kv_stride)].contiguous()

            # Forward pass
            out_i, lse_i = sparse_mla_fwd_interface(q_i.contiguous(), kv_i.contiguous(), zz_indices_i, sm_scale = softmax_scale)

            outs.append(out_i.contiguous())
            lses.append(lse_i.contiguous())

        # out: [B, seq_len_shard, h, dim] -> [seq_len, B, h, dim]
        out = torch.cat(outs, dim=2)
        out = einops.rearrange(out, 'b s h d -> s b h d')

        # Save contexts for backward pass
        # outs: [[B, seq_len_shard, nheads // kv_group, dim], ...., [B, seq_len_shard, nheads // kv_group, dim]], repeat kv_group // heads_kv_stride times
        # lses: [[B, seq_len_shard, heads_kv_stride], ...., [B, seq_len_shard, heads_kv_stride]], repeat kv_group // heads_kv_stride times
        ctx.save_for_backward(q, kv, indices, *outs, *lses)
        ctx.K = K
        ctx.dropout = attention_dropout
        ctx.softmax_scale = softmax_scale
        ctx.heads_kv_stride = heads_kv_stride  # TODO make it configurable
        ctx.pg = pg

        return out

    @staticmethod
    def backward(ctx, dout):
        '''Backward pass for the native attention function with context parallelism'''

        # Initialize or resume constants and communication group
        q, kv, indices, *rest = ctx.saved_tensors
        K = ctx.K
        nheads = q.shape[2]
        kv_group = kv.shape[2]
        heads_kv_stride = ctx.heads_kv_stride
        softmax_scale = ctx.softmax_scale
        assert kv_group % heads_kv_stride == 0

        outs = rest[: kv_group // heads_kv_stride]
        lses = rest[kv_group // heads_kv_stride :]

        pg = ctx.pg
        cp_size = 1
        if pg is not None:
            cp_size = torch.distributed.get_world_size(pg)
        comm = AllGatherComm(group=pg)

        # Initialize KV buffers
        kv_buffer = torch.empty(
            (kv.shape[0] * cp_size, kv.shape[1], heads_kv_stride, kv.shape[3]),
            dtype=kv.dtype,
            device=kv.device,
        )
        kv_buffer_copy = torch.empty_like(kv_buffer)

        # All-gather first chunk of KV buffers
        dq = []
        dkv = []
        kv_0 = kv[:, :, :heads_kv_stride].contiguous()
        comm.all_gather(kv_buffer_copy, kv_0)

        # Prepare topk
        zz_indices = indices.transpose(1, 2)

        # Iterate over heads
        for i in range(0, kv_group, heads_kv_stride):
            # Slice query and output for this iteration
            q_slice = slice(i * nheads // kv_group, (i + heads_kv_stride) * nheads // kv_group)
            q_i = q[:, :, q_slice]
            dout_i = dout[:, :, q_slice]

            # Wait for previous all-gather to complete
            comm.wait()
            kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

            # All-gather the next portion of KV buffers if not the last iteration
            if i < kv_group - heads_kv_stride:
                kvsl = i + heads_kv_stride
                kvsr = kvsl + heads_kv_stride
                send_kv = kv[:, :, kvsl:kvsr].contiguous()
                comm.all_gather(kv_buffer_copy, send_kv)

            # Prepare key, value for attention
            kv_i = kv_buffer

            # Rearrange query, key, value to (b, s, h, d)
            q_i = einops.rearrange(q_i, 's b h d -> b s h d')
            kv_i = einops.rearrange(kv_i, 's b h d -> b s h d')
            dout_i = einops.rearrange(dout_i, 's b h d -> b s h d')
            zz_indices_i = zz_indices[:, :, i:(i+heads_kv_stride)].contiguous()

            # Backward pass
            # TODO: needs casual = True, may not be compatible with zz
            dq_i, _dkv_i = sparse_mla_bwd(q_i.contiguous(), kv_i.contiguous(), outs[i], dout_i.contiguous(), zz_indices_i, lses[i], softmax_scale, True)

            # Rearrange gradients to (s, b, h, d)
            dq_i = einops.rearrange(dq_i, 'b s h d -> s b h d')
            _dkv_i = einops.rearrange(_dkv_i, 'b s h d -> s b h d')
            if pg is None:
                dkv_i = _dkv_i
            else:
                # Reduce-scatter gradients if CP > 1
                dkv_i = torch.zeros(
                    (kv_i.shape[1] // cp_size, kv_i.shape[0], kv_i.shape[2], kv_i.shape[3]),
                    device=kv_i.device,
                    dtype=kv_i.dtype,
                )
                torch.distributed.reduce_scatter_tensor(dkv_i, _dkv_i, group=pg)

            # Collect gradients
            dq.append(dq_i)
            dkv.append(dkv_i)

        # Concatenate gradients and return
        dq = torch.cat(dq, dim=2)
        dkv = torch.cat(dkv, dim=2)
        return dq, dkv, None, None, None, None, None



