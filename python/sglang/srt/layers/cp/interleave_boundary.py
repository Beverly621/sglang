# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    attn_cp_reduce_scatter_tensor,
    is_allocation_symmetric,
)
from sglang.srt.runtime_context import get_parallel


def mixer_parallelism():
    """A sequence mixer uses CP ranks as head-TP ranks, also during decode."""
    parallel = get_parallel()
    over_cp = (
        parallel.enable_prefill_cp
        and parallel.attn_cp_size > 1
        and parallel.cp_strategy == "interleave"
    )
    if over_cp:
        assert parallel.attn_tp_size == 1
        return parallel.attn_cp_size, parallel.attn_cp_rank, True
    return parallel.attn_tp_size, parallel.attn_tp_rank, False


def is_interleave_extend(forward_batch):
    from sglang.srt.layers.cp.interleave import InterleaveContextParallelMetadata

    return forward_batch.forward_mode.is_context_parallel_extend() and isinstance(
        forward_batch.attn_cp_metadata, InterleaveContextParallelMetadata
    )


def validate_mixer_batch(forward_batch):
    # The shared CP runner enters the language model after token embedding;
    # it does not run the model's multimodal embedding routine.
    if is_interleave_extend(forward_batch) and forward_batch.contains_mm_inputs():
        raise ValueError(
            "Interleave CP with sequence mixers currently supports text-only input."
        )


def mixer_to_sequence_order(hidden_states, forward_batch):
    """Undo the boundary's rank-major gather before sequence-dependent compute.

    The residual and mHC coefficients do not move: only the already-read mixer
    input is reordered. Remove physical CP padding before the recurrent kernel.
    """
    if not is_interleave_extend(forward_batch):
        return hidden_states
    metadata = forward_batch.attn_cp_metadata
    if metadata.gather_index is not None:
        return hidden_states.index_select(0, metadata.gather_index)
    size = get_parallel().attn_cp_size
    rows = hidden_states.shape[0] // size
    return (
        hidden_states.reshape(size, rows, *hidden_states.shape[1:])
        .transpose(0, 1)
        .flatten(0, 1)[: metadata.total_seq_lens]
        .contiguous()
    )


def mixer_to_rank_order(hidden_states, forward_batch, gathered_rows):
    """Restore the declared output rows, ready for the boundary's CP sum."""
    if not is_interleave_extend(forward_batch):
        return hidden_states
    metadata = forward_batch.attn_cp_metadata
    padded = hidden_states.new_zeros((gathered_rows, *hidden_states.shape[1:]))
    if metadata.gather_index is not None:
        return padded.index_copy_(0, metadata.gather_index, hidden_states)
    padded[: hidden_states.shape[0]] = hidden_states
    size = get_parallel().attn_cp_size
    return (
        padded.reshape(-1, size, *hidden_states.shape[1:])
        .transpose(0, 1)
        .flatten(0, 1)
        .contiguous()
    )


def attn_cp_gather(hidden_states: torch.Tensor):
    """Gather equal interleave shards in rank order for token-local FFNs.

    Size from the actual shard: the DP scratch length can already describe a
    shard when dense FFNs run over TP, and is not a CP collective's output size.
    """
    parallel = get_parallel()
    with use_symmetric_memory(
        parallel.attn_cp_group, disabled=not is_allocation_symmetric()
    ):
        gathered = hidden_states.new_empty(
            (hidden_states.shape[0] * parallel.attn_cp_size, *hidden_states.shape[1:])
        )
    attn_cp_all_gather_into_tensor(gathered, hidden_states.contiguous())
    return gathered


def attn_cp_reduce_scatter(hidden_states: torch.Tensor):
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    input_hidden_states = hidden_states
    hidden_states = hidden_states.tensor_split(cp_size)[cp_rank]
    attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)
    return hidden_states
