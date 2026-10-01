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
