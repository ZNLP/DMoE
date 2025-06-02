# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
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
""" Qwen2moe model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)

# GEMMOE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
#     "Crystalcareai/GemMoE-Beta-1": "https://huggingface.co/Crystalcareai/GemMoE-Beta-1/resolve/main/config.json",
# }


class Qwen2moeDynConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GemmoeModel`]. It is used to instantiate a Gemmoe
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Gemmoe-7B.

    e.g. [mhenrichsen/gemmoe-7b](https://huggingface.co/mhenrichsen/gemmoe-7b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Gemmoe model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GemmoeModel`]
        hidden_size (`int`, *optional*, defaults to 3072):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 24576):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to 
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 256):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder.  
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):  
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.  
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`): 
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        query_pre_attn_scalar (`float`, *optional*, defaults to 224): scaling factor used on the attention scores
        sliding_window (`int`, *optional*, defaults to 4096): in Gemma2, every other layer uses sliding window attention. This is the
            size of the sliding window.
        final_logit_softcapping (`float`, *optional*, defaults to 30.0): scaling factor when applying tanh softcapping on the logits.
        attn_logit_softcapping (`float`, *optional*, defaults to 50.0): scaling factor when applying tanh softcapping on the attention scores.
        cache_implementation (`str`, *optional*, defaults to `"hybrid"`): the cache type to be used with `generate`.
        moe_layer_index (`int`, defaults set to 0 [all dense layer]. 2**(num_hidden_layers)-1 refer to all moe layer.)
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            The number of experts used in the sparse mixture of experts layer.
        num_local_experts (`int`, *optional*, defaults to 8):  
            The number of local experts used in the sparse mixture of experts layer.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.01):
            The coefficient for the auxiliary loss of the router.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not to output the logits of the routers. They are useful for computing the router loss, and
            should not be returned during inference.

    ```python
    >>> from transformers import GemmoeModel, GemmoeConfig

    >>> # Initializing a Gemmoe gemmoe-7b style configuration
    >>> configuration = GemmoeConfig()

    >>> # Initializing a model from the gemmoe-7b style configuration
    >>> model = GemmoeModel(configuration)

    >>> # Accessing the model configuration 
    >>> configuration = model.config
    ```"""

    model_type = "qwen2moedyn"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        decoder_sparse_step=1,
        moe_intermediate_size=1408,
        shared_expert_intermediate_size=5632,
        num_experts_per_tok=4,
        num_experts=60,
        moe_layer_index=0,
        norm_topk_prob=False,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers
        self.moe_layer_index = moe_layer_index

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )