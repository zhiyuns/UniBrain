# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from data.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    patchify, 
)
from .qwen2_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.cache_utils.taylorseer import cache_init

from tqdm import tqdm


class BagelConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        freeze_gen=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift
        self.freeze_gen = freeze_gen


class Bagel(PreTrainedModel):
    config_class = BagelConfig
    base_model_prefix = 'bagel'

    def __init__(self, language_model, vit_model, config: BagelConfig):
        super().__init__(config)    
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def forward(self, self_forcing=False, **kwargs) -> Dict[str, Optional[torch.Tensor]]:
        if self_forcing:
            return self.self_forcing_forward(**kwargs)
        else:
            return self.forward_normal(**kwargs)
    
    def forward_normal(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and 
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
        """
        if hasattr(self.language_model, "get_base_model"):
            base_lm = self.language_model.get_base_model()
            packed_text_embedding = base_lm.model.embed_tokens(packed_text_ids)
        else:
            packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if self.config.visual_und:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens, 
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        if self.config.visual_gen:
            p = self.latent_patch_size
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
            packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        mse = None
        if self.config.visual_gen and mse_loss_indexes is not None:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
            target = noise - packed_latent_clean # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
            has_mse = packed_timesteps > 0
            mse = (packed_mse_preds - target[has_mse]) ** 2

        ce = None
        if ce_loss_indexes is not None:
            if hasattr(self.language_model, "get_base_model"):
                base_lm = self.language_model.get_base_model()
                packed_ce_preds = base_lm.lm_head(last_hidden_state[ce_loss_indexes])
            else:
                packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        return dict(mse=mse, ce=ce)

    def init_contexts(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }

        self.gen_context = gen_context
    
    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()
    
    def self_forcing_forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
        **kwargs
    ) -> dict:
        total_mse_loss = []
        device = packed_text_ids.device
        
        # =====================================================================
        # 1. UPFRONT PRE-EMBEDDING (Solves Text/ViT Chunk FSDP Deadlocks)
        # =====================================================================
        
        if hasattr(self.language_model, "get_base_model"):
            base_lm = self.language_model.get_base_model()
            packed_text_embedding = base_lm.model.embed_tokens(packed_text_ids)
        else:
            packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
            
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if self.config.visual_und and packed_vit_tokens is not None:
             with torch.no_grad():
                cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)).to(torch.int32)
                max_seqlen = torch.max(vit_token_seqlens).item()
                packed_vit_token_embed = self.vit_model(
                    packed_pixel_values=packed_vit_tokens, 
                    packed_flattened_position_ids=packed_vit_position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
                packed_vit_token_embed = self.connector(packed_vit_token_embed)
                vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
                packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
                packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        # =====================================================================
        # 2. SYNCHRONIZE GLOBAL COUNTS AND EXIT STEPS
        # =====================================================================
        local_vae_count = len(padded_latent) if padded_latent is not None else 0
        
        # Find the maximum number of VAE images any GPU has in this batch
        max_vae_tensor = torch.tensor([local_vae_count], device=device, dtype=torch.long)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(max_vae_tensor, op=torch.distributed.ReduceOp.MAX)
        global_max_vae = max_vae_tensor.item()

        local_num_samples = len(sample_lens)
        max_samples_tensor = torch.tensor([local_num_samples], device=device, dtype=torch.long)
        if dist.is_initialized():
            dist.all_reduce(max_samples_tensor, op=dist.ReduceOp.MAX)
        global_max_samples = max_samples_tensor.item()

        # Determine which VAE belongs to which sample
        sample_starts = [0] + list(torch.cumsum(torch.tensor(sample_lens), dim=0).cpu().numpy())
        local_vaes_per_sample = [0] * global_max_samples
        local_vae_lists = [[] for _ in range(global_max_samples)]
        
        temp_vae_cursor = 0
        temp_token_cursor = 0

        for s_idx in range(local_num_samples):
            s_end = sample_starts[s_idx + 1]
            while temp_vae_cursor < (len(padded_latent) if padded_latent is not None else 0):
                start_token_idx = packed_vae_token_indexes[temp_token_cursor].item()
                if start_token_idx < s_end:
                    local_vaes_per_sample[s_idx] += 1
                    local_vae_lists[s_idx].append(temp_vae_cursor)
                    h, w = patchified_vae_latent_shapes[temp_vae_cursor]
                    temp_vae_cursor += 1
                    temp_token_cursor += h * w
                else:
                    break

        # Find the max number of VAEs PER SAMPLE across all ranks
        counts_tensor = torch.tensor(local_vaes_per_sample, device=device, dtype=torch.long)
        if dist.is_initialized():
            dist.all_reduce(counts_tensor, op=dist.ReduceOp.MAX)
        global_max_vae_per_sample = counts_tensor.tolist()
        total_global_vaes = sum(global_max_vae_per_sample)
        
        num_steps = 1
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)[:-1]
        
        exit_steps_flat = self.generate_and_sync_list(num_blocks=total_global_vaes, num_denoising_steps=num_steps, device=device)

        # print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: local_vae_count={local_vae_count}, global_max_vae={global_max_vae}, exit_steps={exit_steps_flat}")
        # =====================================================================
        # 3. AUTOREGRESSIVE SELF-FORCING LOOP (Padded for FSDP Sync)
        # =====================================================================
        generated_latents_list = []
        p = self.latent_patch_size
        global_vae_token_cursor = 0 
        prev_processed_idx = 0
        exit_step_cursor = 0

        # Helper to snapshot the cache for Gradient Checkpointing
        def clone_cache_snapshot(cache: NaiveCache) -> NaiveCache:
            if cache is None: return None
            snapshot = NaiveCache(cache.num_layers)
            for l_idx in range(cache.num_layers):
                if l_idx in cache.key_cache and cache.key_cache[l_idx] is not None:
                    # --- CRITICAL FIX: Use .clone() to prevent in-place mutation ---
                    # This ensures MoE routing is 100% deterministic during the backward pass!
                    snapshot.key_cache[l_idx] = cache.key_cache[l_idx].clone()
                    snapshot.value_cache[l_idx] = cache.value_cache[l_idx].clone()
            return snapshot
        
        dummy_query_indexes = torch.zeros(1, dtype=torch.long, device=device)
        dummy_kv_indexes = torch.empty(0, dtype=torch.long, device=device)
        dummy_kv_lens_tensor = torch.tensor([0], dtype=torch.int32, device=device)
        dummy_x_t = torch.zeros((1, p * p * self.latent_channel), device=device, dtype=packed_sequence.dtype)
        dummy_pos = torch.zeros((1,), device=device, dtype=torch.long)
        dummy_seq = torch.zeros((1, self.hidden_size), device=device, dtype=packed_sequence.dtype)
        
        # --- OUTER LOOP: ITERATE OVER SAMPLES ---
        for s_idx in range(global_max_samples):
            is_real_sample = s_idx < local_num_samples
            s_start = sample_starts[s_idx] if is_real_sample else prev_processed_idx
            s_end = sample_starts[s_idx + 1] if is_real_sample else prev_processed_idx
            
            past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
            curr_kv_len = 0
            prev_processed_idx = s_start
            
            # --- FIX: SELF-FORCING STATE MACHINE ---
            # Holds the prediction from a Noisy VAE to inject into the subsequent Clean VAE
            last_generated_image = None
        
        
            # Helper to process context gaps cleanly using the pre-embedded sequence
            def update_context_cache(target_idx):
                nonlocal prev_processed_idx, curr_kv_len, past_key_values
                gap_len = target_idx - prev_processed_idx
                
                self.language_model.eval()
                with torch.no_grad():
                    if gap_len > 0:
                        # REAL CALL
                        gap_seq = packed_sequence[prev_processed_idx : target_idx]
                        gap_pos = packed_position_ids[prev_processed_idx : target_idx]
                        
                        out = self.language_model(
                            packed_sequence=gap_seq,
                            sample_lens=[gap_len],
                            packed_position_ids=gap_pos,
                            past_key_values=past_key_values,
                            packed_query_indexes=torch.arange(curr_kv_len, curr_kv_len + gap_len, dtype=torch.long, device=device),
                            packed_key_value_indexes=torch.arange(curr_kv_len, dtype=torch.long, device=device),
                            key_values_lens=torch.tensor([curr_kv_len], dtype=torch.int32, device=device),
                            update_past_key_values=True,
                            use_cache=True,
                            is_causal=True
                        )
                        past_key_values = out.past_key_values if hasattr(out, 'past_key_values') else out[1]
                        curr_kv_len += gap_len
                    else:
                        # DUMMY CALL (Keeps FSDP synced if another rank is doing a real call)
                        
                        self.language_model(
                            packed_sequence=dummy_seq,
                            sample_lens=[1],
                            packed_position_ids=dummy_pos,
                            past_key_values=None,
                            # --- FIX: Empty Cache Pointers ---
                            packed_key_value_indexes=dummy_kv_indexes,
                            key_values_lens=dummy_kv_lens_tensor,
                            packed_query_indexes=dummy_query_indexes,
                            update_past_key_values=False, # Protect cache
                            use_cache=True,
                            is_causal=True
                        )
                prev_processed_idx = target_idx

            local_vae_count_for_this_sample = len(local_vae_lists[s_idx]) if is_real_sample else 0
            generated_sequence = []
            true_sequence = []
            # --- THE SYNCHRONIZED LOOP ---
            for v_idx in range(global_max_vae_per_sample[s_idx]):
                is_real_vae = is_real_sample and (v_idx < local_vae_count_for_this_sample)
                exit_step_idx = exit_steps_flat[exit_step_cursor]
                exit_step_cursor += 1

                if is_real_vae:
                    # Prepare Real Data
                    real_vae_global_idx = local_vae_lists[s_idx][v_idx]
                    target_latent = padded_latent[real_vae_global_idx]
                    
                    h, w = patchified_vae_latent_shapes[real_vae_global_idx]
                    target_latent = target_latent[:, :h * p, :w * p]
                    # x_t = torch.randn_like(target_latent) 
                    noise = torch.randn_like(target_latent) 

                    # Jitter start time between e.g. 0.9 and 1.0
                    max_jitter = 1.0 / num_steps
                    t_start = 1.0 - (torch.rand(1, device=device).item() * max_jitter)
                    
                    timesteps = torch.linspace(t_start, 0.0, num_steps + 1, device=device)[:-1]
                    next_timesteps = torch.linspace(t_start, 0.0, num_steps + 1, device=device)[1:]
                    dt_schedule = timesteps - next_timesteps # Exact dt size for Euler steps
                    x_t = (1.0 - t_start) * target_latent + t_start * noise

                    num_tokens = h * w
                    
                    cur_vae_start_idx = packed_vae_token_indexes[global_vae_token_cursor].item()

                    # --- BIDIRECTIONAL TEXT TAG FIX ---
                    # The text token immediately before VAE is S. The one immediately after is E.
                    S_idx = cur_vae_start_idx - 1
                    E_idx = cur_vae_start_idx + num_tokens

                    # 1. Stop the KV cache BEFORE the start token! 
                    update_context_cache(S_idx)

                    # 2. Extract static embeddings for S and E to pass into diffusion
                    S_emb = packed_sequence[S_idx : S_idx + 1]
                    E_emb = packed_sequence[E_idx : E_idx + 1]
                    
                    # 3. RoPE pos IDs for the whole block [S, VAE..., E]
                    rope_pos_ids = packed_position_ids[S_idx : E_idx + 1]
                    dummy_rope_pos_ids = torch.zeros((3,), device=device, dtype=torch.long)

                    raw_timesteps = packed_timesteps[global_vae_token_cursor : global_vae_token_cursor + num_tokens]
                    sig_timesteps = torch.sigmoid(raw_timesteps)
                    is_input_image = (sig_timesteps < 1e-4).all().item()
                    
                    img_pos_ids = packed_latent_position_ids[global_vae_token_cursor : global_vae_token_cursor + num_tokens]
                    
                    latent_token_pos_emb = self.latent_pos_embed(img_pos_ids)
                    dummy_latent_token_pos_emb = latent_token_pos_emb[:1]
                    
                else:
                    # Prepare Dummy Data
                    update_context_cache(prev_processed_idx)
                    num_tokens = 1
                    
                    h, w = 1, 1 
                    x_t = None
                    target_latent = None
                    noise = None 
                    
                    # Dummy Sequence of length 3: [S, VAE, E]
                    S_emb = torch.zeros((1, self.hidden_size), device=device, dtype=packed_sequence.dtype)
                    E_emb = torch.zeros((1, self.hidden_size), device=device, dtype=packed_sequence.dtype)
                    dummy_rope_pos_ids = torch.zeros((3,), device=device, dtype=torch.long)
                    
                    img_pos_ids = torch.zeros((1,), device=device, dtype=torch.long)
                    latent_token_pos_emb = self.latent_pos_embed(img_pos_ids)
                    dummy_latent_token_pos_emb = latent_token_pos_emb[:1]

                current_generated_image = None

                query_indexes = torch.arange(curr_kv_len, curr_kv_len + num_tokens + 2, device=packed_text_ids.device)
                kv_indexes = torch.arange(0, curr_kv_len, dtype=torch.long, device=packed_text_ids.device)
                kv_lens_tensor = torch.tensor([curr_kv_len], dtype=torch.int32, device=packed_text_ids.device)

                # Diffusion Steps
                # print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Starting diffusion for VAE idx {local_vae_count_for_this_sample} with exit step {exit_step_idx} and num_tokens {num_tokens}, is_real_vae={is_real_vae}")
                
                # for debugging: perfect trajectory at t=0.5
                # if is_real_vae:
                #     debug_t = 0.5
                #     perfect_x_t = (1 - debug_t) * target_latent + debug_t * noise

                #     # Format for model exactly like the loop does
                #     latent_reshaped = perfect_x_t[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                #     packed_x_t = torch.einsum("chpwq->hwpqc", latent_reshaped).reshape(-1, p * p * self.latent_channel)
                #     t_expand = torch.tensor([debug_t], device=device)# .expand(packed_x_t.shape[0])
                #     packed_timestep_embeds = self.time_embedder(t_expand)

                #     # Pass to model
                #     vae_hidden = self.vae2llm(packed_x_t) + packed_timestep_embeds + latent_token_pos_emb
                #     hidden_states = torch.cat([S_emb, vae_hidden, E_emb], dim=0)

                #     packed_vae_token_indexes_local = torch.arange(1, num_tokens + 1, dtype=torch.long, device=packed_text_ids.device)
                #     packed_text_indexes_local = torch.tensor([0, num_tokens + 1], dtype=torch.long, device=packed_text_ids.device)

                #     out = self.language_model(
                #         packed_sequence=hidden_states, 
                #         sample_lens=[num_tokens + 2],  # Length is now + 2
                #         packed_position_ids=rope_pos_ids,
                        
                #         past_key_values=past_key_values,
                #         packed_query_indexes=query_indexes,
                #         packed_key_value_indexes=kv_indexes,
                #         key_values_lens=kv_lens_tensor,
                #         update_past_key_values=True, # Allow bidirectional spatial attention
                        
                #         use_cache=True, 
                #         mode="gen",
                #         packed_vae_token_indexes=packed_vae_token_indexes_local,
                #         packed_text_indexes=packed_text_indexes_local,
                #         is_causal=False,
                #     )

                #     vel_tokens = self.llm2vae(out.packed_query_sequence)[packed_vae_token_indexes_local]
                #     vel_reshaped = torch.einsum("hwpqc->chpwq", vel_tokens.reshape(h, w, p, p, self.latent_channel)).reshape(self.latent_channel, h * p, w * p)

                #     debug_loss = torch.nn.functional.mse_loss(vel_reshaped, noise - target_latent)
                #     print(f"PERFECT TRAJECTORY LOSS at t=0.5: {debug_loss.mean().item():.4f}, RANK {dist.get_rank() if dist.is_initialized() else 0}, VAE IDX {local_vae_count_for_this_sample}, EXIT STEP {exit_step_idx}")
                
                # else:
                #     debug_t = 0.5
                #     t_expand = torch.tensor([debug_t], device=device)# .expand(packed_x_t.shape[0])
                #     dummy_time_emb = self.time_embedder(t_expand)
                #     vae_dummy_hidden = self.vae2llm(dummy_x_t) + dummy_time_emb + dummy_latent_token_pos_emb
                #     dummy_hidden = torch.cat([S_emb, vae_dummy_hidden, E_emb], dim=0)
                #     dummy_query_idxes = torch.arange(curr_kv_len, curr_kv_len + 3, device=packed_text_ids.device)
                #     # dummy forward to keep FSDP in sync
                #     self.language_model(
                #         packed_sequence=dummy_hidden,
                #         sample_lens=[1],
                #         packed_position_ids=dummy_pos,
                #         past_key_values=None,
                #         packed_key_value_indexes=dummy_kv_indexes,
                #         key_values_lens=dummy_kv_lens_tensor,
                #         packed_query_indexes=dummy_query_idxes,
                #         update_past_key_values=False, # Protect cache
                #         use_cache=True,
                #         mode="gen",
                #         packed_vae_token_indexes=torch.arange(0, 1, dtype=torch.long, device=device), # Dummy token index
                #         packed_text_indexes=None,
                #         is_causal=False
                #     )
                #     vel_tokens = self.llm2vae(dummy_hidden)
                    
                for step_idx, t_curr in enumerate(timesteps):
                    is_exit_step = (step_idx == exit_step_idx)
                    ctx = torch.set_grad_enabled(is_exit_step and not self.config.freeze_gen)
                    if is_exit_step: self.language_model.train()
                    else: self.language_model.eval()
                    
                    with ctx:
                        if is_real_vae and not is_input_image:
                            cache_for_this_step = clone_cache_snapshot(past_key_values) if is_exit_step else past_key_values
                            latent_reshaped = x_t[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                            packed_x_t = torch.einsum("chpwq->hwpqc", latent_reshaped).reshape(-1, p * p * self.latent_channel)

                            shifted_t = self.timestep_shift * t_curr / (1 + (self.timestep_shift - 1) * t_curr)
                            t_expand = shifted_t.view(1).expand(packed_x_t.shape[0])
                            packed_timestep_embeds = self.time_embedder(t_expand)
                            
                            # --- COMBINE INTO SINGLE BLOCK ---
                            vae_hidden = self.vae2llm(packed_x_t) + packed_timestep_embeds + latent_token_pos_emb
                            hidden_states = torch.cat([S_emb, vae_hidden, E_emb], dim=0)
                            
                            # Local indexes for VAE tokens are offset by 1 because of S_emb
                            packed_vae_token_indexes_local = torch.arange(1, num_tokens + 1, dtype=torch.long, device=packed_text_ids.device)
                            packed_text_indexes_local = torch.tensor([0, num_tokens + 1], dtype=torch.long, device=packed_text_ids.device)
                            
                            out = self.language_model(
                                packed_sequence=hidden_states, 
                                sample_lens=[num_tokens + 2],  # Length is now + 2
                                packed_position_ids=rope_pos_ids,
                                
                                past_key_values=cache_for_this_step,
                                packed_query_indexes=query_indexes,
                                packed_key_value_indexes=kv_indexes,
                                key_values_lens=kv_lens_tensor,
                                update_past_key_values=False,
                                
                                use_cache=True, 
                                mode="gen",
                                packed_vae_token_indexes=packed_vae_token_indexes_local,
                                packed_text_indexes=packed_text_indexes_local,
                                is_causal=False,
                            )
                            
                            last_hidden = out.packed_query_sequence[packed_vae_token_indexes_local]
                            velocity_pred_tokens = self.llm2vae(last_hidden)
                            # Target the VAE tokens specifically
                            
                            velocity_pred_reshaped = velocity_pred_tokens.reshape(h, w, p, p, self.latent_channel)
                            velocity_pred_reshaped = torch.einsum("hwpqc->chpwq", velocity_pred_reshaped)
                            velocity_pred_reshaped = velocity_pred_reshaped.reshape(self.latent_channel, h * p, w * p)
                            
                            # Pad back to original size if x_t was padded? 
                            # usually x_t is exactly h*p, w*p.
                            velocity_pred = velocity_pred_reshaped  # with shape (latent_channel, h*p, w*p)
                            
                            if is_exit_step:
                                target_velocity = noise - target_latent
                                # target_velocity = (x_t - target_latent) / t_curr
                                
                                # Now compute MSE on the velocity, exactly like original pre-training!
                                loss_gen = torch.nn.functional.mse_loss(velocity_pred, target_velocity, reduction='none')
                                
                                loss_gen = loss_gen.reshape(self.latent_channel, h, p, w, p)
                                loss_gen = torch.einsum("chpwq->hwpqc", loss_gen).reshape(-1, p * p * self.latent_channel)
                                total_mse_loss.append(loss_gen)
                                
                                # Set generated image for Cache Update using the prediction
                                # current_generated_image = x_t - t_curr * velocity_pred
                                # filter the timestep that are very small to stabilize training
                                current_generated_image = x_t - t_curr * velocity_pred
                                generated_sequence.append(current_generated_image)
                                true_sequence.append(target_latent)
                                
                                # calculate loss on x0
                                # target_trimmed = target_latent.reshape(self.latent_channel, h * p, w * p)
                                # loss_gen = torch.nn.functional.mse_loss(current_generated_image, target_trimmed, reduction='none')
                                # loss_gen = loss_gen.reshape(self.latent_channel, h, p, w, p)
                                # loss_gen = torch.einsum("chpwq->hwpqc", loss_gen).reshape(-1, p * p * self.latent_channel)
                                # total_mse_loss.append(loss_gen)

                                # vanilla loss calculation on velocity (pointing from x_t to noise)
                                # target_velocity = noise - target_latent
                                # loss_gen = torch.nn.functional.mse_loss(velocity_pred, target_velocity, reduction='none')
                                # loss_gen = loss_gen.reshape(self.latent_channel, h, p, w, p)
                                # loss_gen = torch.einsum("chpwq->hwpqc", loss_gen).reshape(-1, p * p * self.latent_channel)
                                # total_mse_loss.append(loss_gen)
                                
                                # # Set generated image for Cache Update
                                # current_generated_image = x_t - t_curr * velocity_pred
                                last_generated_image = current_generated_image.detach()
                                break
                            else:
                                dt = dt_schedule[step_idx]
                                denoised_pred = x_t - t_curr * velocity_pred
                                next_timestep = t_curr - dt
                                noise = torch.randn_like(target_latent) 
                                x_t = denoised_pred * (1 - next_timestep) + next_timestep * noise
                        else:
                            # DUMMY DIFFUSION CALL
                            t_expand = t_curr.view(1)
                            # t_expand = self.timestep_shift * t_expand / (1 + (self.timestep_shift - 1) * t_expand)
                            dummy_latent = torch.zeros((1, p * p * self.latent_channel), device=device, dtype=packed_sequence.dtype)
                            clean_timesteps = torch.zeros((1,), device=device)
                            dummy_time_emb = self.time_embedder(clean_timesteps)
                            vae_dummy_hidden = self.vae2llm(dummy_latent) + dummy_time_emb + dummy_latent_token_pos_emb
                            
                            dummy_hidden = torch.cat([S_emb, vae_dummy_hidden, E_emb], dim=0)
                            dummy_query_idx = torch.arange(3, dtype=torch.long, device=device)

                            out = self.language_model(
                                packed_sequence=dummy_hidden, sample_lens=[3], packed_position_ids=dummy_rope_pos_ids,
                                past_key_values=None, packed_key_value_indexes=dummy_kv_indexes,
                                key_values_lens=dummy_kv_lens_tensor, packed_query_indexes=dummy_query_idx,
                                update_past_key_values=False, use_cache=True, mode="gen",
                                is_causal=False, packed_text_indexes=torch.tensor([0, 2], dtype=torch.long, device=device), 
                                packed_vae_token_indexes=torch.arange(1, 2, dtype=torch.long, device=device)
                            )
                            dummy_vel = self.llm2vae(out.packed_query_sequence)[1:-1]
                            if is_exit_step:
                                dummy_loss = dummy_vel * 0.0
                                total_mse_loss.append(dummy_loss)
                                if is_real_vae and is_input_image:
                                    current_generated_image = target_latent
                                break

                # Cache Update for Generated VAE
                self.language_model.eval()

                # --- NEW: FSDP SYNC FOR CACHE UPDATE ---
                # We only want to update the cache for Input Images (Clean VAEs).
                # Noisy VAEs (Generation Targets) should NEVER be appended to the KV cache!
                local_needs_cache = 1 if (is_real_vae and is_input_image) else 0
                needs_cache_tensor = torch.tensor([local_needs_cache], device=device, dtype=torch.long)
                if dist.is_initialized():
                    dist.all_reduce(needs_cache_tensor, op=dist.ReduceOp.MAX)
                global_needs_cache = needs_cache_tensor.item() > 0

                with torch.no_grad():
                    if is_real_vae:
                        if current_generated_image is None: current_generated_image = x_t
                        current_generated_image = current_generated_image.detach()

                        if is_input_image:
                            if last_generated_image is not None:
                                # This is a Target Context Image! 
                                # Overwrite the Ground Truth with our generated prediction!
                                current_generated_image = last_generated_image
                                last_generated_image = None # Consume it
                            else:
                                # This is a True Input Image (no preceding Noisy VAE).
                                # Leave current_generated_image as the Ground Truth target_latent.
                                pass

                        generated_latents_list.append(current_generated_image)
                        
                        global_vae_token_cursor += num_tokens
                        # Advance prev_processed_idx PAST the E_idx! 
                        # This tells the next update_context_cache to resume at the text AFTER the image.
                        prev_processed_idx = E_idx + 1
                        
                        if global_needs_cache:
                            if is_input_image:
                                latent_reshaped = current_generated_image[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                                flat_generated_latent = torch.einsum("chpwq->hwpqc", latent_reshaped).reshape(-1, p * p * self.latent_channel)
                                
                                clean_timesteps = torch.zeros((num_tokens,), device=device)
                                packed_timestep_embeds = self.time_embedder(clean_timesteps)
                                
                                # --- COMBINE CLEAN VAE INTO SINGLE BLOCK ---
                                vae_clean_hidden = self.vae2llm(flat_generated_latent) + packed_timestep_embeds + latent_token_pos_emb
                                final_hidden = torch.cat([S_emb, vae_clean_hidden, E_emb], dim=0)
                                
                                packed_vae_token_indexes_local = torch.arange(1, num_tokens + 1, dtype=torch.long, device=packed_text_ids.device)
                                packed_text_indexes_local = torch.tensor([0, num_tokens + 1], dtype=torch.long, device=packed_text_ids.device)

                                out = self.language_model(
                                    packed_sequence=final_hidden,
                                    sample_lens=[num_tokens + 2],
                                    packed_position_ids=rope_pos_ids,
                                    past_key_values=past_key_values,
                                    packed_query_indexes=query_indexes,
                                    packed_key_value_indexes=kv_indexes,
                                    key_values_lens=kv_lens_tensor,
                                    update_past_key_values=True,
                                    use_cache=True,
                                    mode="gen",
                                    is_causal=False,
                                    packed_text_indexes=packed_text_indexes_local,
                                    packed_vae_token_indexes=packed_vae_token_indexes_local
                                )
                                past_key_values = out.past_key_values if hasattr(out, 'past_key_values') else out[1]
                                
                                # Cache advanced by num_tokens AND the S and E tokens!
                                curr_kv_len += (num_tokens + 2)
                            else:
                                # Dummy call exactly matching length 3
                                dummy_latent = torch.zeros((1, p * p * self.latent_channel), device=device, dtype=packed_sequence.dtype)
                                clean_timesteps = torch.zeros((1,), device=device)
                                shifted_t = self.timestep_shift * t_curr / (1 + (self.timestep_shift - 1) * t_curr)
                                t_expand = shifted_t.view(1)
                                dummy_time_emb = self.time_embedder(t_expand)
                                vae_dummy_hidden = self.vae2llm(dummy_latent) + dummy_time_emb + dummy_latent_token_pos_emb
                                
                                dummy_hidden = torch.cat([S_emb, vae_dummy_hidden, E_emb], dim=0)
                                dummy_query_idx = torch.arange(3, dtype=torch.long, device=device)

                                self.language_model(
                                    packed_sequence=dummy_hidden, sample_lens=[3], packed_position_ids=dummy_rope_pos_ids,
                                    past_key_values=None, packed_key_value_indexes=dummy_kv_indexes,
                                    key_values_lens=dummy_kv_lens_tensor, packed_query_indexes=dummy_query_idx,
                                    update_past_key_values=False, use_cache=True, mode="gen",
                                    is_causal=False, packed_text_indexes=torch.tensor([0, 2], dtype=torch.long, device=device),
                                    packed_vae_token_indexes=torch.arange(1, 2, dtype=torch.long, device=device)
                                )
                    else:
                        # Dummy Rank Logic
                        if global_needs_cache:
                            # Same exact Length 3 dummy logic as above
                            dummy_latent = torch.zeros((1, p * p * self.latent_channel), device=device, dtype=packed_sequence.dtype)
                            clean_timesteps = torch.zeros((1,), device=device)
                            dummy_time_emb = self.time_embedder(clean_timesteps)
                            vae_dummy_hidden = self.vae2llm(dummy_latent) + dummy_time_emb + dummy_latent_token_pos_emb
                            
                            dummy_hidden = torch.cat([S_emb, vae_dummy_hidden, E_emb], dim=0)
                            dummy_query_idx = torch.arange(3, dtype=torch.long, device=device)

                            self.language_model(
                                packed_sequence=dummy_hidden, sample_lens=[3], packed_position_ids=dummy_rope_pos_ids,
                                past_key_values=None, packed_key_value_indexes=dummy_kv_indexes,
                                key_values_lens=dummy_kv_lens_tensor, packed_query_indexes=dummy_query_idx,
                                update_past_key_values=False, use_cache=True, mode="gen",
                                is_causal=False, packed_text_indexes=torch.tensor([0, 2], dtype=torch.long, device=device), 
                                packed_vae_token_indexes=torch.arange(1, 2, dtype=torch.long, device=device)
                            )

            # --- POST-VAE: Process the remaining trailing text for this specific sample ---
            # update_context_cache(s_end)
            
            # --- MEMORY RELEASE: Destroy this sample's KV Cache! ---
            del past_key_values

        # =====================================================================
        # 4. POST-LOOP: CROSS-ENTROPY LOSS CALCULATION
        # =====================================================================
        # Because we already embedded Text and ViT into packed_sequence at the top,
        # we only need to re-embed the GENERATED VAE images back into it.

        # print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Finished all VAEs. Now preparing for CE loss calculation with total_mse_loss count {len(total_mse_loss)} and generated_latents_list count {len(generated_latents_list)}.")

        if len(generated_latents_list) > 0:
            packed_latent_list = []
            for latent, (h, w) in zip(generated_latents_list, patchified_vae_latent_shapes):
                latent_reshaped = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                flat_latent = torch.einsum("chpwq->hwpqc", latent_reshaped).reshape(-1, p * p * self.latent_channel)
                packed_latent_list.append(flat_latent)
            
            packed_latent = torch.cat(packed_latent_list, dim=0)
            packed_timesteps_clean = torch.zeros((packed_latent.shape[0],), device=device)
            packed_timestep_embeds = self.time_embedder(packed_timesteps_clean)
            packed_pos_embed = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
            packed_sequence[packed_vae_token_indexes] = packed_latent

        # --- CRITICAL FIX: ALL RANKS EXECUTE THIS FORWARD PASS ---
        # Do not hide this inside an 'if ce_loss_indexes is not None' block!
        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks
        
        # Standard parallel training forward
        self.language_model.train()
        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask, # Ensure this is prepared
            packed_position_ids=packed_position_ids,
            use_cache=False,
            **extra_inputs,
        )

        # Safely compute CE loss
        ce_loss = None
        has_ce_targets = (ce_loss_indexes is not None) and (len(ce_loss_indexes) > 0)
        
        if has_ce_targets:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce_loss = torch.nn.functional.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")
        else:
            # Dummy loss to prevent hang during gradient reduction
            dummy_pred = self.language_model.lm_head(last_hidden_state[0:1]) 
            ce_loss = (dummy_pred.sum() * 0.0).view(-1)

        final_mse = torch.cat(total_mse_loss, dim=0) if len(total_mse_loss) > 0 else (packed_sequence.sum() * 0.0).view(-1)

        # print(f"Rank {dist.get_rank() if dist.is_initialized() else 0}: Final MSE shape {final_mse.shape}, CE loss shape {ce_loss.shape}")

        return dict(mse=final_mse if not self.config.freeze_gen else None, ce=ce_loss)

    
    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids, device='cuda'):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        for k, v in generation_input.items():
            generation_input[k] = v.to(device)

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        if hasattr(self.language_model, "get_base_model"):
            base_lm = self.language_model.get_base_model()
            packed_text_embedding = base_lm.model.embed_tokens(packed_text_ids)
        else:
            packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        if hasattr(self.language_model, "get_base_model"):
            language_model = self.language_model.get_base_model()
        else:
            language_model = self.language_model
        output = language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, device='cuda'):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2), 
                self.vit_patch_size, 
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        for k, v in generation_input.items():
            generation_input[k] = v.to(device)

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
        # OPTIONAL ARGS for text wrappers
        packed_text_ids: Optional[torch.LongTensor] = None,
        packed_text_indexes: Optional[torch.LongTensor] = None,
        return_packed_vit_token_embed: bool = False,
    ):
        # 1. Initialize packed_sequence with zeros
        packed_sequence = packed_vit_tokens.new_zeros((sum(packed_seqlens), self.hidden_size))
        if packed_text_ids is not None and packed_text_indexes is not None and packed_text_ids.numel() > 0:
            if hasattr(self.language_model, "get_base_model"):
                base_lm = self.language_model.get_base_model()
                packed_text_embedding = base_lm.model.embed_tokens(packed_text_ids)
            else:
                packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
            packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
            packed_sequence[packed_text_indexes] = packed_text_embedding

        # 3. Process ViT Tokens
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()

        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens,
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
        
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        # 4. Forward Pass (Inference Mode)
        extra_inputs = {}
        if self.use_moe:
            # For ViT, we usually treat it as 'und' (understanding) mode
            # But we need valid indices. If text is missing, we must ensure indices are correct.
            # If text is None, valid_und_mask might fail if we pass empty tensor?
            # Let's construct safe indices.
            if packed_text_indexes is None:
                 packed_text_indexes = torch.tensor([], dtype=torch.long, device=packed_sequence.device)
            
            # Combine text (if any) and vit indices for Understanding
            packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes])
            
            extra_inputs = {
                "mode": "und", 
                # Note: In inference, Qwen2MoEDecoderLayer doesn't use these for 'und' mode usually?
                # Check Qwen2Navit.forward_inference: 
                # if mode == 'und', it just does self.mlp(x). It doesn't use indices.
                # So we might not need to pass them!
            }

        if hasattr(self.language_model, "get_base_model"):
            language_model = self.language_model.get_base_model()
        else:
            language_model = self.language_model

        output = language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        
        if return_packed_vit_token_embed:
            return output.past_key_values, packed_vit_token_embed
        return output.past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0, device='cuda'):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size), device=device)
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor.to(device)

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)

        return generation_input, newlens, new_rope

    @torch.no_grad

    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: Optional[torch.Tensor],
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
        # OPTIONAL ARGS
        packed_text_ids: Optional[torch.LongTensor] = None,
        packed_text_indexes: Optional[torch.LongTensor] = None,
        padded_latent=None,
    ):
        # 1. Initialize packed_sequence
        packed_sequence = torch.zeros((sum(packed_seqlens), self.hidden_size), device=packed_vae_position_ids.device, dtype=self.dtype)
        if padded_latent is None:
            packed_sequence = next(self.vit_model.parameters()).new_zeros((sum(packed_seqlens), self.hidden_size))
        else:
            packed_sequence = padded_latent.new_zeros((sum(packed_seqlens), self.hidden_size))

        if packed_text_ids is not None and packed_text_indexes is not None and packed_text_ids.numel() > 0:
            if hasattr(self.language_model, "get_base_model"):
                base_lm = self.language_model.get_base_model()
                packed_text_embedding = base_lm.model.embed_tokens(packed_text_ids)
            else:
                packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
            
            packed_sequence[packed_text_indexes] = packed_text_embedding

        # 3. Process VAE Latents
        if padded_latent is None:
            padded_latent = vae_model.encode(padded_images)

            p = self.latent_patch_size
            packed_latent = list()
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent = torch.cat(packed_latent, dim=0)
        else:
            packed_latent = padded_latent

        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
            
        packed_sequence[packed_vae_token_indexes] = packed_latent

        # 4. Forward Pass
        extra_inputs = {}
        if self.use_moe:
            # Ensure indices exist even if empty
            if packed_text_indexes is None:
                 packed_text_indexes = torch.tensor([], dtype=torch.long, device=packed_sequence.device)
                 
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        if hasattr(self.language_model, "get_base_model"):
            language_model = self.language_model.get_base_model()
        else:
            language_model = self.language_model
            
        output = language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )

        return output.past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids, device='cuda'):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_init_noises.append(
                torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
            )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        # for k, v in generation_input.items():
        #     generation_input[k] = v.to(device)

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes, device='cuda'):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        # for k, v in generation_input.items():
        #     generation_input[k] = v.to(device)

        return generation_input

    @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # cache_args
        enable_taylorseer=False,
    ):
        if enable_taylorseer:
            self.language_model.model.enable_taylorseer = True
            model_pred_cache_dic, model_pred_current = cache_init(self, num_timesteps)
            model_pred_text_cache_dic, model_pred_text_current = cache_init(self, num_timesteps)
            model_pred_img_cache_dic, model_pred_img_current = cache_init(self, num_timesteps)
        else:
            self.language_model.model.enable_taylorseer = False
            model_pred_cache_dic, model_pred_current = None, None
            model_pred_text_cache_dic, model_pred_text_current = None, None
            model_pred_img_cache_dic, model_pred_img_current = None, None
    
        x_t = packed_init_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts =  timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep, 
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
                # cache
                model_pred_cache_dic=model_pred_cache_dic,
                model_pred_current=model_pred_current,
                model_pred_text_cache_dic=model_pred_text_cache_dic,
                model_pred_text_current=model_pred_text_current,
                model_pred_img_cache_dic=model_pred_img_cache_dic,
                model_pred_img_current=model_pred_img_current,
            )

            x_t = x_t - v_t.to(x_t.device) * dts[i] # velocity pointing from data to noise
        
        if enable_taylorseer:
            del model_pred_cache_dic, model_pred_current
            del model_pred_text_cache_dic, model_pred_text_current
            del model_pred_img_cache_dic, model_pred_img_current

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    @torch.no_grad
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # cache
        model_pred_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_current: Optional[int] = None,
        model_pred_text_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_text_current: Optional[int] = None,
        model_pred_img_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_img_current: Optional[int] = None,
    ):
        if hasattr(self.language_model, "get_base_model"):
            base_lm = self.language_model.get_base_model()
            packed_text_embedding = base_lm.model.embed_tokens(packed_text_ids)
        else:
            packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }
        
        if self.language_model.model.enable_taylorseer:
            self.language_model.model.cache_dic = model_pred_cache_dic
            self.language_model.model.current = model_pred_current

        if hasattr(self.language_model, "get_base_model"):
            language_model = self.language_model.get_base_model()
        else:
            language_model = self.language_model
        output = language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if self.language_model.model.enable_taylorseer:
                self.language_model.model.cache_dic = model_pred_text_cache_dic
                self.language_model.model.current = model_pred_text_current
            if hasattr(self.language_model, "get_base_model"):
                language_model = self.language_model.get_base_model()
            else:
                language_model = self.language_model
            cfg_text_output = language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            if self.language_model.model.enable_taylorseer:
                self.language_model.model.cache_dic = model_pred_img_cache_dic
                self.language_model.model.current = model_pred_img_current
            if hasattr(self.language_model, "get_base_model"):
                language_model = self.language_model.get_base_model()
            else:
                language_model = self.language_model
            cfg_img_output = language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        return v_t

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids, device='cuda'):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['bos_token_id'])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        for k, v in generation_input.items():
            generation_input[k] = v.to(device)

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            if hasattr(self.language_model, "get_base_model"):
                base_lm = self.language_model.get_base_model()
                packed_text_embedding = base_lm.model.embed_tokens(curr_tokens)
            else:
                packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            if hasattr(self.language_model, "get_base_model"):
                language_model = self.language_model.get_base_model()
            else:
                language_model = self.language_model
            output = language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            if hasattr(self.language_model, "get_base_model"):
                base_lm = self.language_model.get_base_model()
                pred_logits = base_lm.lm_head(packed_query_sequence)
            else:
                pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    # for evaluation
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # add images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                images=[image], 
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        return output