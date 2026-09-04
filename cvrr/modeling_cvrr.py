"""Hugging Face compatible implementation of CVRR on Qwen2.5-VL.

The implementation intentionally contains only the released method:

1. run the frozen multimodal and text-only branches through ``ell_star``;
2. use the next native decoder layer once to obtain the multimodal initial
   question state and persistent visual scaffold;
3. reuse that layer with one shared LoRA for recurrent transitions; and
4. expose only the final question state to the remaining decoder layers.

The answer-time cache below the strict interface is built exclusively from the
text-only branch.  Persistent visual rows are used inside recurrence but are
never inserted into the answer decoder cache.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLPreTrainedModel,
    apply_multimodal_rotary_pos_emb,
)

from .configuration_cvrr import CVRRConfig


@dataclass
class _SplitContext:
    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    causal_mask: torch.Tensor | None
    cache_position: torch.Tensor
    past_key_values: Cache | None
    attention_mask: torch.Tensor | None


@dataclass
class CVRROutput(ModelOutput):
    """Output returned by :class:`CVRRForConditionalGeneration`."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    recurrent_states: tuple[torch.FloatTensor, ...] | None = None


def _make_context(
    text_model,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    past_key_values: Cache | None = None,
    cache_position: torch.Tensor | None = None,
) -> _SplitContext:
    if cache_position is None:
        seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            seen,
            seen + hidden_states.shape[1],
            device=hidden_states.device,
        )
    if position_ids.ndim == 2:
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    mask = create_causal_mask(
        config=text_model.config,
        input_embeds=hidden_states,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=None,
    )
    return _SplitContext(
        hidden_states=hidden_states,
        position_ids=position_ids,
        position_embeddings=text_model.rotary_emb(hidden_states, position_ids),
        causal_mask=mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
    )


def _run_layers(
    text_model,
    context: _SplitContext,
    start: int,
    stop: int | None = None,
    *,
    hidden_states: torch.Tensor | None = None,
    use_cache: bool = False,
) -> torch.Tensor:
    stop = len(text_model.layers) if stop is None else stop
    state = context.hidden_states if hidden_states is None else hidden_states
    for layer in text_model.layers[start:stop]:
        state = layer(
            state,
            attention_mask=context.causal_mask,
            position_ids=None,
            past_key_values=context.past_key_values,
            use_cache=use_cache,
            cache_position=context.cache_position,
            position_embeddings=context.position_embeddings,
        )
        if isinstance(state, tuple):
            state = state[0]
    return state


def _vision_span_mask(input_ids: torch.LongTensor, config: CVRRConfig) -> torch.Tensor:
    result = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in (
        config.vision_start_token_id,
        config.vision_end_token_id,
        config.image_token_id,
        config.video_token_id,
    ):
        result |= input_ids.eq(token_id)
    return result


def _select_rows_padded(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather variable-count rows and right-pad them within a batch."""
    counts = mask.sum(dim=1)
    width = int(counts.max())
    output = values.new_zeros(values.shape[0], width, values.shape[-1])
    padding = torch.ones(
        values.shape[0], width, dtype=torch.bool, device=values.device
    )
    for row, count_tensor in enumerate(counts):
        count = int(count_tensor)
        output[row, :count] = values[row, mask[row]]
        padding[row, :count] = False
    return output, padding


def _replace_rows_from_padded(
    full_state: torch.Tensor,
    selection_mask: torch.Tensor,
    padded_rows: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    selected = selection_mask.sum(dim=-1)
    supplied = (~padding_mask).sum(dim=-1)
    if not torch.equal(selected, supplied):
        raise ValueError(
            "selected and supplied row counts differ: "
            f"{selected.tolist()} != {supplied.tolist()}"
        )
    output = full_state.clone()
    for row, count_tensor in enumerate(selected):
        count = int(count_tensor)
        output[row, selection_mask[row]] = padded_rows[row, :count]
    return output


def _answer_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    per_example: bool,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    flat_labels = labels.reshape(-1)
    if not per_example:
        return nn.functional.cross_entropy(
            flat_logits, flat_labels, ignore_index=-100
        )
    token_loss = nn.functional.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=-100,
        reduction="none",
    ).view(labels.shape[0], -1)
    valid = labels.ne(-100).view(labels.shape[0], -1)
    counts = valid.sum(dim=-1)
    nonempty = counts.gt(0)
    if not bool(nonempty.any()):
        return logits.float().sum() * 0.0
    sample_loss = (token_loss * valid).sum(dim=-1) / counts.clamp_min(1)
    return sample_loss[nonempty].mean()


def _build_lora_config(config: CVRRConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        layers_to_transform=[config.recurrent_layer],
        layers_pattern="layers",
        task_type=None,
    )


class CVRRForConditionalGeneration(Qwen2_5_VLPreTrainedModel):
    """CVRR with a strict recurrent visual path and frozen Qwen2.5-VL body."""

    config_class = CVRRConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
    main_input_name = "input_ids"
    accepts_loss_kwargs = False

    def __init__(
        self,
        config: CVRRConfig,
        backbone: Qwen2_5_VLForConditionalGeneration | None = None,
    ) -> None:
        super().__init__(config)
        config.validate_cvrr()
        self.backbone = (
            backbone
            if backbone is not None
            else Qwen2_5_VLForConditionalGeneration._from_config(config)
        )
        self.post_init()

        self.adapter_config = None
        if config.lora_rank > 0:
            from peft import inject_adapter_in_model

            self.adapter_config = _build_lora_config(config)
            inject_adapter_in_model(self.adapter_config, self.backbone)
            self._set_adapter_execution(False)

    @classmethod
    def from_backbone(
        cls,
        pretrained_model_name_or_path: str,
        config: CVRRConfig,
        **kwargs: Any,
    ) -> "CVRRForConditionalGeneration":
        """Construct CVRR from a stock Qwen2.5-VL checkpoint."""
        backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
        config.base_model_name_or_path = str(pretrained_model_name_or_path)
        model = cls(config, backbone=backbone)
        return model.to(backbone.dtype)

    @property
    def vl(self):
        return self.backbone.model

    @property
    def text_model(self):
        return self.backbone.model.language_model

    @property
    def dtype(self) -> torch.dtype:
        return self.backbone.dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_input_embeddings(self):
        return self.vl.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.vl.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.backbone.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.backbone.set_output_embeddings(value)

    def freeze_for_training(self) -> dict[str, int]:
        """Freeze the backbone and enable only recurrent-layer LoRA tensors."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        lora = 0
        for name, parameter in self.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)
                lora += parameter.numel()
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        if trainable != lora:
            raise RuntimeError(
                f"CVRR must train only LoRA tensors: trainable={trainable}, lora={lora}"
            )
        return {"trainable": trainable, "total": total, "lora": lora}

    def _adapter_layers(self) -> list[nn.Module]:
        if self.config.lora_rank == 0:
            return []
        from peft.tuners.tuners_utils import BaseTunerLayer

        return [m for m in self.backbone.modules() if isinstance(m, BaseTunerLayer)]

    def _set_adapter_execution(self, enabled: bool) -> None:
        for module in self._adapter_layers():
            module._disable_adapters = not enabled

    @contextmanager
    def _adapter_execution(self, enabled: bool):
        modules = self._adapter_layers()
        previous = [module.disable_adapters for module in modules]
        for module in modules:
            module._disable_adapters = not enabled
        try:
            yield
        finally:
            for module, was_disabled in zip(modules, previous):
                module._disable_adapters = was_disabled

    def _multimodal_boundary(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        _SplitContext,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return the frozen multimodal boundary scaffold and question rows."""
        embeddings = self.vl.get_input_embeddings()(input_ids)
        image_features = self.vl.get_image_features(pixel_values, image_grid_thw)
        image_features = torch.cat(image_features, dim=0).to(
            embeddings.device, embeddings.dtype
        )
        image_mask, _ = self.vl.get_placeholder_mask(
            input_ids,
            inputs_embeds=embeddings,
            image_features=image_features,
        )
        embeddings = embeddings.masked_scatter(image_mask, image_features)
        position_ids, _ = self.vl.get_rope_index(
            input_ids,
            image_grid_thw,
            None,
            second_per_grid_ts=None,
            attention_mask=attention_mask,
        )
        context = _make_context(
            self.text_model,
            embeddings,
            position_ids,
            attention_mask,
        )
        lower = _run_layers(
            self.text_model,
            context,
            0,
            self.config.ell_star + 1,
        )
        visual_rows = _vision_span_mask(input_ids, self.config)
        text_rows = ~visual_rows
        if attention_mask is not None:
            text_rows &= attention_mask.bool()
            visual_rows &= attention_mask.bool()
        question_state, question_padding = _select_rows_padded(lower, text_rows)
        question_ids, id_padding = _select_rows_padded(
            input_ids.unsqueeze(-1), text_rows
        )
        if not torch.equal(question_padding, id_padding):
            raise RuntimeError("multimodal hidden/id row layouts differ")
        return (
            lower.detach(),
            context,
            text_rows,
            question_state.detach(),
            question_padding,
            question_ids.squeeze(-1),
        )

    def _text_boundary(
        self,
        question_ids: torch.LongTensor,
        question_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, _SplitContext, DynamicCache]:
        batch_size, question_length = question_ids.shape
        embeddings = self.vl.get_input_embeddings()(question_ids)
        if question_attention_mask is None:
            question_attention_mask = torch.ones(
                batch_size,
                question_length,
                dtype=torch.long,
                device=question_ids.device,
            )
        keep = question_attention_mask.bool()
        positions = keep.long().cumsum(dim=-1) - 1
        positions = positions.masked_fill(~keep, 0)
        position_ids = positions.unsqueeze(0).expand(3, -1, -1)
        cache = DynamicCache(config=self.text_model.config)
        context = _make_context(
            self.text_model,
            embeddings,
            position_ids,
            question_attention_mask,
            past_key_values=cache,
        )
        lower = _run_layers(
            self.text_model,
            context,
            0,
            self.config.ell_star + 1,
            use_cache=True,
        )
        context.hidden_states = lower
        return lower, context, cache

    def _select_rope_rows(
        self,
        rope: torch.Tensor,
        text_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rope.ndim == 3:
            return _select_rows_padded(rope, text_rows)
        if rope.ndim != 4 or rope.shape[0] != 3:
            raise ValueError(f"unexpected M-RoPE shape: {tuple(rope.shape)}")
        selected = []
        shared_padding = None
        for axis in range(3):
            rows, padding = _select_rows_padded(rope[axis], text_rows)
            selected.append(rows)
            if shared_padding is None:
                shared_padding = padding
            elif not torch.equal(shared_padding, padding):
                raise RuntimeError("M-RoPE axis layouts differ")
        return torch.stack(selected, dim=0), shared_padding

    def _recurrent_question_transition(
        self,
        full_input: torch.Tensor,
        context: _SplitContext,
        text_rows: torch.Tensor,
        *,
        block_visual_access: bool = False,
        blocked_key_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the shared native layer only at question output rows.

        Keys and values still cover the complete multimodal scaffold, including
        the fixed persistent visual rows.  Decoder-layer attention and MLP are
        pointwise in output rows, so this is algebraically equivalent to a full
        layer call followed by question-row selection while avoiding discarded
        visual-row MLP activations.
        """
        layer = self.text_model.layers[self.config.recurrent_layer]
        attention = layer.self_attn
        if getattr(layer, "attention_type", "full_attention") != "full_attention":
            raise NotImplementedError(
                "CVRR requires full attention at the recurrent layer"
            )

        normalized_full = layer.input_layernorm(full_input)
        normalized_q, q_padding = _select_rows_padded(normalized_full, text_rows)
        residual_q, residual_padding = _select_rows_padded(full_input, text_rows)
        if not torch.equal(q_padding, residual_padding):
            raise RuntimeError("recurrent question row layout changed")

        batch_size, query_length, _ = normalized_q.shape
        full_length = normalized_full.shape[1]
        head_dim = attention.head_dim
        query = attention.q_proj(normalized_q).view(
            batch_size, query_length, -1, head_dim
        ).transpose(1, 2)
        key = attention.k_proj(normalized_full).view(
            batch_size, full_length, -1, head_dim
        ).transpose(1, 2)
        value = attention.v_proj(normalized_full).view(
            batch_size, full_length, -1, head_dim
        ).transpose(1, 2)

        cos_full, sin_full = context.position_embeddings
        cos_q, cos_padding = self._select_rope_rows(cos_full, text_rows)
        sin_q, sin_padding = self._select_rope_rows(sin_full, text_rows)
        if not torch.equal(cos_padding, q_padding) or not torch.equal(
            sin_padding, q_padding
        ):
            raise RuntimeError("recurrent M-RoPE row layout changed")
        section = attention.rope_scaling["mrope_section"]
        query, _ = apply_multimodal_rotary_pos_emb(
            query, query, cos_q, sin_q, section
        )
        _, key = apply_multimodal_rotary_pos_emb(
            key, key, cos_full, sin_full, section
        )

        key_indices = torch.arange(full_length, device=full_input.device)
        query_indices = torch.zeros(
            batch_size,
            query_length,
            dtype=torch.long,
            device=full_input.device,
        )
        for row in range(batch_size):
            physical = text_rows[row].nonzero(as_tuple=False).squeeze(-1)
            query_indices[row, : physical.numel()] = physical
        visible = key_indices.view(1, 1, -1) <= query_indices.unsqueeze(-1)
        if context.attention_mask is not None:
            visible &= context.attention_mask[:, None, :].bool()
        if block_visual_access:
            # Compute-matched no-reread control: all question-query outputs are
            # retained, but keys belonging to persistent visual rows are not
            # visible after native multimodal initialization.
            visible &= text_rows[:, None, :]
        if blocked_key_rows is not None:
            if (
                blocked_key_rows.dtype != torch.bool
                or blocked_key_rows.shape != text_rows.shape
            ):
                raise ValueError(
                    "blocked_key_rows must be a boolean tensor matching text_rows"
                )
            visible &= ~blocked_key_rows[:, None, :]
        visible &= (~q_padding).unsqueeze(-1)
        for row in range(batch_size):
            if bool(q_padding[row].any()):
                visible[row, q_padding[row], 0] = True

        attended, _ = sdpa_attention_forward(
            attention,
            query,
            key,
            value,
            visible.unsqueeze(1),
            dropout=attention.attention_dropout if self.training else 0.0,
            scaling=attention.scaling,
        )
        attended = attention.o_proj(
            attended.reshape(batch_size, query_length, -1).contiguous()
        )
        hidden = residual_q + attended
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return hidden.masked_fill(q_padding.unsqueeze(-1), 0), q_padding

    def _strict_cache_check(self, cache: Cache, expected_length: int) -> None:
        """Assert that no multimodal prefix was written to answer-time KV."""
        for layer_index in range(len(self.text_model.layers)):
            length = int(cache.get_seq_length(layer_index))
            if length != expected_length:
                raise RuntimeError(
                    "strict-path violation: answer cache at layer "
                    f"{layer_index} has length {length}, expected text-only "
                    f"length {expected_length}"
                )

    def prefill(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        question_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        question_attention_mask: torch.Tensor | None = None,
        *,
        return_recurrent_states: bool = False,
    ) -> tuple[torch.Tensor, DynamicCache, tuple[torch.Tensor, ...] | None]:
        """Build the strict text-only/recurrent split prefix cache."""
        if input_ids is None or pixel_values is None or image_grid_thw is None:
            raise ValueError("multimodal input_ids, pixel_values and image_grid_thw are required")
        if question_ids is None:
            raise ValueError("question_ids must be the separately tokenized text-only prompt")

        with torch.no_grad():
            (
                multimodal_boundary,
                multimodal_context,
                text_rows,
                multimodal_question,
                multimodal_padding,
                multimodal_question_ids,
            ) = self._multimodal_boundary(
                input_ids,
                pixel_values,
                image_grid_thw,
                attention_mask,
            )

            text_boundary, text_context, cache = self._text_boundary(
                question_ids,
                question_attention_mask,
            )
            q_padding = (
                torch.zeros_like(question_ids, dtype=torch.bool)
                if question_attention_mask is None
                else question_attention_mask.eq(0)
            )
            if multimodal_question.shape != text_boundary.shape:
                raise ValueError(
                    "multimodal and text-only question states have different shapes: "
                    f"{tuple(multimodal_question.shape)} != {tuple(text_boundary.shape)}"
                )
            if not torch.equal(multimodal_padding, q_padding):
                raise ValueError("multimodal and text-only question padding differ")
            valid = ~q_padding
            if not torch.equal(multimodal_question_ids[valid], question_ids[valid]):
                raise ValueError(
                    "text-only prompt tokens do not match non-visual multimodal tokens"
                )

            with self._adapter_execution(False):
                text_anchor = _run_layers(
                    self.text_model,
                    text_context,
                    self.config.recurrent_layer,
                    self.config.recurrent_layer + 1,
                    hidden_states=text_boundary,
                    use_cache=True,
                )
                first_full = _run_layers(
                    self.text_model,
                    multimodal_context,
                    self.config.recurrent_layer,
                    self.config.recurrent_layer + 1,
                    hidden_states=multimodal_boundary,
                    use_cache=False,
                )
            first_state, first_padding = _select_rows_padded(first_full, text_rows)
            if not torch.equal(first_padding, q_padding):
                raise RuntimeError("initial recurrent state layout differs from text prompt")

        state = first_state
        states = [state]
        beta = self.config.beta
        for _ in range(1, self.config.num_recurrent_steps):
            recurrent_input = _replace_rows_from_padded(
                first_full,
                text_rows,
                state,
                q_padding,
            )
            with self._adapter_execution(True):
                proposal, proposal_padding = self._recurrent_question_transition(
                    recurrent_input,
                    multimodal_context,
                    text_rows,
                )
            if not torch.equal(proposal_padding, q_padding):
                raise RuntimeError("recurrent transition changed question layout")
            state = torch.lerp(state, proposal, beta)
            state = state.masked_fill(q_padding.unsqueeze(-1), 0)
            states.append(state)

        upper = _run_layers(
            self.text_model,
            text_context,
            self.config.upper_decoder_start,
            None,
            hidden_states=state.to(text_anchor.dtype),
            use_cache=True,
        )
        self._strict_cache_check(cache, question_ids.shape[1])
        return upper, cache, tuple(states) if return_recurrent_states else None

    def _decode_tokens(
        self,
        embeddings: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        cache: Cache,
        cache_position: torch.LongTensor,
    ) -> torch.Tensor:
        lower_context = _make_context(
            self.text_model,
            embeddings,
            position_ids,
            attention_mask,
            past_key_values=cache,
            cache_position=cache_position,
        )
        lower = _run_layers(
            self.text_model,
            lower_context,
            0,
            self.config.upper_decoder_start,
            use_cache=True,
        )
        upper_context = _make_context(
            self.text_model,
            embeddings,
            position_ids,
            attention_mask,
            past_key_values=cache,
            cache_position=cache_position,
        )
        return _run_layers(
            self.text_model,
            upper_context,
            self.config.upper_decoder_start,
            None,
            hidden_states=lower,
            use_cache=True,
        )

    def _prefix_seed(
        self,
        upper: torch.Tensor,
        question_ids: torch.LongTensor,
        question_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = (
            torch.ones_like(question_ids, dtype=torch.long)
            if question_attention_mask is None
            else question_attention_mask
        )
        lengths = mask.bool().sum(dim=-1).long()
        last = (lengths - 1).clamp_min(0)
        seed = upper[
            torch.arange(upper.shape[0], device=upper.device), last
        ].unsqueeze(1)
        return seed, mask, lengths

    def next_token_logits(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        question_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        question_attention_mask: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        """Return logits at the first answer position."""
        upper, _, _ = self.prefill(
            input_ids,
            pixel_values,
            image_grid_thw,
            question_ids,
            attention_mask,
            question_attention_mask,
        )
        seed, _, _ = self._prefix_seed(
            upper, question_ids, question_attention_mask
        )
        return self.backbone.lm_head(self.text_model.norm(seed))

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        question_ids: torch.LongTensor | None = None,
        question_attention_mask: torch.Tensor | None = None,
        answer_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        output_recurrent_states: bool = False,
        **kwargs: Any,
    ) -> CVRROutput:
        if kwargs:
            raise TypeError(f"unexpected forward arguments: {sorted(kwargs)}")
        if past_key_values is not None:
            raise ValueError("CVRR forward always constructs a fresh strict prefix cache")
        if return_dict is False:
            raise ValueError("CVRR supports return_dict=True only")
        if any(x is None for x in (input_ids, pixel_values, image_grid_thw, question_ids)):
            raise ValueError(
                "input_ids, pixel_values, image_grid_thw, and question_ids are required"
            )

        upper, cache, recurrent_states = self.prefill(
            input_ids,
            pixel_values,
            image_grid_thw,
            question_ids,
            attention_mask,
            question_attention_mask,
            return_recurrent_states=output_recurrent_states,
        )
        seed, q_mask, q_lengths = self._prefix_seed(
            upper, question_ids, question_attention_mask
        )
        hidden_for_logits = seed

        if answer_ids is not None and answer_ids.shape[1] > 1:
            answer_embeddings = self.vl.get_input_embeddings()(answer_ids)
            answer_length = answer_ids.shape[1]
            cache_position = torch.arange(
                question_ids.shape[1],
                question_ids.shape[1] + answer_length,
                device=answer_ids.device,
            )
            logical_positions = q_lengths[:, None] + torch.arange(
                answer_length, device=answer_ids.device
            )[None, :]
            position_ids = logical_positions.unsqueeze(0).expand(3, -1, -1)
            answer_mask = (
                labels.ne(-100).to(q_mask.dtype)
                if labels is not None and labels.shape == answer_ids.shape
                else torch.ones_like(answer_ids, dtype=q_mask.dtype)
            )
            combined_mask = torch.cat([q_mask, answer_mask], dim=-1)
            answer_hidden = self._decode_tokens(
                answer_embeddings,
                position_ids,
                combined_mask,
                cache,
                cache_position,
            )
            hidden_for_logits = torch.cat(
                [hidden_for_logits, answer_hidden[:, :-1]], dim=1
            )

        logits = self.backbone.lm_head(self.text_model.norm(hidden_for_logits))
        loss = None
        if labels is not None:
            if answer_ids is None or labels.shape != answer_ids.shape:
                raise ValueError("labels and answer_ids must have the same shape")
            loss = _answer_cross_entropy(
                logits,
                labels,
                per_example=self.config.per_example_answer_loss,
            )
        return CVRROutput(
            loss=loss,
            logits=logits,
            past_key_values=cache if use_cache else None,
            recurrent_states=recurrent_states,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        question_ids: torch.LongTensor | None = None,
        question_attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        eos_token_id: int | None = None,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> torch.LongTensor:
        """Greedy generation from the strict split prefix.

        The returned tensor contains generated answer tokens only.  A CVRR
        prompt has separate multimodal and text-only representations, so there
        is no single token prefix to concatenate to the output.
        """
        if do_sample:
            raise ValueError("the released evaluation protocol uses greedy decoding")
        if kwargs:
            raise TypeError(f"unsupported generation arguments: {sorted(kwargs)}")
        if any(x is None for x in (input_ids, pixel_values, image_grid_thw, question_ids)):
            raise ValueError(
                "input_ids, pixel_values, image_grid_thw, and question_ids are required"
            )
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        upper, cache, _ = self.prefill(
            input_ids,
            pixel_values,
            image_grid_thw,
            question_ids,
            attention_mask,
            question_attention_mask,
        )
        seed, q_mask, q_lengths = self._prefix_seed(
            upper, question_ids, question_attention_mask
        )
        next_token = self.backbone.lm_head(self.text_model.norm(seed)).argmax(-1)
        generated = [next_token]
        done = torch.zeros(
            next_token.shape[0], dtype=torch.bool, device=next_token.device
        )
        if eos_token_id is not None:
            done |= next_token.squeeze(1).eq(eos_token_id)

        for step in range(1, max_new_tokens):
            if bool(done.all()):
                break
            embeddings = self.vl.get_input_embeddings()(next_token)
            cache_position = torch.tensor(
                [question_ids.shape[1] + step - 1], device=question_ids.device
            )
            logical_position = q_lengths + step - 1
            position_ids = logical_position.view(1, -1, 1).expand(3, -1, -1)
            generated_mask = torch.ones(
                question_ids.shape[0],
                step,
                dtype=q_mask.dtype,
                device=q_mask.device,
            )
            decode_mask = torch.cat([q_mask, generated_mask], dim=-1)
            hidden = self._decode_tokens(
                embeddings,
                position_ids,
                decode_mask,
                cache,
                cache_position,
            )
            next_token = self.backbone.lm_head(
                self.text_model.norm(hidden)
            ).argmax(-1)
            if eos_token_id is not None:
                next_token = torch.where(
                    done.unsqueeze(1),
                    next_token.new_full(next_token.shape, eos_token_id),
                    next_token,
                )
                done |= next_token.squeeze(1).eq(eos_token_id)
            generated.append(next_token)
        return torch.cat(generated, dim=1)


CVRRForConditionalGeneration.register_for_auto_class(
    "AutoModelForImageTextToText"
)


__all__ = ["CVRRForConditionalGeneration", "CVRROutput"]
