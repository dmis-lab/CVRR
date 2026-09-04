"""Exact split-forward runtime shared by the released CVRR analyses.

The default ``strict`` readout sends only a question-shaped recurrent state to
the upper answer decoder.  The two non-strict paths are evaluation-only causal
controls: ``visual_rows_restored`` reinserts post-cell visual rows, while
``dual_path_bypass`` exposes an untouched native multimodal prefix in addition
to the intervened recurrent stream.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.analysis_cvrr.core.common import configure_huggingface
from scripts.analysis_cvrr.core.data import PreparedExample, configure_visual_cap


@dataclass
class NativeTrace:
    """Native boundary extraction for one prepared example."""

    example: PreparedExample
    mm_context: Any
    question_context: Any
    boundary_full: Any
    first_full: Any
    text_rows: Any
    visual_rows: Any
    question_valid: Any
    base_anchor: Any
    r1: Any
    evidence: Any
    cell_index: int
    upper_start: int
    mm_last: int
    question_last: int

    @property
    def visual_tokens(self) -> int:
        return int(self.visual_rows.sum().item())

    @property
    def question_tokens(self) -> int:
        return int(self.question_valid.sum().item())


class CVRRRuntime:
    """Evaluation-only access to a released Qwen2.5-VL CVRR checkpoint."""

    def __init__(
        self,
        *,
        checkpoint: str,
        processor_name: str | None = None,
        device: str = "cuda:0",
        max_visual_tokens: int = 8192,
        beta: float = 0.5,
        adapter_scale: float = 1.0,
        offline: bool = True,
    ) -> None:
        configure_huggingface(offline=offline)

        import torch
        from transformers import AutoProcessor

        from cvrr import CVRRConfig, CVRRForConditionalGeneration

        self.torch = torch
        candidate = Path(checkpoint).expanduser()
        self.checkpoint = str(candidate.resolve()) if candidate.exists() else checkpoint
        config = CVRRConfig.from_pretrained(
            self.checkpoint,
            local_files_only=offline,
        )
        if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be finite and lie in [0,1]")
        config.beta = float(beta)
        config.validate_cvrr()

        model, loading = CVRRForConditionalGeneration.from_pretrained(
            self.checkpoint,
            config=config,
            dtype=torch.bfloat16,
            output_loading_info=True,
            local_files_only=offline,
        )
        missing = list(loading.get("missing_keys", []))
        unexpected = list(loading.get("unexpected_keys", []))
        if missing or unexpected:
            raise RuntimeError(
                "checkpoint did not load exactly: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )

        self.parameter_counts = model.freeze_for_training()
        self.adapter_scale_info = self._set_adapter_scale(
            model, float(adapter_scale)
        )
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.config = self.model.config
        self.model_type = str(self.config.model_type)
        self.cell_index = int(self.config.recurrent_layer)
        self.upper_start = int(self.config.upper_decoder_start)
        self.beta = float(self.config.beta)

        processor_source = (
            processor_name
            or getattr(self.config, "base_model_name_or_path", "")
            or self.checkpoint
        )
        self.processor_name = str(processor_source)
        self.processor = AutoProcessor.from_pretrained(
            processor_source,
            use_fast=True,
            local_files_only=offline,
        )
        configure_visual_cap(
            self.processor,
            self.config,
            int(max_visual_tokens),
        )
        self.max_visual_tokens = int(max_visual_tokens)

    @staticmethod
    def _set_adapter_scale(model, scale: float) -> dict[str, Any]:
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("adapter_scale must be finite and non-negative")
        changed = 0
        adapters: set[str] = set()
        for module in model._adapter_layers():
            scaling = getattr(module, "scaling", None)
            if not isinstance(scaling, dict):
                continue
            for name, value in list(scaling.items()):
                scaling[name] = float(value) * scale
                adapters.add(str(name))
                changed += 1
        return {
            "global_scale": scale,
            "scaled_modules": changed,
            "adapters": sorted(adapters),
        }

    @contextlib.contextmanager
    def adapters(self, enabled: bool):
        with self.model._adapter_execution(bool(enabled)):
            yield

    def _cell(self, hidden, context, *, adapters: bool):
        from cvrr.modeling_cvrr import _run_layers

        with self.adapters(adapters):
            return _run_layers(
                self.model.text_model,
                context,
                self.cell_index,
                self.cell_index + 1,
                hidden_states=hidden,
                use_cache=False,
            )

    @staticmethod
    def _selected_rows(hidden, mask):
        if hidden.shape[0] != 1 or mask.shape[0] != 1:
            raise ValueError("analysis runtime currently requires batch size one")
        count = int(mask.sum().item())
        return hidden[mask].reshape(1, count, hidden.shape[-1])

    @staticmethod
    def _replace_rows(full, mask, rows):
        count = int(mask.sum().item())
        if rows.shape[:2] != (1, count):
            raise ValueError(
                f"replacement shape {tuple(rows.shape)} does not match {count} rows"
            )
        result = full.clone()
        result[mask] = rows.reshape(count, rows.shape[-1])
        return result

    def extract(self, example: PreparedExample) -> NativeTrace:
        """Run through the first native multimodal read without recurrence."""

        torch = self.torch
        from cvrr.modeling_cvrr import _make_context, _run_layers

        mm = example.multimodal
        pixel_values = mm["pixel_values"].to(dtype=self.model.dtype)
        (
            boundary_full,
            mm_context,
            text_rows,
            _multimodal_question,
            _multimodal_padding,
            _multimodal_ids,
        ) = self.model._multimodal_boundary(
            mm["input_ids"],
            pixel_values,
            mm["image_grid_thw"],
            mm.get("attention_mask"),
        )
        first_full = self._cell(boundary_full, mm_context, adapters=False)

        attention = mm.get("attention_mask")
        if attention is None:
            attention = torch.ones_like(mm["input_ids"])
        valid_full = attention.gt(0)
        visual_rows = (~text_rows) & valid_full

        question_ids = example.question_ids
        question_mask = example.question_attention_mask
        question_valid = question_mask.gt(0)
        q_embeds = self.model.get_input_embeddings()(question_ids)
        q_positions_1d = question_valid.long().cumsum(dim=-1) - 1
        q_positions_1d = q_positions_1d.masked_fill(~question_valid, 0)
        q_positions = q_positions_1d.unsqueeze(0).expand(3, -1, -1)
        question_context = _make_context(
            self.model.text_model,
            q_embeds,
            q_positions,
            question_mask,
        )
        q_star = _run_layers(
            self.model.text_model,
            question_context,
            0,
            self.cell_index,
            hidden_states=q_embeds,
            use_cache=False,
        )
        base_anchor = self._cell(q_star, question_context, adapters=False)
        r1 = self._selected_rows(first_full, text_rows)
        evidence = self._selected_rows(first_full, visual_rows)

        selected_ids = mm["input_ids"][text_rows]
        expected_ids = question_ids[question_valid]
        if selected_ids.shape != expected_ids.shape or not torch.equal(
            selected_ids, expected_ids
        ):
            raise RuntimeError(
                f"multimodal/text-only token mismatch for {example.item_id}"
            )
        if r1.shape != base_anchor.shape:
            raise RuntimeError(
                f"question state mismatch: R1={tuple(r1.shape)} "
                f"B={tuple(base_anchor.shape)}"
            )

        return NativeTrace(
            example=example,
            mm_context=mm_context,
            question_context=question_context,
            boundary_full=boundary_full,
            first_full=first_full,
            text_rows=text_rows,
            visual_rows=visual_rows,
            question_valid=question_valid,
            base_anchor=base_anchor,
            r1=r1,
            evidence=evidence,
            cell_index=self.cell_index,
            upper_start=self.upper_start,
            mm_last=int(attention.sum().item()) - 1,
            question_last=int(question_mask.sum().item()) - 1,
        )

    def rollout(
        self,
        trace: NativeTrace,
        *,
        steps: int,
        beta: float | None = None,
        initial_state=None,
        evidence=None,
        visual_access: str = "every_step",
        recurrent_adapters: bool = True,
    ) -> list[Any]:
        """Roll out ``R1..RT`` with optional crossed state/evidence controls."""

        if steps < 1:
            raise ValueError("steps must be >= 1")
        if visual_access not in {"every_step", "first_only", "no_visual"}:
            raise ValueError(f"unknown visual_access={visual_access!r}")
        coefficient = self.beta if beta is None else float(beta)
        if not 0.0 <= coefficient <= 1.0:
            raise ValueError("beta must lie in [0,1]")

        if initial_state is None:
            initial_state = (
                trace.base_anchor if visual_access == "no_visual" else trace.r1
            )
        if initial_state.shape != trace.r1.shape:
            raise ValueError("initial state does not match target question rows")

        if visual_access == "no_visual":
            state = initial_state
            states = [state]
            for _ in range(1, steps):
                proposal = self._cell(
                    state,
                    trace.question_context,
                    adapters=recurrent_adapters,
                )
                state = torch_lerp(state, proposal, coefficient)
                states.append(state)
            return states

        if evidence is None:
            evidence = trace.evidence
        if evidence.shape != trace.evidence.shape:
            raise ValueError(
                "crossed evidence must have the same visual-token shape; "
                f"got {tuple(evidence.shape)} vs {tuple(trace.evidence.shape)}"
            )

        scaffold = trace.first_full.clone()
        scaffold[trace.visual_rows] = evidence.reshape(
            int(trace.visual_rows.sum().item()), evidence.shape[-1]
        )
        state = initial_state
        states = [state]
        for _ in range(1, steps):
            recurrent_input = self._replace_rows(scaffold, trace.text_rows, state)
            with self.adapters(recurrent_adapters):
                proposal, proposal_padding = (
                    self.model._recurrent_question_transition(
                        recurrent_input,
                        trace.mm_context,
                        trace.text_rows,
                        block_visual_access=visual_access == "first_only",
                    )
                )
            if bool(proposal_padding.any()):
                raise RuntimeError("batch-one analysis unexpectedly produced padding")
            state = torch_lerp(state, proposal, coefficient)
            states.append(state)
        return states

    def _lm_head(self, hidden):
        normalized = self.model.text_model.norm(hidden)
        return self.model.backbone.lm_head(normalized)

    def strict_logits(self, trace: NativeTrace, candidate):
        """First-answer-token logits with no image-conditioned upper prefix."""

        from cvrr.modeling_cvrr import _run_layers

        if candidate.shape != trace.base_anchor.shape:
            raise ValueError("strict candidate must be question-shaped")
        with self.adapters(False):
            upper = _run_layers(
                self.model.text_model,
                trace.question_context,
                trace.upper_start,
                None,
                hidden_states=candidate.to(trace.base_anchor.dtype),
                use_cache=False,
            )
        return self._lm_head(
            upper[:, trace.question_last : trace.question_last + 1]
        )[0, 0]

    def visual_rows_restored_logits(self, trace: NativeTrace, candidate):
        """Restore post-cell visual rows, but not a clean multimodal Q stream."""

        from cvrr.modeling_cvrr import _run_layers

        restored = self._replace_rows(
            trace.first_full,
            trace.text_rows,
            candidate.to(trace.first_full.dtype),
        )
        with self.adapters(False):
            upper = _run_layers(
                self.model.text_model,
                trace.mm_context,
                trace.upper_start,
                None,
                hidden_states=restored,
                use_cache=False,
            )
        return self._lm_head(upper[:, trace.mm_last : trace.mm_last + 1])[0, 0]

    def full_mm_oracle_logits(self, trace: NativeTrace):
        """Decode the untouched native multimodal continuation."""

        from cvrr.modeling_cvrr import _run_layers

        with self.adapters(False):
            upper = _run_layers(
                self.model.text_model,
                trace.mm_context,
                trace.upper_start,
                None,
                hidden_states=trace.first_full,
                use_cache=False,
            )
        return self._lm_head(upper[:, trace.mm_last : trace.mm_last + 1])[0, 0]

    def dual_path_bypass_logits(self, trace: NativeTrace, candidate):
        """Expose an untouched multimodal prefix plus recurrent query stream."""

        import torch

        from cvrr.modeling_cvrr import _make_context, _run_layers

        candidate = candidate.to(trace.first_full.dtype)
        if candidate.shape != trace.r1.shape:
            raise ValueError("dual-path candidate does not match question state")
        mm_attention = trace.mm_context.attention_mask
        if mm_attention is None:
            mm_attention = torch.ones(
                trace.first_full.shape[:2],
                dtype=torch.long,
                device=trace.first_full.device,
            )
        candidate_attention = torch.ones(
            candidate.shape[:2],
            dtype=mm_attention.dtype,
            device=mm_attention.device,
        )
        attention = torch.cat((mm_attention, candidate_attention), dim=1)

        mm_positions = trace.mm_context.position_ids
        last_position = mm_positions[:, :, trace.mm_last : trace.mm_last + 1]
        offsets = torch.arange(
            1,
            candidate.shape[1] + 1,
            dtype=last_position.dtype,
            device=last_position.device,
        ).view(1, 1, -1)
        positions = torch.cat((mm_positions, last_position + offsets), dim=-1)
        combined = torch.cat((trace.first_full, candidate), dim=1)
        context = _make_context(
            self.model.text_model,
            combined,
            positions,
            attention,
        )
        with self.adapters(False):
            upper = _run_layers(
                self.model.text_model,
                context,
                trace.upper_start,
                None,
                hidden_states=combined,
                use_cache=False,
            )
        return self._lm_head(upper[:, -1:])[0, 0]

    def token_scores(
        self,
        trace: NativeTrace,
        candidate,
        token_ids: list[int] | tuple[int, ...],
        *,
        path: str,
        representation: str = "recurrent",
    ) -> list[float]:
        """Score a recurrent state or operational residual at one token step."""

        if representation == "recurrent":
            candidate = trace.base_anchor + (candidate - trace.base_anchor)
        elif representation == "residual":
            candidate = trace.base_anchor + candidate
        elif representation != "decoder":
            raise ValueError(f"unknown candidate representation {representation!r}")
        if candidate.shape != trace.base_anchor.shape:
            raise ValueError(
                "decoder candidate shape mismatch: "
                f"{tuple(candidate.shape)} vs {tuple(trace.base_anchor.shape)}"
            )
        if path == "strict":
            logits = self.strict_logits(trace, candidate)
        elif path in {"bypass_restored", "visual_rows_restored"}:
            logits = self.visual_rows_restored_logits(trace, candidate)
        elif path == "dual_path_bypass":
            logits = self.dual_path_bypass_logits(trace, candidate)
        elif path == "full_mm_oracle":
            logits = self.full_mm_oracle_logits(trace)
        else:
            raise ValueError(f"unknown decoder path {path!r}")
        return [float(logits[token_id].float().item()) for token_id in token_ids]

    def canonical_state(self, trace: NativeTrace, *, steps: int | None = None):
        horizon = (
            int(self.config.num_recurrent_steps) if steps is None else int(steps)
        )
        return self.rollout(trace, steps=horizon)[-1]

    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "model_type": self.model_type,
            "processor": self.processor_name,
            "ell_star": int(self.config.ell_star),
            "cell_layer": self.cell_index,
            "trained_steps": int(self.config.num_recurrent_steps),
            "beta": self.beta,
            "max_visual_tokens": self.max_visual_tokens,
            "adapter_scale": self.adapter_scale_info,
            "parameter_counts": self.parameter_counts,
            "strict_answer_path": "question_state_only_above_recurrent_cell",
            "bypass_controls": {
                "visual_rows_restored": "restore_post_cell_visual_rows_only",
                "dual_path_bypass": (
                    "untouched_multimodal_prefix_plus_recurrent_query_stream"
                ),
                "full_mm_oracle": "untouched_multimodal_continuation",
            },
        }


def torch_lerp(previous, proposal, beta: float):
    return previous + float(beta) * (proposal - previous)


def norm_matched_noise(reference, *, seed: int):
    """Independent Gaussian direction with every token-row norm preserved."""

    import torch

    generator = torch.Generator(device=reference.device)
    generator.manual_seed(int(seed))
    noise = torch.randn(
        reference.shape,
        dtype=torch.float32,
        device=reference.device,
        generator=generator,
    )
    target_norm = reference.float().norm(dim=-1, keepdim=True)
    noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (noise * target_norm).to(reference.dtype)


def state_geometry(previous, current) -> dict[str, float]:
    import torch

    before = previous.float().reshape(-1)
    after = current.float().reshape(-1)
    delta = after - before
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(before, after, dim=0)),
        "relative_update": float(delta.norm() / before.norm().clamp_min(1e-8)),
        "update_rms": float(delta.square().mean().sqrt()),
        "state_rms": float(before.square().mean().sqrt()),
    }


def release_trace(trace: NativeTrace) -> None:
    """Drop large tensor references before processing another image."""

    for field in (
        "boundary_full",
        "first_full",
        "base_anchor",
        "r1",
        "evidence",
        "mm_context",
        "question_context",
    ):
        setattr(trace, field, None)


__all__ = [
    "CVRRRuntime",
    "NativeTrace",
    "norm_matched_noise",
    "release_trace",
    "state_geometry",
]
