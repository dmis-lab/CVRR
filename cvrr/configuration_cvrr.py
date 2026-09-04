"""Configuration for CVRR on Qwen2.5-VL."""

from __future__ import annotations

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig


class CVRRConfig(Qwen2_5_VLConfig):
    """Qwen2.5-VL configuration with the small set of CVRR hyperparameters.

    ``ell_star`` is the zero-indexed, inclusive end of the frozen lower
    branch.  Layer ``ell_star + 1`` is reused as the shared recurrent
    transition and the answer decoder begins at ``ell_star + 2``.
    """

    model_type = "cvrr_qwen2_5_vl"

    def __init__(
        self,
        ell_star: int = 20,
        num_recurrent_steps: int = 4,
        beta: float = 0.5,
        lora_rank: int = 32,
        lora_alpha: int = 12,
        lora_dropout: float = 0.01,
        base_model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        per_example_answer_loss: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.ell_star = int(ell_star)
        self.num_recurrent_steps = int(num_recurrent_steps)
        self.beta = float(beta)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.base_model_name_or_path = str(base_model_name_or_path)
        self.per_example_answer_loss = bool(per_example_answer_loss)
        self.auto_map = {
            "AutoConfig": "configuration_cvrr.CVRRConfig",
            "AutoModelForImageTextToText": (
                "modeling_cvrr.CVRRForConditionalGeneration"
            ),
        }
        self.validate_cvrr()

    @property
    def recurrent_layer(self) -> int:
        return self.ell_star + 1

    @property
    def upper_decoder_start(self) -> int:
        return self.ell_star + 2

    def validate_cvrr(self) -> None:
        n_layers = int(self.text_config.num_hidden_layers)
        if not 0 <= self.ell_star < n_layers - 1:
            raise ValueError(
                f"ell_star must be in [0, {n_layers - 2}], got {self.ell_star}"
            )
        if self.num_recurrent_steps < 1:
            raise ValueError("num_recurrent_steps must be at least 1")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must lie in [0, 1]")
        if self.lora_rank < 0:
            raise ValueError("lora_rank must be non-negative")
        if self.lora_rank and self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive when LoRA is enabled")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must lie in [0, 1)")


CVRRConfig.register_for_auto_class()


__all__ = ["CVRRConfig"]
