"""Public CVRR package."""

from .configuration_cvrr import CVRRConfig
from .modeling_cvrr import CVRRForConditionalGeneration, CVRROutput

from transformers import AutoConfig, AutoModelForImageTextToText

AutoConfig.register(CVRRConfig.model_type, CVRRConfig, exist_ok=True)
AutoModelForImageTextToText.register(
    CVRRConfig, CVRRForConditionalGeneration, exist_ok=True
)

__all__ = ["CVRRConfig", "CVRRForConditionalGeneration", "CVRROutput"]
