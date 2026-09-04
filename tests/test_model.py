import json

import torch

from cvrr import CVRRForConditionalGeneration
from cvrr.modeling_cvrr import _replace_rows_from_padded, _run_layers


def test_only_recurrent_layer_lora_is_trainable(tiny_model):
    assert tiny_model.accepts_loss_kwargs is False
    trainable = [
        name for name, parameter in tiny_model.named_parameters() if parameter.requires_grad
    ]
    assert trainable
    assert all("lora_" in name for name in trainable)
    assert all(".layers.3." in name for name in trainable)


def test_strict_forward_has_finite_loss_and_gradients(tiny_model, tiny_batch):
    tiny_model.train()
    output = tiny_model(**tiny_batch, output_recurrent_states=True)
    assert output.logits.shape == (1, 2, 1000)
    assert torch.isfinite(output.loss)
    assert len(output.recurrent_states) == 3
    output.loss.backward()
    gradients = [
        parameter.grad
        for parameter in tiny_model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_compute_matched_visual_access_control(tiny_model, tiny_batch):
    tiny_model.eval()
    with torch.inference_mode():
        (
            boundary,
            context,
            text_rows,
            _question,
            padding,
            _question_ids,
        ) = tiny_model._multimodal_boundary(
            tiny_batch["input_ids"],
            tiny_batch["pixel_values"],
            tiny_batch["image_grid_thw"],
            tiny_batch["attention_mask"],
        )
        with tiny_model._adapter_execution(False):
            first_full = _run_layers(
                tiny_model.text_model,
                context,
                tiny_model.config.recurrent_layer,
                tiny_model.config.recurrent_layer + 1,
                hidden_states=boundary,
            )
        first_state = first_full[text_rows].reshape(
            1, int(text_rows.sum()), first_full.shape[-1]
        )
        recurrent_input = _replace_rows_from_padded(
            first_full,
            text_rows,
            first_state,
            padding,
        )
        enabled, enabled_padding = tiny_model._recurrent_question_transition(
            recurrent_input,
            context,
            text_rows,
        )
        blocked, blocked_padding = tiny_model._recurrent_question_transition(
            recurrent_input,
            context,
            text_rows,
            block_visual_access=True,
        )
    assert enabled.shape == blocked.shape == first_state.shape
    assert torch.equal(enabled_padding, blocked_padding)
    assert torch.isfinite(enabled).all() and torch.isfinite(blocked).all()
    assert not torch.equal(enabled, blocked)


def test_greedy_generation(tiny_model, tiny_batch):
    tiny_model.eval()
    inputs = {
        key: tiny_batch[key]
        for key in (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "question_ids",
            "question_attention_mask",
        )
    }
    generated = tiny_model.generate(**inputs, max_new_tokens=3)
    assert generated.shape[0] == 1
    assert 1 <= generated.shape[1] <= 3


def test_save_pretrained_round_trip(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path)
    serialized = json.loads((tmp_path / "config.json").read_text())
    assert serialized["model_type"] == "cvrr_qwen2_5_vl"
    assert set(serialized["auto_map"]) == {
        "AutoConfig",
        "AutoModelForImageTextToText",
    }
    assert serialized["auto_map"]["AutoModelForImageTextToText"].endswith(
        ".CVRRForConditionalGeneration"
    )
    loaded, info = CVRRForConditionalGeneration.from_pretrained(
        tmp_path, output_loading_info=True
    )
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    expected = tiny_model.state_dict()
    actual = loaded.state_dict()
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
