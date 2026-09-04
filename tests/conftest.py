from __future__ import annotations

import pytest
import torch

from cvrr import CVRRConfig, CVRRForConditionalGeneration


@pytest.fixture(scope="session")
def tiny_config():
    hidden = 64
    heads = 4
    head_dim = hidden // heads
    return CVRRConfig(
        text_config={
            "hidden_size": hidden,
            "intermediate_size": 2 * hidden,
            "num_hidden_layers": 5,
            "num_attention_heads": heads,
            "num_key_value_heads": 2,
            "vocab_size": 1000,
            "max_position_embeddings": 2048,
            "use_sliding_window": False,
            "rope_scaling": {
                "type": "mrope",
                "mrope_section": [
                    head_dim // 8,
                    3 * head_dim // 16,
                    3 * head_dim // 16,
                ],
            },
        },
        vision_config={
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": hidden,
            "in_chans": 3,
            "patch_size": 14,
            "spatial_patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "window_size": 112,
            "fullatt_block_indexes": [1],
            "tokens_per_second": 2,
        },
        image_token_id=900,
        vision_start_token_id=901,
        vision_end_token_id=902,
        video_token_id=903,
        pad_token_id=0,
        eos_token_id=2,
        ell_star=2,
        num_recurrent_steps=3,
        beta=0.5,
        lora_rank=4,
        lora_alpha=2,
        lora_dropout=0.0,
    )


@pytest.fixture
def tiny_model(tiny_config):
    torch.manual_seed(0)
    model = CVRRForConditionalGeneration(tiny_config).float()
    model.freeze_for_training()
    return model


@pytest.fixture
def tiny_batch():
    torch.manual_seed(1)
    temporal, height, width = 1, 4, 4
    image_tokens = temporal * height * width // 4
    input_ids = torch.tensor(
        [[10, 11, 12, 901, *([900] * image_tokens), 902, 20, 21, 22, 23]],
        dtype=torch.long,
    )
    question_ids = torch.tensor([[10, 11, 12, 20, 21, 22, 23]])
    patch_dim = 3 * 2 * 14 * 14
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": torch.randn(
            temporal * height * width, patch_dim, dtype=torch.float32
        ),
        "image_grid_thw": torch.tensor(
            [[temporal, height, width]], dtype=torch.long
        ),
        "question_ids": question_ids,
        "question_attention_mask": torch.ones_like(question_ids),
        "answer_ids": torch.tensor([[30, 2]], dtype=torch.long),
        "labels": torch.tensor([[30, 2]], dtype=torch.long),
    }
