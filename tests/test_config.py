from cvrr import CVRRConfig


def test_config_round_trip(tiny_config, tmp_path):
    tiny_config.save_pretrained(tmp_path)
    loaded = CVRRConfig.from_pretrained(tmp_path)
    assert loaded.ell_star == 2
    assert loaded.recurrent_layer == 3
    assert loaded.upper_decoder_start == 4
    assert loaded.num_recurrent_steps == 3
    assert loaded.beta == 0.5
    assert loaded.lora_rank == 4


def test_invalid_boundary_is_rejected():
    try:
        CVRRConfig(
            text_config={"num_hidden_layers": 2},
            ell_star=1,
        )
    except ValueError as error:
        assert "ell_star" in str(error)
    else:
        raise AssertionError("invalid boundary was accepted")
