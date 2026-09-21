import pytest

from cvrr.benchmarks import summarize


def mmvp_row(qid, correct):
    return {"benchmark": "mmvp", "task": "MMVP", "qid": str(qid), "correct": correct}


@pytest.mark.parametrize("num_shards", [2, 3, 4])
def test_mmvp_shards_require_original_pairs_and_merge_independent_of_order(num_shards):
    rows = [
        mmvp_row(qid, correct)
        for qid, correct in enumerate(
            [True, False, True, True, False, True, True, True], start=1
        )
    ]
    shards = [rows[shard::num_shards] for shard in range(num_shards)]
    for shard in shards:
        summary = summarize(shard)
        assert "pair" not in summary
        assert summary["incomplete_pairs"] == len(shard)

    merged = [row for shard in reversed(shards) for row in reversed(shard)]
    summary = summarize(merged)
    assert summary == summarize(rows)
    assert summary["pair"] == {"accuracy": 50.0, "correct": 2, "count": 4}
    assert summary["overall"] == {"accuracy": 75.0, "correct": 6, "count": 8}
    assert "incomplete_pairs" not in summary


@pytest.mark.parametrize("qids,incomplete", [([1], 1), ([2, 3], 2), ([1, 2, 3], 1)])
def test_mmvp_incomplete_pairs_do_not_produce_a_partial_pair_score(qids, incomplete):
    summary = summarize(mmvp_row(qid, True) for qid in qids)
    assert "pair" not in summary
    assert summary["incomplete_pairs"] == incomplete
    assert summary["overall"]["count"] == len(qids)


def test_mmvp_duplicate_question_is_rejected():
    with pytest.raises(ValueError, match="duplicate MMVP question ID"):
        summarize([mmvp_row(1, True), mmvp_row(2, True), mmvp_row(1, True)])


@pytest.mark.parametrize("qid", [0, -1])
def test_mmvp_nonpositive_question_id_is_rejected(qid):
    with pytest.raises(ValueError, match="must be positive"):
        summarize([mmvp_row(qid, True)])


def test_other_benchmarks_do_not_enter_mmvp_pairs():
    rows = [mmvp_row(2, True), mmvp_row(1, False)]
    rows.append({"benchmark": "vstar", "task": "direct_attributes", "correct": True})
    summary = summarize(rows)
    assert summary["pair"] == {"accuracy": 0.0, "correct": 0, "count": 1}
    assert summary["tasks"]["direct_attributes"]["accuracy"] == 100.0
    assert "pair" not in summarize(rows[-1:])
    assert "incomplete_pairs" not in summarize(rows[-1:])
