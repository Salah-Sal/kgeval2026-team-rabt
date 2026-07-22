import pytest

from kgeval.chunking import chunk_spans


def test_empty():
    assert chunk_spans([], 10) == []


def test_single_chunk_exact_fit():
    assert chunk_spans([3, 4, 3], 10) == [(0, 3)]


def test_splits_at_word_boundary():
    # 4+4 fits, adding the next 4 would exceed 10
    assert chunk_spans([4, 4, 4, 4, 4], 10) == [(0, 2), (2, 4), (4, 5)]


def test_over_budget_word_gets_own_chunk():
    assert chunk_spans([2, 50, 2], 10) == [(0, 1), (1, 2), (2, 3)]


def test_spans_cover_all_words_contiguously():
    costs = [5, 1, 9, 2, 2, 2, 7, 1, 1]
    spans = chunk_spans(costs, 10)
    assert spans[0][0] == 0 and spans[-1][1] == len(costs)
    for (_, a_end), (b_start, _) in zip(spans, spans[1:]):
        assert a_end == b_start
    for start, end in spans:
        if end - start > 1:  # multi-word chunks respect the budget
            assert sum(costs[start:end]) <= 10


def test_zero_budget_rejected():
    with pytest.raises(ValueError):
        chunk_spans([1], 0)
