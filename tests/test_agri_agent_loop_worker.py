from agrimanager.adapter.agent_loop.worker import _select_even_worker_count


def test_select_even_worker_count_keeps_divisible_batches():
    assert _select_even_worker_count(batch_size=16, max_workers=8) == 8


def test_select_even_worker_count_handles_small_validation_batches():
    assert _select_even_worker_count(batch_size=3, max_workers=8) == 3


def test_select_even_worker_count_uses_largest_even_divisor():
    assert _select_even_worker_count(batch_size=9, max_workers=8) == 3
    assert _select_even_worker_count(batch_size=17, max_workers=8) == 1
