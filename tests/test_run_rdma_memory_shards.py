from __future__ import annotations

from scripts.run_rdma_memory_shards import parse_result


def test_parse_result_uses_last_json_line() -> None:
    output = 'progress\n{"ready":true,"bytes":42}\n'
    assert parse_result(output) == {"ready": True, "bytes": 42}
