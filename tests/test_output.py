"""output.py 纯逻辑单测。"""

from __future__ import annotations

import json

import numpy as np

from src.output import JsonOutputWriter, build_person_record
from src.tracker import TrackedObject
from src.uniform import UniformResult


def test_json_writer_roundtrip(tmp_path) -> None:
    p = tmp_path / "out.jsonl"
    with JsonOutputWriter(p) as w:
        w.write_frame(
                frame_id=0,
                timestamp_ms=33.3,
                records=[build_person_record(
                    TrackedObject(track_id=1, xyxy=np.array([0, 0, 10, 20]), confidence=0.9, is_confirmed=True),
                    True,
                    uniform_result=(
                    UniformResult(
                        is_uniform=True,
                        red_ratio=0.5,
                        red_ratio_main=0.5,
                        red_ratio_lower=0.4,
                        roi_main=(0, 0, 10, 20),
                        roi_lower=(0, 0, 10, 20),
                    )
                    ),
                )],
            )
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["people"][0]["person_id"] == 1
    assert data[0]["people"][0]["uniform"] is True
    assert data[0]["people"][0]["bbox"] == [0.0, 0.0, 10.0, 20.0]
    assert data[0]["people"][0]["red_ratio_lower"] == 0.4


def test_json_writer_empty_ok(tmp_path) -> None:
    p = tmp_path / "empty.jsonl"
    with JsonOutputWriter(p) as w:
        w.write_frame(frame_id=0, timestamp_ms=0.0, records=[])
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data == [{"frame_id": 0, "timestamp_ms": 0.0, "utc_time": data[0]["utc_time"], "people": []}]
