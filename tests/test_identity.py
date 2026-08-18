"""identity.py 单元测试：延迟态状态机 + 暂存轨迹恢复。"""

from __future__ import annotations

import numpy as np

from src.identity import (
    IdentityTracker,
    UNIFORM_ACTIVE,
    UNIFORM_CONFIRMED,
    UNIFORM_PENDING,
)
from src.tracker import TrackedObject
from src.uniform import UniformResult


def _obj(bt_id: int, xyxy, conf: float = 0.8) -> TrackedObject:
    return TrackedObject(
        track_id=bt_id,
        xyxy=np.array(xyxy, dtype=float),
        confidence=conf,
        is_confirmed=True,
    )


def _uniform(is_uniform: bool) -> UniformResult:
    return UniformResult(
        is_uniform=is_uniform,
        red_ratio=0.9 if is_uniform else 0.01,
        red_ratio_main=0.9 if is_uniform else 0.01,
        red_ratio_lower=0.9 if is_uniform else 0.01,
        red_ratio_connected=0.9 if is_uniform else 0.01,
        baseline_ratio=0.1,
        roi_main=(0, 0, 10, 10),
        roi_lower=(0, 0, 10, 10),
    )


def test_pending_period_outputs_none() -> None:
    tr = IdentityTracker(uniform_pending_frames=5, uniform_window=3)
    for i in range(4):  # 观察期内
        persons = tr.update(i, [_obj(1, [0, 0, 100, 200])], {1: _uniform(True)})
    assert persons[0].uniform is None
    assert persons[0].uniform_state == UNIFORM_PENDING


def test_uniform_label_stays_without_flip() -> None:
    tr = IdentityTracker(uniform_pending_frames=2, uniform_window=3, reverse_streak_frames=3)
    # 前 2 帧观察期 + 3 帧工装 → ACTIVE 已穿
    for i in range(5):
        persons = tr.update(i, [_obj(1, [0, 0, 100, 200])], {1: _uniform(True)})
    assert persons[0].uniform is True
    assert persons[0].uniform_state == UNIFORM_ACTIVE
    # 2 帧反向证据（< 3 帧阈值）→ 不翻转
    for i in range(5, 7):
        persons = tr.update(i, [_obj(1, [0, 0, 100, 200])], {1: _uniform(False)})
    assert persons[0].uniform is True
    # 第 3 帧连续反向 → 翻转
    persons = tr.update(7, [_obj(1, [0, 0, 100, 200])], {1: _uniform(False)})
    assert persons[0].uniform is False


def test_uniform_interrupted_evidence_does_not_flip() -> None:
    tr = IdentityTracker(uniform_pending_frames=2, uniform_window=3, reverse_streak_frames=3)
    for i in range(5):
        persons = tr.update(i, [_obj(1, [0, 0, 100, 200])], {1: _uniform(True)})
    assert persons[0].uniform is True
    # 1 帧反向 + 1 帧正向 + 1 帧反向（不连续）→ 保持已穿
    for i, u in [(5, False), (6, True), (7, False)]:
        persons = tr.update(i, [_obj(1, [0, 0, 100, 200])], {1: _uniform(u)})
    assert persons[0].uniform is True


def test_uniform_upgrade_from_no_uniform() -> None:
    tr = IdentityTracker(uniform_pending_frames=2, uniform_window=3, reverse_streak_frames=3)
    for i in range(5):
        persons = tr.update(i, [_obj(1, [0, 0, 100, 200])], {1: _uniform(False)})
    assert persons[0].uniform is False
    assert persons[0].uniform_state == UNIFORM_ACTIVE
    # 连续 3 帧反向证据 → 升级为已穿
    for i in range(5, 8):
        persons = tr.update(i, [_obj(1, [0, 0, 100, 200])], {1: _uniform(True)})
    assert persons[0].uniform is True


def test_uniform_confirmed_marker() -> None:
    tr = IdentityTracker(
        uniform_pending_frames=2,
        uniform_window=3,
        uniform_confirm_frames=4,
        uniform_confirm_high=0.9,
        reverse_streak_frames=3,
    )
    # 持续工装超过 confirm_frames → CONFIRMED 标记（不阻止翻转）
    for i in range(10):
        persons = tr.update(i, [_obj(1, [0, 0, 100, 200])], {1: _uniform(True)})
    assert persons[0].uniform is True
    assert persons[0].uniform_state == UNIFORM_CONFIRMED
    # 连续 3 帧反向 → 仍可翻转（高置信不锁定）
    for i in range(10, 13):
        persons = tr.update(i, [_obj(1, [0, 0, 100, 200])], {1: _uniform(False)})
    assert persons[0].uniform is False


def test_track_restore_reuses_person_id() -> None:
    tr = IdentityTracker(track_ttl_frames=10, match_distance_factor=1.5)
    # 帧 0-2：bt_id=1 → person_id=1
    for i in range(3):
        persons = tr.update(i, [_obj(1, [100 + i * 5, 100, 200 + i * 5, 300])], {})
    first_id = persons[0].person_id
    assert first_id == 1
    # 帧 3-5：目标消失（bt_id=1 不再出现）
    for i in range(3, 6):
        tr.update(i, [], {})
    # 帧 6：重新出现，位置接近（bt_id=2 是 ByteTrack 新 id）→ 恢复 person_id=1
    persons = tr.update(6, [_obj(2, [130, 100, 230, 300])], {})
    assert persons[0].person_id == first_id
    assert persons[0].bt_id == 2


def test_track_ttl_expires() -> None:
    tr = IdentityTracker(track_ttl_frames=5, match_distance_factor=1.5)
    for i in range(3):
        tr.update(i, [_obj(1, [100, 100, 200, 300])], {})
    for i in range(3, 9):
        tr.update(i, [], {})
    # 超时后重新出现 → 新建 person_id
    persons = tr.update(9, [_obj(2, [100, 100, 200, 300])], {})
    assert persons[0].person_id != 1
