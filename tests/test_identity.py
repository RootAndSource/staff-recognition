"""identity.py 单元测试：工装概率双向锁定 + 暂存轨迹恢复。"""

from __future__ import annotations

import numpy as np

from src.identity import IdentityTracker, UNIFORM_ACTIVE, UNIFORM_LOCKED
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
        roi_main=(0, 0, 10, 10),
        roi_lower=(0, 0, 10, 10),
    )


def test_uniform_locks_high_and_stays() -> None:
    tr = IdentityTracker(uniform_window=4, uniform_lock_high=0.8)
    # 前 4 帧全部穿工装 → 锁定已穿
    for i in range(4):
        persons = tr.update(i, [_obj(1, [0, 0, 100, 200])], {1: _uniform(True)})
    assert persons[0].uniform is True
    assert persons[0].uniform_state == UNIFORM_LOCKED
    # 后续帧红色消失（趴姿被遮挡）→ 标签保持已穿（不会翻转）
    for i in range(4, 8):
        persons = tr.update(i, [_obj(1, [0, 0, 100, 200])], {1: _uniform(False)})
    assert persons[0].uniform is True
    assert persons[0].uniform_state == UNIFORM_LOCKED


def test_uniform_low_is_not_locked_and_can_upgrade() -> None:
    tr = IdentityTracker(uniform_window=4, uniform_lock_high=0.8)
    for i in range(4):
        persons = tr.update(i, [_obj(1, [0, 0, 100, 200])], {1: _uniform(False)})
    assert persons[0].uniform is False
    assert persons[0].uniform_state == UNIFORM_ACTIVE  # 未穿不锁定
    # 后续帧红色证据充分 → 升级为已穿并锁定
    for i in range(4, 8):
        persons = tr.update(i, [_obj(1, [0, 0, 100, 200])], {1: _uniform(True)})
    assert persons[0].uniform is True
    assert persons[0].uniform_state == UNIFORM_LOCKED


def test_uniform_mixed_window_uses_probability() -> None:
    tr = IdentityTracker(uniform_window=6, uniform_lock_high=0.8)
    # 6 帧中 4 帧工装：概率 0.67，中间态 → 不锁定，标签取多数
    seq = [True, True, False, True, False, True]
    for i, u in enumerate(seq):
        persons = tr.update(i, [_obj(1, [0, 0, 100, 200])], {1: _uniform(u)})
    assert persons[0].uniform_state == UNIFORM_ACTIVE
    assert persons[0].uniform is True
    assert abs(persons[0].uniform_prob - 4 / 6) < 1e-6


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
