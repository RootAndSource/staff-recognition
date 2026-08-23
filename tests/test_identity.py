"""identity.py 单元测试：延迟态状态机 + 画面内恢复 + 边界释放 + 确认期。"""

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


def _frame(w: int = 400, h: int = 300) -> np.ndarray:
    """合成空白帧（灰底）。"""
    return np.full((h, w, 3), 128, dtype=np.uint8)


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


def _tr(**kw) -> IdentityTracker:
    defaults = dict(
        uniform_pending_frames=2,
        uniform_window=3,
        uniform_confirm_frames=4,
        uniform_confirm_high=0.9,
        reverse_streak_frames=3,
        track_ttl_frames=300,
        boundary_ttl_frames=60,
        match_distance_factor=2.5,
        appearance_corr_threshold=0.85,
        person_confirm_frames=3,
        leave_visible_ratio=0.4,
    )
    defaults.update(kw)
    return IdentityTracker(**defaults)


def test_cold_start_outputs_label_immediately() -> None:
    # 新设计：冷启动直接出标签（无 pending），记忆从低置信开始
    tr = _tr(uniform_pending_frames=5, person_confirm_frames=1)
    persons = tr.update(0, _frame(), [_obj(1, [50, 50, 150, 250])], {1: _uniform(True)})
    assert persons[0].uniform is True
    assert persons[0].uniform_state == UNIFORM_ACTIVE


def test_uniform_label_stays_without_flip() -> None:
    tr = _tr()
    # 冷启动即出标签，持续穿工装建立记忆
    for i in range(5):
        persons = tr.update(i, _frame(), [_obj(1, [50, 50, 150, 250])], {1: _uniform(True)})
    assert persons[0].uniform is True
    assert persons[0].uniform_state == UNIFORM_ACTIVE
    # 2 帧反向证据（不足以翻转）→ 保持穿
    for i in range(5, 7):
        persons = tr.update(i, _frame(), [_obj(1, [50, 50, 150, 250])], {1: _uniform(False)})
    assert persons[0].uniform is True
    # 第 3 帧反向：未串->串基准 3 帧可翻，但记忆压制仍保持穿（排除换装）
    persons = tr.update(7, _frame(), [_obj(1, [50, 50, 150, 250])], {1: _uniform(False)})
    assert persons[0].uniform is True
    # 持续反向直到超过动态阈值才翻转
    for i in range(8, 70):
        persons = tr.update(i, _frame(), [_obj(1, [50, 50, 150, 250])], {1: _uniform(False)})
    assert persons[0].uniform is False


def test_uniform_interrupted_evidence_does_not_flip() -> None:
    tr = _tr()
    for i in range(5):
        persons = tr.update(i, _frame(), [_obj(1, [50, 50, 150, 250])], {1: _uniform(True)})
    assert persons[0].uniform is True
    # 1 帧反向 + 1 帧正向 + 1 帧反向（不连续）→ 保持已穿
    for i, u in [(5, False), (6, True), (7, False)]:
        persons = tr.update(i, _frame(), [_obj(1, [50, 50, 150, 250])], {1: _uniform(u)})
    assert persons[0].uniform is True


def test_uniform_upgrade_from_no_uniform() -> None:
    tr = _tr()
    for i in range(5):
        persons = tr.update(i, _frame(), [_obj(1, [50, 50, 150, 250])], {1: _uniform(False)})
    assert persons[0].uniform is False
    assert persons[0].uniform_state == UNIFORM_ACTIVE
    # 连续 3 帧反向证据 → 升级为已穿
    for i in range(5, 8):
        persons = tr.update(i, _frame(), [_obj(1, [50, 50, 150, 250])], {1: _uniform(True)})
    assert persons[0].uniform is True


def test_uniform_confirmed_marker() -> None:
    tr = _tr()
    # 持续工装超过 confirm_frames → CONFIRMED 标记
    for i in range(10):
        persons = tr.update(i, _frame(), [_obj(1, [50, 50, 150, 250])], {1: _uniform(True)})
    assert persons[0].uniform is True
    assert persons[0].uniform_state == UNIFORM_CONFIRMED
    # 连续 3 帧反向不足以翻转（记忆压制）
    for i in range(10, 13):
        persons = tr.update(i, _frame(), [_obj(1, [50, 50, 150, 250])], {1: _uniform(False)})
    assert persons[0].uniform is True
    # 持续反向超过动态阈值才翻转（不锁死）
    for i in range(13, 80):
        persons = tr.update(i, _frame(), [_obj(1, [50, 50, 150, 250])], {1: _uniform(False)})
    assert persons[0].uniform is False


def test_track_restore_reuses_person_id() -> None:
    tr = _tr(track_ttl_frames=10, person_confirm_frames=1)
    # 帧 0-2：bt_id=1 → person_id=1
    for i in range(3):
        persons = tr.update(i, _frame(), [_obj(1, [150 + i * 5, 100, 250 + i * 5, 250])], {})
    first_id = persons[0].person_id
    assert first_id == 1
    # 帧 3-5：目标消失（bt_id=1 不再出现）
    for i in range(3, 6):
        tr.update(i, _frame(), [], {})
    # 帧 6：重新出现，位置接近（bt_id=2 是 ByteTrack 新 id）→ 恢复 person_id=1
    persons = tr.update(6, _frame(), [_obj(2, [170, 100, 270, 250])], {})
    assert persons[0].person_id == first_id
    assert persons[0].bt_id == "t2"
    assert persons[0].restored is True


def test_track_ttl_expires() -> None:
    tr = _tr(track_ttl_frames=5, person_confirm_frames=1)
    for i in range(3):
        tr.update(i, _frame(), [_obj(1, [100, 100, 200, 250])], {})
    for i in range(3, 9):
        tr.update(i, _frame(), [], {})
    # 超时后重新出现 → 新建 person_id
    persons = tr.update(9, _frame(), [_obj(2, [100, 100, 200, 250])], {})
    assert persons[0].person_id != 1


def test_boundary_release_allows_new_id() -> None:
    tr = _tr(person_confirm_frames=1)
    # 框大部分出画面（可见 <40%）→ 离开，释放不暂存
    for i in range(3):
        tr.update(i, _frame(), [_obj(1, [-50, 100, 30, 250])], {})
    for i in range(3, 6):
        tr.update(i, _frame(), [], {})
    # 同位置重新出现 → 新 ID（不是恢复）
    persons = tr.update(6, _frame(), [_obj(2, [-50, 100, 30, 250])], {})
    assert persons[0].person_id != 1
    assert persons[0].restored is False


def test_clipped_top_box_not_released() -> None:
    """画面顶部被裁切的正常目标（可见完整）不算离开，消失后应恢复。"""
    tr = _tr(track_ttl_frames=90, person_confirm_frames=1)
    for i in range(3):
        tr.update(i, _frame(), [_obj(1, [50, -10, 150, 80])], {})  # y1 出画面但大部分可见
    for i in range(3, 20):
        tr.update(i, _frame(), [], {})
    persons = tr.update(20, _frame(), [_obj(2, [50, -10, 150, 80])], {})
    assert persons[0].person_id == 1
    assert persons[0].restored is True


def test_person_confirm_period_skips_flashing_objects() -> None:
    tr = _tr(person_confirm_frames=3)
    # 帧 0-1：新目标出现但未满确认期 → 不输出
    persons = tr.update(0, _frame(), [_obj(1, [100, 100, 200, 250])], {})
    assert persons == []
    persons = tr.update(1, _frame(), [_obj(1, [100, 100, 200, 250])], {})
    assert persons == []
    # 帧 2：连续第 3 帧 → 分配正式 ID
    persons = tr.update(2, _frame(), [_obj(1, [100, 100, 200, 250])], {})
    assert len(persons) == 1
    assert persons[0].person_id == 1


def test_inner_disappearance_restores_even_if_bt_id_changes() -> None:
    tr = _tr(track_ttl_frames=90, person_confirm_frames=1)
    # 画面内（非边界）目标
    for i in range(3):
        tr.update(i, _frame(), [_obj(1, [150, 100, 250, 250])], {})
    for i in range(3, 20):
        tr.update(i, _frame(), [], {})
    # 20 帧后（< 90 帧 TTL）位置略移的新 bt_id → 必须恢复原 ID（画面内不分裂）
    persons = tr.update(20, _frame(), [_obj(2, [160, 100, 260, 250])], {})
    assert persons[0].person_id == 1
    assert persons[0].restored is True
