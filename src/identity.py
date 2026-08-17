"""身份与工装状态管理层（位于 ByteTrack 之上）。

两条领域先验：
1. 员工穿了工装就不会脱：
   每个 person 的工装标签用滑动窗口概率判定，概率足够高时锁定为“已穿工装”；
   “未穿工装”不锁定（可能被遮挡误判），后续证据充分可升级为“已穿”。
2. 员工不会凭空消失（走出边界/完全遮挡除外）：
   轨迹消失后暂存一段时间，新检测框位置匹配则复用原 person_id 与工装状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .uniform import UniformResult


UNIFORM_LOCKED = "locked"
UNIFORM_ACTIVE = "active"


@dataclass
class PersonObject:
    """IdentityTracker 输出的稳定人员对象。"""

    person_id: int
    bt_id: int                      # 底层 ByteTrack track_id（调试用）
    xyxy: np.ndarray
    confidence: float
    uniform: bool                   # 最终工装标签（锁定或实时）
    uniform_state: str              # "locked" / "active"
    uniform_prob: float             # 当前工装概率（0-1）


@dataclass
class _StoredTrack:
    """暂存轨迹（用于遮挡后恢复 ID）。"""

    person_id: int
    last_center: np.ndarray         # 最后位置（中心点）
    last_box: np.ndarray            # 最后检测框
    last_velocity: np.ndarray       # 外推速度（中心点，像素/帧）
    last_frame_id: int
    uniform: bool
    uniform_state: str
    uniform_prob: float
    uniform_history: list[bool] = field(default_factory=list)


@dataclass
class _PersonState:
    """单个人员的跨帧状态。"""

    person_id: int
    uniform_history: list[bool] = field(default_factory=list)
    uniform: bool | None = None
    uniform_state: str = UNIFORM_ACTIVE
    uniform_prob: float = 0.5
    last_center: np.ndarray | None = None
    last_box: np.ndarray | None = None
    last_velocity: np.ndarray | None = None
    last_frame_id: int = -1
    missing_frames: int = 0

    def update_motion(self, center: np.ndarray, box: np.ndarray, frame_id: int) -> None:
        """更新位置与速度（用于遮挡期间的轨迹外推）。"""
        if self.last_center is not None and frame_id > self.last_frame_id:
            dt = max(frame_id - self.last_frame_id, 1)
            self.last_velocity = (center - self.last_center) / dt
        else:
            self.last_velocity = np.zeros(2)
        self.last_center = center
        self.last_box = box
        self.last_frame_id = frame_id
        self.missing_frames = 0

    def predicted_center(self, frame_id: int) -> np.ndarray:
        """按最后速度外推当前位置；无速度时返回最后位置。"""
        if self.last_center is None:
            return np.zeros(2)
        if self.last_velocity is not None and frame_id > self.last_frame_id:
            return self.last_center + self.last_velocity * (frame_id - self.last_frame_id)
        return self.last_center


class IdentityTracker:
    """稳定身份与工装状态管理。

    职责：
    - 维护 ByteTrack track_id → 稳定 person_id 的映射；
    - 工装标签滑动窗口概率判定 + 已穿单向锁定；
    - 轨迹暂存恢复（位置匹配复用旧 ID 与工装状态）。
    """

    def __init__(
        self,
        *,
        uniform_window: int = 10,
        uniform_lock_high: float = 0.8,
        track_ttl_frames: int = 90,          # 3 秒 @30fps
        match_distance_factor: float = 1.5,  # 匹配距离 = 框宽 * factor
    ) -> None:
        self.uniform_window = uniform_window
        self.uniform_lock_high = uniform_lock_high
        self.track_ttl_frames = track_ttl_frames
        self.match_distance_factor = match_distance_factor

        self._bt_to_person: dict[int, int] = {}
        self._persons: dict[int, _PersonState] = {}
        self._stored: dict[int, _StoredTrack] = {}
        self._next_person_id = 1

    def reset(self) -> None:
        self._bt_to_person.clear()
        self._persons.clear()
        self._stored.clear()
        self._next_person_id = 1

    def _update_uniform(
        self,
        state: _PersonState,
        raw_uniform: bool,
        raw_result: UniformResult,
    ) -> None:
        """滑动窗口概率更新 + 单向锁定（已穿锁定，未穿可升级）。"""
        if state.uniform_state == UNIFORM_LOCKED:
            return  # 已穿工装：永久锁定，不再变化
        state.uniform_history.append(raw_uniform)
        if len(state.uniform_history) > self.uniform_window:
            state.uniform_history.pop(0)
        prob = sum(state.uniform_history) / len(state.uniform_history)
        state.uniform_prob = prob
        if len(state.uniform_history) >= self.uniform_window:
            if prob >= self.uniform_lock_high:
                state.uniform = True
                state.uniform_state = UNIFORM_LOCKED
            else:
                state.uniform = prob >= 0.5

    def _try_restore(self, center: np.ndarray, box: np.ndarray, frame_id: int) -> int | None:
        """尝试把新检测框与暂存轨迹匹配；成功返回 person_id。"""
        if not self._stored:
            return None
        box_w = max(box[2] - box[0], 1.0)
        max_dist = box_w * self.match_distance_factor
        best_id: int | None = None
        best_dist = float("inf")
        for pid, st in self._stored.items():
            dist = float(np.linalg.norm(center - st.last_center))
            if dist <= max_dist and dist < best_dist:
                best_id = pid
                best_dist = dist
        if best_id is not None:
            st = self._stored.pop(best_id)
            state = _PersonState(
                person_id=best_id,
                uniform_history=st.uniform_history,
                uniform=st.uniform,
                uniform_state=st.uniform_state,
                uniform_prob=st.uniform_prob,
            )
            self._persons[best_id] = state
            return best_id
        return None

    def update(
        self,
        frame_id: int,
        tracked: list[Any],
        uniform_results: dict[int, UniformResult],
    ) -> list[PersonObject]:
        """输入 ByteTrack 目标列表 + 工装判定结果，输出稳定人员列表。"""
        # 1. 清理过期暂存轨迹
        expired = [
            pid for pid, st in self._stored.items()
            if frame_id - st.last_frame_id > self.track_ttl_frames
        ]
        for pid in expired:
            del self._stored[pid]
            self._persons.pop(pid, None)

        # 2. 处理当前帧检测框
        new_person_ids: set[int] = set()
        objects: list[PersonObject] = []
        for obj in tracked:
            bt_id = obj.track_id
            box = obj.xyxy
            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

            person_id = self._bt_to_person.get(bt_id)
            if person_id is None:
                # 新 track_id：先尝试与暂存轨迹匹配，匹配不上才新建
                person_id = self._try_restore(center, box, frame_id)
                if person_id is None:
                    person_id = self._next_person_id
                    self._next_person_id += 1
                    self._persons[person_id] = _PersonState(person_id=person_id)
                self._bt_to_person[bt_id] = person_id

            state = self._persons[person_id]
            state.update_motion(center, box, frame_id)
            new_person_ids.add(person_id)

            raw = uniform_results.get(bt_id)
            if raw is not None:
                self._update_uniform(state, raw.is_uniform, raw)

            uniform = bool(state.uniform) if state.uniform is not None else raw.is_uniform if raw else False
            objects.append(
                PersonObject(
                    person_id=person_id,
                    bt_id=bt_id,
                    xyxy=box,
                    confidence=obj.confidence,
                    uniform=uniform,
                    uniform_state=state.uniform_state,
                    uniform_prob=state.uniform_prob,
                )
            )

        # 3. 本次帧未出现的活跃人员 → 移入暂存池
        for pid in list(self._persons.keys()):
            if pid not in new_person_ids:
                state = self._persons[pid]
                if state.last_center is not None:
                    self._stored[pid] = _StoredTrack(
                        person_id=pid,
                        last_center=state.last_center,
                        last_box=state.last_box,
                        last_velocity=(
                            state.last_velocity
                            if state.last_velocity is not None
                            else np.zeros(2)
                        ),
                        last_frame_id=state.last_frame_id,
                        uniform=bool(state.uniform) if state.uniform is not None else False,
                        uniform_state=state.uniform_state,
                        uniform_prob=state.uniform_prob,
                        uniform_history=list(state.uniform_history),
                    )
                del self._persons[pid]

        # 4. 清理已消失的 bt 映射
        active_bt = {o.track_id for o in tracked}
        self._bt_to_person = {
            bt: pid for bt, pid in self._bt_to_person.items() if bt in active_bt
        }
        return objects
