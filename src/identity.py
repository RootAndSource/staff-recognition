"""身份与工装状态管理层（位于 ByteTrack 之上）。

两条领域先验：
1. 画面内的员工不会凭空消失：目标消失后暂存（10 秒）并做位置/外观匹配，
   重新出现必须复用原 person_id；只有走出画面边界才允许释放（再进来是新 ID）。
2. 工装标签用延迟态状态机（pending/active/confirmed），翻转需要连续反向证据，
   不永久锁定、可被纠正。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .uniform import UniformResult


UNIFORM_PENDING = "pending"
UNIFORM_ACTIVE = "active"
UNIFORM_CONFIRMED = "confirmed"


@dataclass
class PersonObject:
    """IdentityTracker 输出的稳定人员对象。"""

    person_id: int
    bt_id: str                      # 底层 ByteTrack track_id（调试用）
    xyxy: np.ndarray
    confidence: float
    uniform: bool | None            # 工装标签（pending 阶段为 None）
    uniform_state: str              # "pending" / "active" / "confirmed"
    uniform_prob: float             # 当前工装概率（0-1）
    restored: bool = False          # 本帧是否从暂存恢复
    restored_from: int | None = None  # 恢复来源 person_id
    red_ratio_connected: float = 0.0   # 最大红色连通域占比（判定依据）
    baseline_ratio: float = 0.0        # 场景红色基线（判定依据）


@dataclass
class _StoredTrack:
    """暂存轨迹（画面内消失 → 等待恢复）。"""

    person_id: int
    last_center: np.ndarray
    last_box: np.ndarray
    last_velocity: np.ndarray
    last_frame_id: int
    appearance: np.ndarray | None     # HSV 直方图外观签名
    uniform: bool | None
    uniform_state: str
    uniform_prob: float
    uniform_history: list[bool] = field(default_factory=list)
    reverse_streak: int = 0
    confirmed_frames: int = 0


@dataclass
class _PendingCandidate:
    """新 track_id 确认期候选（连续 N 帧稳定才分配正式 person_id）。"""

    bt_id: str
    first_frame: int
    last_frame: int
    last_center: np.ndarray
    count: int = 1


@dataclass
class _PersonState:
    """单个人员的跨帧状态。"""

    person_id: int
    uniform_history: list[bool] = field(default_factory=list)
    uniform: bool | None = None
    uniform_state: str = UNIFORM_ACTIVE
    uniform_prob: float = 0.5
    reverse_streak: int = 0
    confirmed_frames: int = 0
    total_frames: int = 0
    last_center: np.ndarray | None = None
    last_box: np.ndarray | None = None
    last_velocity: np.ndarray | None = None
    last_frame_id: int = -1

    def update_motion(self, center: np.ndarray, box: np.ndarray, frame_id: int) -> None:
        if self.last_center is not None and frame_id > self.last_frame_id:
            dt = max(frame_id - self.last_frame_id, 1)
            self.last_velocity = (center - self.last_center) / dt
        else:
            self.last_velocity = np.zeros(2)
        self.last_center = center
        self.last_box = box
        self.last_frame_id = frame_id

    def predicted_center(self, frame_id: int) -> np.ndarray:
        if self.last_center is None:
            return np.zeros(2)
        if self.last_velocity is not None and frame_id > self.last_frame_id:
            return self.last_center + self.last_velocity * (frame_id - self.last_frame_id)
        return self.last_center


class IdentityTracker:
    """稳定身份与工装状态管理。

    职责：
    - 维护 ByteTrack track_id → 稳定 person_id 的映射；
    - 画面内消失 → 暂存并恢复（位置 + 外观），边界离开 → 释放；
    - 新 track_id 先观察确认，避免短命误检消耗 ID；
    - 工装标签延迟态状态机（PENDING/ACTIVE/CONFIRMED）。
    """

    def __init__(
        self,
        *,
        uniform_pending_frames: int = 10,
        uniform_window: int = 30,
        uniform_confirm_frames: int = 60,
        uniform_confirm_high: float = 0.9,
        reverse_streak_frames: int = 5,
        track_ttl_frames: int = 300,       # 画面内消失暂存 10 秒 @30fps
        boundary_ttl_frames: int = 60,     # 边界附近消失暂存 2 秒
        match_distance_factor: float = 2.5,   # 位置匹配距离 = 框宽 * factor
        appearance_corr_threshold: float = 0.85,  # 外观相关性兜底阈值
        person_confirm_frames: int = 3,    # 新 ID 确认期（连续稳定帧数）
        leave_visible_ratio: float = 0.4,  # 检测框可见面积 < 该比例才视为走出画面
    ) -> None:
        self.uniform_pending_frames = uniform_pending_frames
        self.uniform_window = uniform_window
        self.uniform_confirm_frames = uniform_confirm_frames
        self.uniform_confirm_high = uniform_confirm_high
        self.reverse_streak_frames = reverse_streak_frames
        self.track_ttl_frames = track_ttl_frames
        self.boundary_ttl_frames = boundary_ttl_frames
        self.match_distance_factor = match_distance_factor
        self.appearance_corr_threshold = appearance_corr_threshold
        self.person_confirm_frames = person_confirm_frames
        self.leave_visible_ratio = leave_visible_ratio

        self._bt_to_person: dict[str, int] = {}
        self._persons: dict[int, _PersonState] = {}
        self._stored: dict[int, _StoredTrack] = {}
        self._pending: dict[str, _PendingCandidate] = {}
        self._next_person_id = 1
        self.stats = {"restored": 0, "released_boundary": 0, "new_persons": 0}

    def reset(self) -> None:
        self._bt_to_person.clear()
        self._persons.clear()
        self._stored.clear()
        self._pending.clear()
        self._next_person_id = 1
        self.stats = {"restored": 0, "released_boundary": 0, "new_persons": 0}

    # ---------- 外观特征 ----------

    @staticmethod
    def _extract_appearance(
        frame_bgr: cv2.Mat, box: np.ndarray, frame_w: int, frame_h: int
    ) -> np.ndarray | None:
        """检测框 ROI 的 HSV 3D 直方图（16x16x8），归一化后作为外观签名。"""
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [16, 16, 8], [0, 180, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist

    @staticmethod
    def _appearance_corr(a: np.ndarray | None, b: np.ndarray | None) -> float:
        """两个外观签名的相关性（0-1）。"""
        if a is None or b is None:
            return 0.0
        return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))

    # ---------- 工装延迟态 ----------

    def _update_uniform(
        self,
        state: _PersonState,
        raw_uniform: bool,
        raw_result: UniformResult,
    ) -> None:
        """延迟态状态机：PENDING → ACTIVE → CONFIRMED，翻转需要连续反向证据。"""
        state.total_frames += 1
        if state.total_frames < self.uniform_pending_frames:
            state.uniform_state = UNIFORM_PENDING
            state.uniform = None
            state.uniform_prob = 0.5
            return

        state.uniform_history.append(raw_uniform)
        if len(state.uniform_history) > self.uniform_window:
            state.uniform_history.pop(0)
        prob = sum(state.uniform_history) / len(state.uniform_history)
        state.uniform_prob = prob

        if state.uniform is None:
            state.uniform = raw_uniform
            state.uniform_state = UNIFORM_ACTIVE
            state.reverse_streak = 0
            state.confirmed_frames = 0
            return

        if raw_uniform != state.uniform:
            state.reverse_streak += 1
            if state.reverse_streak >= self.reverse_streak_frames:
                state.uniform = raw_uniform
                state.reverse_streak = 0
                state.confirmed_frames = 0
        else:
            state.reverse_streak = 0
            if state.uniform is not None:
                state.uniform_history = [
                    u for u in state.uniform_history if u == state.uniform
                ] or [state.uniform]

        if state.uniform_state != UNIFORM_CONFIRMED:
            state.uniform_state = UNIFORM_ACTIVE

        if state.uniform is not None:
            consistent = (
                state.reverse_streak == 0
                and sum(state.uniform_history) >= self.uniform_confirm_high * len(state.uniform_history)
            )
            if consistent:
                state.confirmed_frames += 1
                if state.confirmed_frames >= self.uniform_confirm_frames:
                    state.uniform_state = UNIFORM_CONFIRMED
            else:
                state.confirmed_frames = 0
                if state.uniform_state == UNIFORM_CONFIRMED:
                    state.uniform_state = UNIFORM_ACTIVE

    # ---------- 暂存与恢复 ----------

    def _is_leaving_frame(
        self, box: np.ndarray, frame_w: int, frame_h: int
    ) -> bool:
        """检测框大部分已出画面 → 视为走出画面（被画面裁切的正常目标不算）。"""
        x1, y1, x2, y2 = box
        vis_w = max(0.0, min(x2, frame_w) - max(x1, 0.0))
        vis_h = max(0.0, min(y2, frame_h) - max(y1, 0.0))
        visible = vis_w * vis_h
        total = max((x2 - x1) * (y2 - y1), 1e-6)
        return visible / total < self.leave_visible_ratio

    def _try_restore(
        self,
        frame_bgr: cv2.Mat,
        center: np.ndarray,
        box: np.ndarray,
        frame_id: int,
        frame_w: int,
        frame_h: int,
    ) -> int | None:
        """画面内消失的轨迹 → 位置 + 外观匹配恢复原 person_id。"""
        if not self._stored:
            return None
        box_w = max(box[2] - box[0], 1.0)
        max_dist = box_w * self.match_distance_factor
        appearance = self._extract_appearance(frame_bgr, box, frame_w, frame_h)

        best_id: int | None = None
        best_dist = float("inf")
        best_corr = 0.0
        for pid, st in self._stored.items():
            pred = st.last_center
            if st.last_velocity is not None and frame_id > st.last_frame_id:
                pred = st.last_center + st.last_velocity * (frame_id - st.last_frame_id)
            dist = float(np.linalg.norm(center - pred))
            corr = self._appearance_corr(appearance, st.appearance)
            if dist <= max_dist:
                if dist < best_dist:
                    best_id, best_dist = pid, dist
            elif dist <= max_dist * 2.0 and corr >= self.appearance_corr_threshold:
                if corr > best_corr:
                    best_id, best_corr = pid, corr

        if best_id is None:
            return None
        st = self._stored.pop(best_id)
        state = _PersonState(
            person_id=best_id,
            uniform_history=st.uniform_history,
            uniform=st.uniform,
            uniform_state=st.uniform_state,
            uniform_prob=st.uniform_prob,
            reverse_streak=st.reverse_streak,
            confirmed_frames=st.confirmed_frames,
            total_frames=len(st.uniform_history),
        )
        self._persons[best_id] = state
        self.stats["restored"] += 1
        return best_id

    def _store_or_release(
        self,
        frame_bgr: cv2.Mat,
        pid: int,
        state: _PersonState,
        frame_id: int,
        frame_w: int,
        frame_h: int,
    ) -> None:
        """人员消失：画面内 → 暂存等待恢复；边界附近 → 释放（允许新 ID）。"""
        if state.last_center is None:
            return
        if self._is_leaving_frame(state.last_box, frame_w, frame_h):
            self.stats["released_boundary"] += 1
            return
        appearance = (
            self._extract_appearance(frame_bgr, state.last_box, frame_w, frame_h)
            if state.last_box is not None else None
        )
        self._stored[pid] = _StoredTrack(
            person_id=pid,
            last_center=state.last_center,
            last_box=state.last_box,
            last_velocity=(
                state.last_velocity if state.last_velocity is not None else np.zeros(2)
            ),
            last_frame_id=state.last_frame_id,
            appearance=appearance,
            uniform=state.uniform,
            uniform_state=state.uniform_state,
            uniform_prob=state.uniform_prob,
            uniform_history=list(state.uniform_history),
            reverse_streak=state.reverse_streak,
            confirmed_frames=state.confirmed_frames,
        )

    def _prune_stored(self, frame_id: int, frame_w: int, frame_h: int) -> None:
        """清理过期暂存轨迹（画面内 10 秒 / 边界 2 秒）。"""
        expired = []
        for pid, st in self._stored.items():
            near_edge = self._is_leaving_frame(st.last_box, frame_w, frame_h)
            ttl = self.boundary_ttl_frames if near_edge else self.track_ttl_frames
            if frame_id - st.last_frame_id > ttl:
                expired.append(pid)
        for pid in expired:
            del self._stored[pid]

    # ---------- 主更新 ----------

    def update(
        self,
        frame_id: int,
        frame_bgr: cv2.Mat,
        tracked: list[Any],
        uniform_results: dict[int, UniformResult],
    ) -> list[PersonObject]:
        frame_h, frame_w = frame_bgr.shape[:2]
        self._prune_stored(frame_id, frame_w, frame_h)

        new_person_ids: set[int] = set()
        objects: list[PersonObject] = []

        for obj in tracked:
            bt_id = getattr(obj, "bt_key", None) or f"t{obj.track_id}"
            box = obj.xyxy
            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

            person_id = self._bt_to_person.get(bt_id)
            restored = False
            if person_id is None:
                person_id = self._try_restore(
                    frame_bgr, center, box, frame_id, frame_w, frame_h
                )
                if person_id is not None:
                    restored = True
                    self._bt_to_person[bt_id] = person_id
                else:
                    cand = self._pending.get(bt_id)
                    if cand is None:
                        cand = _PendingCandidate(
                            bt_id=bt_id, first_frame=frame_id, last_frame=frame_id,
                            last_center=center,
                        )
                        self._pending[bt_id] = cand
                        if cand.count >= self.person_confirm_frames:
                            person_id = self._next_person_id
                            self._next_person_id += 1
                            self._persons[person_id] = _PersonState(person_id=person_id)
                            self._bt_to_person[bt_id] = person_id
                            self.stats["new_persons"] += 1
                            del self._pending[bt_id]
                    else:
                        cand.last_frame = frame_id
                        cand.last_center = center
                        cand.count += 1
                        if cand.count >= self.person_confirm_frames:
                            person_id = self._next_person_id
                            self._next_person_id += 1
                            self._persons[person_id] = _PersonState(person_id=person_id)
                            self._bt_to_person[bt_id] = person_id
                            self.stats["new_persons"] += 1
                            del self._pending[bt_id]
                    if person_id is None:
                        continue  # 确认期内不输出

            state = self._persons[person_id]
            state.update_motion(center, box, frame_id)
            new_person_ids.add(person_id)

            raw = uniform_results.get(bt_id)
            if raw is None:
                raw = uniform_results.get(getattr(obj, "track_id", None))
            if raw is not None:
                self._update_uniform(state, raw.is_uniform, raw)

            objects.append(
                PersonObject(
                    person_id=person_id,
                    bt_id=bt_id,
                    xyxy=box,
                    confidence=obj.confidence,
                    uniform=state.uniform,
                    uniform_state=state.uniform_state,
                    uniform_prob=state.uniform_prob,
                    restored=restored,
                    restored_from=person_id if restored else None,
                    red_ratio_connected=raw.red_ratio_connected if raw is not None else 0.0,
                    baseline_ratio=raw.baseline_ratio if raw is not None else 0.0,
                )
            )

        for pid in list(self._persons.keys()):
            if pid not in new_person_ids:
                state = self._persons[pid]
                self._store_or_release(
                    frame_bgr, pid, state, frame_id, frame_w, frame_h
                )
                del self._persons[pid]

        active_bt = {getattr(o, "bt_key", None) or f"t{o.track_id}" for o in tracked}
        self._bt_to_person = {
            bt: pid for bt, pid in self._bt_to_person.items() if bt in active_bt
        }
        expired_pending = [
            bt for bt, c in self._pending.items()
            if frame_id - c.last_frame > max(5, self.person_confirm_frames * 2)
        ]
        for bt in expired_pending:
            del self._pending[bt]

        return objects
