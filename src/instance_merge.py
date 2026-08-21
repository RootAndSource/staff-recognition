"""v0.6 实例归并模块：一人多实例 -> 归并到同一 person（parts 输出）。

依据 Phase 0 标定（docs/research/notes/phase0-instance-merging-calibration.md）：
- 空间先验只能筛候选（x_align >= 0.5、gap 合理、上下堆叠），不能单独判合并；
- 时间证据在静态会议场景判别力弱，仅作辅助；
- 外观判定必须用"规范模板"（两段分别与该人全身/累积模板比对），
  不能两段彼此比对（Phase 0 实测同人两段 HSV 相关性 ~0.015）。

工程实现（第一版）：
- embedding：现有检测器的倒数第二层特征（ultralytics ReID .pt 路径），
  后续可换 OSNet / yolo26s-reid.onnx；
- 模板：person 实例"完整度"（掩码面积 + bbox 高度）达标时提取并 EMA 更新；
- 归并：同帧空间堆叠候选对中，从段与主段模板余弦相似度 >= 阈值 -> 合并。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .identity import PersonObject


def _spatial_candidate(a: np.ndarray, b: np.ndarray) -> bool:
    """空间先验：上下堆叠、x 中心对齐、gap 合理（复用 Phase 0 口径）。"""
    x1a, y1a, x2a, y2a = a
    x1b, y1b, x2b, y2b = b
    wa, ha = x2a - x1a, y2a - y1a
    wb, hb = x2b - x1b, y2b - y1b
    cxa = (x1a + x2a) / 2
    cxb = (x1b + x2b) / 2
    if y2a <= y1b:
        gap = y1b - y2a
    elif y2b <= y1a:
        gap = y1a - y2b
    else:
        return False
    if gap < -0.2 * min(ha, hb) or gap > 0.8 * max(ha, hb):
        return False
    x_align = 1.0 - min(1.0, abs(cxa - cxb) / (0.5 * (wa + wb)))
    return x_align >= 0.5


class InstanceMerger:
    """规范模板 + 空间先验的实例归并器。"""

    def __init__(
        self,
        encoder: Any,
        *,
        similarity_threshold: float = 0.55,
        template_min_area_ratio: float = 0.04,
        template_min_height_ratio: float = 0.30,
        template_ema: float = 0.7,
    ) -> None:
        self.encoder = encoder
        self.similarity_threshold = similarity_threshold
        self.template_min_area_ratio = template_min_area_ratio
        self.template_min_height_ratio = template_min_height_ratio
        self.template_ema = template_ema
        self.templates: dict[int, np.ndarray] = {}

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = a.reshape(-1).astype(np.float64)
        b = b.reshape(-1).astype(np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _extract_embeddings(self, frame_bgr, persons: list[PersonObject]) -> list[np.ndarray | None]:
        """用 ultralytics ReID encoder 提取每个实例的 embedding（mask 区域优先）。"""
        if self.encoder is None or not persons:
            return [None] * len(persons)
        dets = []
        for p in persons:
            box = p.xyxy
            dets.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])
        dets = np.asarray(dets, dtype=float)
        return self.encoder(frame_bgr, dets)

    def process(
        self,
        frame_bgr,
        persons: list[PersonObject],
        frame_id: int,
    ) -> list[PersonObject]:
        """归并同帧分裂实例；更新规范模板。返回合并后的 persons。"""
        if len(persons) < 2:
            # 单人/无人：仅尝试更新模板
            if persons:
                self._update_templates(frame_bgr, persons)
            return persons

        embs = self._extract_embeddings(frame_bgr, persons)
        h, w = frame_bgr.shape[:2]

        # 1) 更新模板（完整度高的实例）
        self._update_templates(frame_bgr, persons, embs, h, w)

        # 2) 空间候选对 -> 模板比对 -> 归并
        merged_out: list[PersonObject] = []
        absorbed: set[int] = set()
        for i in range(len(persons)):
            if i in absorbed:
                continue
            main = persons[i]
            for j in range(i + 1, len(persons)):
                if j in absorbed:
                    continue
                cand = persons[j]
                if not _spatial_candidate(main.xyxy, cand.xyxy):
                    continue
                ei, ej = embs[i], embs[j]
                ti = self.templates.get(main.person_id)
                tj = self.templates.get(cand.person_id)
                score = 0.0
                if ti is not None and ej is not None:
                    score = max(score, self._cosine(ti, ej))
                if tj is not None and ei is not None:
                    score = max(score, self._cosine(tj, ei))
                if score >= self.similarity_threshold:
                    # 归并：cand 并入 main
                    self._absorb(main, cand)
                    absorbed.add(j)
                    main.split_relation = True
                    # 模板合并
                    if ti is not None and ej is not None:
                        self.templates[main.person_id] = (
                            self.template_ema * ti + (1 - self.template_ema) * ej
                        )
            merged_out.append(main)
        return merged_out

    def _absorb(self, main: PersonObject, cand: PersonObject) -> None:
        """把 cand 的实例信息并入 main.parts。"""
        part = {
            "bt_id": cand.bt_id,
            "bbox": [float(v) for v in cand.xyxy],
            "confidence": round(float(cand.confidence), 4),
        }
        if cand.polygon_norm is not None:
            part["mask_polygon"] = [
                [round(float(x), 6), round(float(y), 6)]
                for x, y in cand.polygon_norm
            ]
        if main.parts is None:
            main.parts = []
        main.parts.append(part)
        # 主实例本身也作为 part 之一（保持实例完整列表）
        if not any(p.get("bt_id") == main.bt_id for p in main.parts):
            main.parts.insert(
                0,
                {
                    "bt_id": main.bt_id,
                    "bbox": [float(v) for v in main.xyxy],
                    "confidence": round(float(main.confidence), 4),
                },
            )

    def _update_templates(self, frame_bgr, persons, embs=None, h=0, w=0) -> None:
        """完整度达标的实例 -> 提取 embedding 并 EMA 更新模板。"""
        if self.encoder is None:
            return
        if embs is None:
            embs = self._extract_embeddings(frame_bgr, persons)
        for p, emb in zip(persons, embs):
            if emb is None:
                continue
            box = p.xyxy
            bh = box[3] - box[1]
            area_ratio = (
                float(p.mask.sum() / (h * w))
                if p.mask is not None and h * w > 0
                else 0.0
            )
            if bh < self.template_min_height_ratio * h and area_ratio < self.template_min_area_ratio:
                continue
            if p.person_id in self.templates:
                self.templates[p.person_id] = (
                    self.template_ema * self.templates[p.person_id]
                    + (1 - self.template_ema) * emb
                )
            else:
                self.templates[p.person_id] = emb

    def reset(self) -> None:
        self.templates.clear()
