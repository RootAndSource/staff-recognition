"""M5 实例分割引擎：检测器出框 + SAM2 box-prompt 生成人体实例掩码。

路线说明（2026-08-21 调整）：放弃 YOLO11-seg 微调，直接用最强的现成
分割模型（SAM2.1 large）作为分割底座（导师 v0.5 的"检测器 + SAM2"备选
流程转主线）。掩码语义为"可见部分"，派生框仅用于跟踪与输出。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM, YOLO


@dataclass
class SegmentInstance:
    """单个人体实例：掩码 + 派生框 + 置信度 + 简化多边形。"""

    mask: np.ndarray          # HxW bool 掩码（原图尺寸）
    xyxy: np.ndarray          # 派生框（掩码最小外接矩形）
    confidence: float
    polygon_norm: np.ndarray  # 简化多边形（归一化坐标 Nx2），供 JSONL/渲染


class SegmentationEngine:
    """检测 + SAM2 的实例分割流水线。"""

    def __init__(
        self,
        *,
        det_weights: str,
        det_model=None,
        sam_weights: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.5,
        imgsz: int = 960,
        max_det: int = 40,
        min_mask_area_ratio: float = 0.0001,
        simplify_points: int = 40,
        device: str = "",
    ) -> None:
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.max_det = max_det
        self.min_mask_area_ratio = min_mask_area_ratio
        self.simplify_points = simplify_points
        self.device = device
        self.det_model = None
        if det_model is None:
            self.det = YOLO(det_weights)
        else:
            self.det = det_model
        self.sam = SAM(sam_weights)

    @staticmethod
    def _mask_to_polygon(mask: np.ndarray, max_points: int = 40) -> np.ndarray:
        """掩码轮廓 -> 简化多边形（像素坐标）。"""
        mask_u8 = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((0, 2), dtype=float)
        cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
        if len(cnt) <= max_points:
            return cnt
        closed = np.vstack([cnt, cnt[0]]).astype(np.float32)
        eps = 0.01 * cv2.arcLength(closed, True)
        while True:
            approx = cv2.approxPolyDP(closed, eps, True).reshape(-1, 2)
            if len(approx) <= max_points or eps <= 1e-6:
                break
            eps *= 0.6
        return approx

    def segment_with_boxes(
        self, frame_bgr: cv2.Mat, boxes: np.ndarray, confs: np.ndarray
    ) -> list[SegmentInstance | None]:
        """用外部检测框跑 SAM（与 boxes 一一对应，无效实例为 None）。"""
        h, w = frame_bgr.shape[:2]
        if len(boxes) == 0:
            return []
        sam_res = self.sam(frame_bgr, bboxes=boxes, verbose=False, device=self.device)[0]
        if sam_res.masks is None or len(sam_res.masks) == 0:
            return [None] * len(boxes)
        masks = sam_res.masks.data.cpu().numpy()
        results: list[SegmentInstance | None] = []
        for conf, mask in zip(confs, masks):
            mask_bool = mask > 0.5
            if mask_bool.sum() / (h * w) < self.min_mask_area_ratio:
                results.append(None)
                continue
            ys, xs = np.where(mask_bool)
            if len(xs) == 0:
                results.append(None)
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            if x2 - x1 < 2 or y2 - y1 < 2:
                results.append(None)
                continue
            poly = self._mask_to_polygon(mask_bool, self.simplify_points)
            if len(poly) < 3:
                results.append(None)
                continue
            results.append(
                SegmentInstance(
                    mask=mask_bool,
                    xyxy=np.array([x1, y1, x2, y2], dtype=float),
                    confidence=float(conf),
                    polygon_norm=poly / np.array([w, h], dtype=float),
                )
            )
        return results

    def segment_parts(
        self,
        frame_bgr: cv2.Mat,
        instances: list[SegmentInstance],
        upper_ratio: float = 0.55,
    ) -> list[np.ndarray | None]:
        """M7 部位分割：在实例掩码内切上半身部位。

        用"上半身框"（实例框顶部到 upper_ratio 高度）提示 SAM2 出上半身掩码，
        再与实例掩码取交集，得到工装 ROI 使用的部位掩码（排除桌子/背景）。
        当前只实现 upper_body；head/face/torso/legs 为 B/C/D 预留。
        """
        if not instances:
            return []
        boxes = []
        valid_idx = []
        for i, inst in enumerate(instances):
            x1, y1, x2, y2 = [float(v) for v in inst.xyxy]
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            ub = [x1, y1, x2, y1 + upper_ratio * (y2 - y1)]
            boxes.append(ub)
            valid_idx.append(i)
        if not boxes:
            return [None] * len(instances)
        try:
            sam_res = self.sam(
                frame_bgr, bboxes=np.asarray(boxes, dtype=float),
                verbose=False, device=self.device,
            )[0]
        except Exception:
            return [None] * len(instances)
        parts: list[np.ndarray | None] = [None] * len(instances)
        if sam_res.masks is None or len(sam_res.masks) == 0:
            return parts
        masks = sam_res.masks.data.cpu().numpy()
        for mask, idx in zip(masks, valid_idx):
            upper = mask > 0.5
            inst = instances[idx]
            if inst.mask is None:
                parts[idx] = upper
            else:
                parts[idx] = upper & inst.mask
        return parts

    def segment(self, frame_bgr: cv2.Mat) -> list[SegmentInstance]:
        """对一帧执行实例分割，返回人体实例列表（按置信度降序）。"""
        h, w = frame_bgr.shape[:2]
        det_res = self.det(
            frame_bgr,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            max_det=self.max_det,
            verbose=False,
        )[0]
        if det_res.boxes is None or len(det_res.boxes) == 0:
            return []

        boxes = det_res.boxes.xyxy.cpu().numpy()
        confs = det_res.boxes.conf.cpu().numpy()
        sam_res = self.sam(frame_bgr, bboxes=boxes, verbose=False, device=self.device)[0]
        if sam_res.masks is None or len(sam_res.masks) == 0:
            return []

        masks = sam_res.masks.data.cpu().numpy()  # N x H x W（缩放后尺寸？用 xy 重建更稳）
        instances: list[SegmentInstance] = []
        for conf, mask in zip(confs, masks):
            mask_bool = mask > 0.5
            if mask_bool.sum() / (h * w) < self.min_mask_area_ratio:
                continue
            ys, xs = np.where(mask_bool)
            if len(xs) == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            poly = self._mask_to_polygon(mask_bool, self.simplify_points)
            if len(poly) < 3:
                continue
            instances.append(
                SegmentInstance(
                    mask=mask_bool,
                    xyxy=np.array([x1, y1, x2, y2], dtype=float),
                    confidence=float(conf),
                    polygon_norm=poly / np.array([w, h], dtype=float),
                )
            )
        instances.sort(key=lambda inst: -inst.confidence)
        return instances


def load_segmentation_engine(cfg: dict, det_model=None) -> SegmentationEngine:
    """从 config 构建分割引擎。"""
    scfg = cfg.get("segmentation", {})
    mcfg = cfg.get("model", {})
    return SegmentationEngine(
        det_weights=scfg.get("det_weights") or mcfg.get("full_weights") or mcfg["weights"],
        det_model=det_model,
        sam_weights=scfg.get("sam_weights", "models/sam2.1_l.pt"),
        conf_thres=scfg.get("conf_thres", mcfg.get("conf_thres", 0.25)),
        iou_thres=scfg.get("iou_thres", mcfg.get("iou_thres", 0.5)),
        imgsz=scfg.get("imgsz", mcfg.get("imgsz", 960)),
        max_det=scfg.get("max_det", 40),
        min_mask_area_ratio=scfg.get("min_mask_area_ratio", 0.0001),
        simplify_points=scfg.get("simplify_points", 40),
        device=scfg.get("device", mcfg.get("device", "")),
    )
