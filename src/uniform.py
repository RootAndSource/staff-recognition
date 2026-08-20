"""工装识别：上半身 ROI 裁剪 + HSV 红色判定（红色背心/短袖）。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Detection


@dataclass
class UniformResult:
    """单个目标的工装判定结果。"""

    is_uniform: bool
    red_ratio: float        # 主 ROI 红色像素占比（兼容旧字段，取主段）
    red_ratio_main: float   # 主 ROI（肩胸）红色占比
    red_ratio_lower: float  # 兜底 ROI（腹腰）红色占比
    red_ratio_connected: float  # 主 ROI 内最大红色连通域占比（工装成块先验）
    baseline_ratio: float   # 全检测框红色占比（场景红色基线）
    roi_main: tuple[int, int, int, int]   # ? ROI ????
    roi_lower: tuple[int, int, int, int]  # ?? ROI ????
    uniform_prob: float = 0.5             # ???????????????


@dataclass
class _RoiSpec:
    """相对 ROI 定义（0-1 区间）。"""

    x_begin: float
    y_begin: float
    x_end: float
    y_end: float


@dataclass
class UniformClassifier:
    """基于 HSV 颜色规则的红色工装分类器。

    双段 ROI：主段（肩胸）+ 兜底段（腹腰），任一段红色占比达标即判工装，
    用于缓解斜俯视/桌子遮挡导致躯干落在单一 ROI 之外的问题。
    """

    roi_main: _RoiSpec = _RoiSpec(0.10, 0.15, 0.90, 0.55)
    roi_lower: _RoiSpec = _RoiSpec(0.10, 0.45, 0.90, 0.85)
    hsv_ranges: tuple[tuple[int, int], tuple[int, int]] = ((0, 10), (170, 180))
    s_min: int = 70
    v_min: int = 70
    red_ratio_threshold: float = 0.08
    normalize: bool = True
    baseline_factor: float = 1.5   # ROI ????? >= ?? * factor ????
    connected_min_ratio: float = 0.05  # ???????? ROI ????????????
    classifier_path: str | None = None  # MobileNetV3 ??????M3?????? HSV ???
    classifier_only: bool = True        # True=??????False=???? HSV ???OR?

    @staticmethod
    def _clip_roi(
        frame_h: int, frame_w: int, box: np.ndarray, roi: _RoiSpec
    ) -> tuple[int, int, int, int]:
        """把相对 ROI 换算成检测框内像素坐标并裁剪到画面内。"""
        x1, y1, x2, y2 = box
        bx, by, bw, bh = x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)
        rx1 = int(round(bx + roi.x_begin * bw))
        ry1 = int(round(by + roi.y_begin * bh))
        rx2 = int(round(bx + roi.x_end * bw))
        ry2 = int(round(by + roi.y_end * bh))
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(frame_w, rx2), min(frame_h, ry2)
        return (rx1, ry1, rx2, ry2)

    @staticmethod
    def _preprocess(frame_bgr: cv2.Mat, normalize: bool) -> cv2.Mat:
        """????????LAB CLAHE??normalize=False ??????"""
        if not normalize:
            return frame_bgr
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_plane, a_plane, b_plane = cv2.split(lab)
        l_plane = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_plane)
        lab = cv2.merge((l_plane, a_plane, b_plane))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _red_mask(self, frame_bgr: cv2.Mat, roi_px: tuple[int, int, int, int]) -> np.ndarray:
        """?? ROI ???????frame ??????????"""
        rx1, ry1, rx2, ry2 = roi_px
        if rx2 - rx1 < 3 or ry2 - ry1 < 3:
            return np.zeros((1, 1), dtype=np.uint8)
        roi_img = frame_bgr[ry1:ry2, rx1:rx2]
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # 向量化：避免逐像素 Python 循环（性能关键路径）
        red_mask = (
            ((h >= self.hsv_ranges[0][0]) & (h <= self.hsv_ranges[0][1]))
            | ((h >= self.hsv_ranges[1][0]) & (h <= self.hsv_ranges[1][1]))
        ) & (s >= self.s_min) & (v >= self.v_min)
        return red_mask.astype(np.uint8)

    @staticmethod
    def _largest_component_ratio(red_mask: np.ndarray) -> float:
        """计算最大红色连通域占 ROI 的面积比例（工装成块先验）。"""
        if red_mask.size == 0:
            return 0.0
        num, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
        if num <= 1:
            return 0.0
        areas = stats[1:, cv2.CC_STAT_AREA]
        return float(areas.max()) / red_mask.size

    def _ensure_classifier(self):
        """??????????"""
        if self.classifier_path is None:
            return None
        if getattr(self, "_clf_model", None) is None:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            ckpt = torch.load(self.classifier_path, map_location="cpu")
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self._clf_device = torch.device("cpu")  # ???????????
            self._clf_model = model.to(self._clf_device)
            self._clf_imgsz = int(ckpt.get("imgsz", 128))
            self._clf_transform = transforms.Compose([
                transforms.Resize((self._clf_imgsz, self._clf_imgsz)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        return self._clf_model

    def _classifier_prob(self, frame_bgr: cv2.Mat, roi_px: tuple[int, int, int, int]) -> float:
        """? ROI crop ????????"???"???"""
        model = self._ensure_classifier()
        if model is None:
            return 0.5
        rx1, ry1, rx2, ry2 = roi_px
        if rx2 <= rx1 or ry2 <= ry1:
            return 0.0
        from PIL import Image

        crop = frame_bgr[ry1:ry2, rx1:rx2]
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        import torch

        x = self._clf_transform(img).unsqueeze(0).to(self._clf_device)
        with torch.no_grad():
            prob = float(torch.softmax(model(x), dim=1)[0, 1].item())
        return prob

    def classify_batch(self, frame_bgr: cv2.Mat, dets: list[Detection],
                      rois: list[tuple[int, int, int, int]] | None = None) -> list[UniformResult]:
        """??????????????????????????"""
        frame_bgr = self._preprocess(frame_bgr, self.normalize)
        frame_h, frame_w = frame_bgr.shape[:2]
        # 1) ?? ROI ? HSV ???????cv2 ??
        #    rois ???????v0.3 ??????????????????? ROI
        if rois is not None:
            roi_main_list = [tuple(int(v) for v in r) for r in rois]
            roi_lower_list = roi_main_list
        else:
            roi_main_list = [self._clip_roi(frame_h, frame_w, d.xyxy, self.roi_main) for d in dets]
            roi_lower_list = [self._clip_roi(frame_h, frame_w, d.xyxy, self.roi_lower) for d in dets]
        ratios_main = [float(self._red_mask(frame_bgr, r).mean()) for r in roi_main_list]
        ratios_lower = [float(self._red_mask(frame_bgr, r).mean()) for r in roi_lower_list]
        baselines = []
        for d in dets:
            box_px = (int(d.xyxy[0]), int(d.xyxy[1]), int(d.xyxy[2]), int(d.xyxy[3]))
            box_px = (max(0, box_px[0]), max(0, box_px[1]),
                      min(frame_w, box_px[2]), min(frame_h, box_px[3]))
            baselines.append(float(self._red_mask(frame_bgr, box_px).mean()))
        connecteds = [
            max(self._largest_component_ratio(self._red_mask(frame_bgr, r1)),
                self._largest_component_ratio(self._red_mask(frame_bgr, r2)))
            for r1, r2 in zip(roi_main_list, roi_lower_list)
        ]
        hsv_uniforms = [
            c >= self.connected_min_ratio
            and max(m, lo) >= max(b * self.baseline_factor, self.red_ratio_threshold)
            for c, m, lo, b in zip(connecteds, ratios_main, ratios_lower, baselines)
        ]
        # 2) ???????????
        model = self._ensure_classifier()
        if model is None:
            probs = [0.5] * len(dets)
        else:
            from PIL import Image
            import torch

            crops = []
            for r in roi_main_list:
                rx1, ry1, rx2, ry2 = r
                if rx2 <= rx1 or ry2 <= ry1:
                    crops.append(None)
                    continue
                roi = frame_bgr[ry1:ry2, rx1:rx2]
                crops.append(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)))
            xs = torch.stack([self._clf_transform(c) for c in crops if c is not None])
            with torch.no_grad():
                logits = model(xs)
                probs_all = torch.softmax(logits, dim=1)[:, 1].tolist()
            probs = []
            ci = 0
            for c in crops:
                if c is None:
                    probs.append(0.0)
                else:
                    probs.append(float(probs_all[ci]))
                    ci += 1
        # 3) ????
        results = []
        for i, d in enumerate(dets):
            prob = probs[i]
            if model is not None:
                clf_uniform = prob >= 0.5
                is_uniform = clf_uniform if self.classifier_only else (clf_uniform or hsv_uniforms[i])
            else:
                is_uniform = hsv_uniforms[i]
            results.append(UniformResult(
                is_uniform=is_uniform,
                red_ratio=ratios_main[i],
                red_ratio_main=ratios_main[i],
                red_ratio_lower=ratios_lower[i],
                red_ratio_connected=connecteds[i],
                baseline_ratio=baselines[i],
                roi_main=roi_main_list[i],
                roi_lower=roi_lower_list[i],
                uniform_prob=round(prob, 4),
            ))
        return results

    def classify(self, frame_bgr: cv2.Mat, det: Detection) -> UniformResult:
        """对检测框裁剪主/兜底双 ROI，任一红色占比达标即判工装。"""
        frame_bgr = self._preprocess(frame_bgr, self.normalize)
        frame_h, frame_w = frame_bgr.shape[:2]
        roi_main = self._clip_roi(frame_h, frame_w, det.xyxy, self.roi_main)
        roi_lower = self._clip_roi(frame_h, frame_w, det.xyxy, self.roi_lower)
        ratio_main = float(self._red_mask(frame_bgr, roi_main).mean())
        ratio_lower = float(self._red_mask(frame_bgr, roi_lower).mean())
        # 场景红色基线：整个检测框内的红色占比
        box_px = (int(det.xyxy[0]), int(det.xyxy[1]), int(det.xyxy[2]), int(det.xyxy[3]))
        box_px = (
            max(0, box_px[0]), max(0, box_px[1]),
            min(frame_w, box_px[2]), min(frame_h, box_px[3]),
        )
        baseline = float(self._red_mask(frame_bgr, box_px).mean())
        connected_main = self._largest_component_ratio(self._red_mask(frame_bgr, roi_main))
        connected_lower = self._largest_component_ratio(self._red_mask(frame_bgr, roi_lower))
        connected = max(connected_main, connected_lower)
        hsv_uniform = (
            connected >= self.connected_min_ratio
            and max(ratio_main, ratio_lower) >= max(baseline * self.baseline_factor, self.red_ratio_threshold)
        )
        prob = self._classifier_prob(frame_bgr, roi_main)
        if self._ensure_classifier() is not None:
            clf_uniform = prob >= 0.5
            is_uniform = clf_uniform if self.classifier_only else (clf_uniform or hsv_uniform)
        else:
            is_uniform = hsv_uniform
        return UniformResult(
            is_uniform=is_uniform,
            red_ratio=ratio_main,
            red_ratio_main=ratio_main,
            red_ratio_lower=ratio_lower,
            red_ratio_connected=connected,
            baseline_ratio=baseline,
            roi_main=roi_main,
            roi_lower=roi_lower,
            uniform_prob=round(prob, 4),
        )