"""交互标定：打开截图，用 HSV 与红色占比滑块调参，实时看效果。"""

from __future__ import annotations

import cv2
import numpy as np

from src.detector import Detection
from src.uniform import UniformClassifier


def run_calibrate(image_path: str, config: dict) -> None:
    """交互标定入口：显示原图 + 上半身 ROI 内的红色掩码。"""
    frame = cv2.imread(image_path)
    if frame is None:
        raise IOError(f"无法读取图片: {image_path}")

    ucfg = config["uniform"]
    clf = UniformClassifier(
        roi=(ucfg["roi"]["x_begin"], ucfg["roi"]["y_begin"], ucfg["roi"]["x_end"], ucfg["roi"]["y_end"]),
        hsv_ranges=((ucfg["hsv_red"]["h1_begin"], ucfg["hsv_red"]["h1_end"]),
                    (ucfg["hsv_red"]["h2_begin"], ucfg["hsv_red"]["h2_end"])),
        s_min=ucfg["hsv_red"]["s_min"],
        v_min=ucfg["hsv_red"]["v_min"],
        red_ratio_threshold=ucfg["red_ratio_threshold"],
        normalize=ucfg.get("normalize", True),
    )

    # 用整图作为“检测框”演示：红色占比显示在标题栏，拖滑块看效果
    cv2.namedWindow("calibrate")

    def on_s(min_val: int) -> None:
        clf.s_min = min_val
        _render()

    def on_v(min_val: int) -> None:
        clf.v_min = min_val
        _render()

    def on_ratio(min_val: int) -> None:
        clf.red_ratio_threshold = min_val / 100.0
        _render()

    cv2.createTrackbar("S_min", "calibrate", clf.s_min, 255, on_s)
    cv2.createTrackbar("V_min", "calibrate", clf.v_min, 255, on_v)
    cv2.createTrackbar("ratio%", "calibrate", int(clf.red_ratio_threshold * 100), 50, on_ratio)

    def _render() -> None:
        det = Detection(xyxy=np.array([0, 0, frame.shape[1], frame.shape[0]]), confidence=1.0)
        res = clf.classify(frame, det)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask = (
            ((h >= clf.hsv_ranges[0][0]) & (h <= clf.hsv_ranges[0][1]))
            | ((h >= clf.hsv_ranges[1][0]) & (h <= clf.hsv_ranges[1][1]))
        ) & (s >= clf.s_min) & (v >= clf.v_min)
        mask_img = cv2.merge([mask.astype(np.uint8) * 255] * 3)
        rx1, ry1, rx2, ry2 = res.roi
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        label = f"red_ratio={res.red_ratio:.3f} uniform={res.is_uniform}"
        cv2.putText(frame, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        combined = np.hstack([frame, mask_img])
        cv2.imshow("calibrate", combined)

    _render()
    print("拖动滑块调参，Esc 退出；把最终参数写入 config.yaml 的 uniform 段。")
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
    cv2.destroyAllWindows()
