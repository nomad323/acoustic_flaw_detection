from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flaw_detection import MlDetectionConfig, RuleDetectionConfig, defects_to_dicts, detect_flaws
from flaw_visualize import draw_defect_overlays


def _load_image_gray(image_path: Path) -> np.ndarray:
    """读取并标准化输入图像，转换为内部重建图像坐标。"""
    arr = plt.imread(str(image_path))
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., :3]
        arr = np.mean(arr, axis=2)
    if arr.ndim != 2:
        raise ValueError("Input image must be 2D grayscale or RGB.")
    low = float(np.min(arr))
    high = float(np.max(arr))
    if high > low:
        arr = (arr - low) / (high - low)
    else:
        arr = np.zeros_like(arr)
    # `detect_flaws()` expects reconstruction orientation and will do flipud(T).
    # Recover that orientation from display image for compatibility.
    return np.flipud(arr).T


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Infer flaw parameters from a single image.")
    parser.add_argument("image_path", type=Path, help="Input image path.")
    parser.add_argument("--mode", choices=["rule", "ml"], default="rule", help="Detection mode.")
    parser.add_argument(
        "--pixel-size-mm",
        type=float,
        default=1.0,
        help="Pixel size in mm/px for physical-size conversion.",
    )
    parser.add_argument(
        "--ignore-top-mm",
        type=float,
        default=0.0,
        help="Ignore top shallow layer in display image (mm).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--overlay-path",
        type=Path,
        default=None,
        help="Optional overlay image output path.",
    )
    return parser.parse_args()


def main() -> None:
    """单图推理入口：输出 JSON 并保存叠加图。"""
    args = parse_args()
    image = _load_image_gray(args.image_path)
    pixel_size_mm = float(args.pixel_size_mm)

    rule_cfg = RuleDetectionConfig(ignore_top_mm=float(args.ignore_top_mm))
    ml_cfg = MlDetectionConfig(ignore_top_mm=float(args.ignore_top_mm))
    defects = detect_flaws(
        image=image,
        pixel_size_mm=pixel_size_mm,
        mode=args.mode,
        rule_cfg=rule_cfg,
        ml_cfg=ml_cfg,
    )
    defect_dicts = defects_to_dicts(defects)

    json_text = json.dumps(defect_dicts, ensure_ascii=False, indent=2)
    print(json_text)

    output_json = args.output_json
    if output_json is None:
        # 默认与输入图同目录，自动生成结果文件名。
        output_json = args.image_path.with_name(f"{args.image_path.stem}_defects.json")
    output_json.write_text(json_text, encoding="utf-8")

    overlay_path = args.overlay_path
    if overlay_path is None:
        # 默认与输入图同目录保存叠加可视化。
        overlay_path = args.image_path.with_name(f"{args.image_path.stem}_overlay.png")
    draw_defect_overlays(
        image=image,
        defects=defects,
        pixel_size_mm=pixel_size_mm,
        title=f"Flaw Detection Overlay ({args.mode})",
        save_path=str(overlay_path),
        show=True,
    )


if __name__ == "__main__":
    main()
