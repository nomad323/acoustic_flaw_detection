from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull


@dataclass(slots=True)
class DefectResult:
    """统一缺陷输出结构（单位统一为 mm / mm²）。"""

    centroid_x_mm: float
    centroid_y_mm: float
    area_mm2: float
    equivalent_diameter_mm: float
    inscribed_radius_mm: float
    enclosing_radius_mm: float
    bbox_x_min_mm: float
    bbox_y_min_mm: float
    bbox_x_max_mm: float
    bbox_y_max_mm: float
    bbox_w_mm: float
    bbox_h_mm: float
    contour_xy_mm: list[tuple[float, float]]
    score: float
    mode: str


@dataclass(slots=True)
class RuleDetectionConfig:
    """无机器学习版本（规则法）配置。"""

    smooth_sigma: float = 1.0
    threshold_strategy: str = "otsu"  # otsu | percentile
    threshold_percentile: float = 99.0
    min_area_mm2: float = 1.0
    morph_open_px: int = 1
    morph_close_px: int = 2
    fill_holes: bool = True
    ignore_top_mm: float = 0.0
    max_results: int = 20
    max_area_ratio: float = 0.25
    reject_border_touch: bool = True
    border_margin_px: int = 1
    min_score: float = 0.0
    dedupe_iou_threshold: float = 0.25
    dedupe_center_dist_mm: float = 2.0


@dataclass(slots=True)
class MlDetectionConfig:
    """机器学习版本配置（当前默认走 ultralytics 的 SAM/FastSAM）。"""

    backend: str = "ultralytics_sam"  # ultralytics_sam | ultralytics_fastsam
    model_name: str = "sam_b.pt"
    device: str | None = None
    conf_threshold: float = 0.25
    iou_threshold: float = 0.7
    min_area_mm2: float = 1.0
    ignore_top_mm: float = 0.0
    fallback_to_rule: bool = True
    max_results: int = 20
    max_area_ratio: float = 0.20
    reject_border_touch: bool = True
    border_margin_px: int = 1
    min_score: float = 0.15
    dedupe_iou_threshold: float = 0.25
    dedupe_center_dist_mm: float = 2.0


def to_display_image(image: np.ndarray) -> np.ndarray:
    """将内部图像坐标变换为显示坐标。"""
    return np.flipud(np.asarray(image, dtype=np.float64).T)


def _normalize_01(img: np.ndarray) -> np.ndarray:
    """归一化到 [0, 1] 区间。"""
    low = float(np.min(img))
    high = float(np.max(img))
    if high <= low:
        return np.zeros_like(img, dtype=np.float64)
    return (img - low) / (high - low)


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """纯 numpy Otsu 阈值实现，避免额外依赖。"""
    if values.size == 0:
        return 0.0
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(bins))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    idx = int(np.argmax(sigma_b2))
    return float((edges[idx] + edges[idx + 1]) * 0.5)


def _disk(radius: int) -> np.ndarray:
    """生成圆盘结构元，供形态学开闭运算使用。"""
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _apply_ignore_top(mask: np.ndarray, pixel_size_mm: float, ignore_top_mm: float) -> np.ndarray:
    """按 mm 抑制浅层区域（显示图顶部）。"""
    out = mask.copy()
    rows = int(round(ignore_top_mm / pixel_size_mm)) if pixel_size_mm > 0 else 0
    if rows > 0:
        rows = min(rows, out.shape[0])
        out[:rows, :] = False
    return out


def _boundary_points_xy_px(component: np.ndarray) -> np.ndarray:
    """提取连通域边界点并尽量压缩为凸包点集。"""
    edge = component & (~ndi.binary_erosion(component))
    pts_rc = np.argwhere(edge)
    if pts_rc.shape[0] == 0:
        pts_rc = np.argwhere(component)
    if pts_rc.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    pts_xy = np.column_stack((pts_rc[:, 1], pts_rc[:, 0])).astype(np.float64)
    if pts_xy.shape[0] >= 3:
        try:
            hull = ConvexHull(pts_xy)
            pts_xy = pts_xy[hull.vertices]
        except Exception:
            pass
    return pts_xy


def _ritter_circle_radius_xy_px(points_xy: np.ndarray) -> float:
    """Ritter 近似最小外接圆半径（像素单位）。"""
    if points_xy.shape[0] == 0:
        return 0.0
    p0 = points_xy[0]
    d0 = np.sum((points_xy - p0) ** 2, axis=1)
    p1 = points_xy[int(np.argmax(d0))]
    d1 = np.sum((points_xy - p1) ** 2, axis=1)
    p2 = points_xy[int(np.argmax(d1))]
    center = 0.5 * (p1 + p2)
    radius = math.sqrt(float(np.sum((p2 - center) ** 2)))
    for p in points_xy:
        diff = p - center
        dist = math.sqrt(float(np.sum(diff * diff)))
        if dist > radius:
            radius = 0.5 * (radius + dist)
            if dist > 0:
                center = center + ((dist - radius) / dist) * diff
    return float(radius)


def _results_from_masks(
    masks: list[np.ndarray],
    intensity_01: np.ndarray,
    pixel_size_mm: float,
    min_area_mm2: float,
    ignore_top_mm: float,
    max_results: int,
    max_area_ratio: float,
    reject_border_touch: bool,
    border_margin_px: int,
    min_score: float,
    dedupe_iou_threshold: float,
    dedupe_center_dist_mm: float,
    mode: str,
) -> list[DefectResult]:
    """把候选 mask 转成工程可读缺陷参数，并做抑制/去重。"""
    results: list[DefectResult] = []
    px_area_mm2 = pixel_size_mm * pixel_size_mm
    total_area_px = float(intensity_01.shape[0] * intensity_01.shape[1])
    for mask in masks:
        comp = _apply_ignore_top(mask.astype(bool), pixel_size_mm, ignore_top_mm)
        area_px = int(np.count_nonzero(comp))
        if area_px <= 0:
            continue
        # 过滤"包住整图"的伪目标（常见于超声散射背景）。
        if max_area_ratio > 0 and (area_px / total_area_px) > max_area_ratio:
            continue
        if reject_border_touch:
            ys_b, xs_b = np.nonzero(comp)
            if ys_b.size > 0:
                if (
                    np.min(ys_b) <= border_margin_px
                    or np.min(xs_b) <= border_margin_px
                    or np.max(ys_b) >= (comp.shape[0] - 1 - border_margin_px)
                    or np.max(xs_b) >= (comp.shape[1] - 1 - border_margin_px)
                ):
                    continue
        area_mm2 = area_px * px_area_mm2
        if area_mm2 < min_area_mm2:
            continue

        ys, xs = np.nonzero(comp)
        x_min_px = float(np.min(xs))
        x_max_px = float(np.max(xs))
        y_min_px = float(np.min(ys))
        y_max_px = float(np.max(ys))
        centroid_x_px = float(np.mean(xs))
        centroid_y_px = float(np.mean(ys))
        bbox_w_mm = (x_max_px - x_min_px + 1.0) * pixel_size_mm
        bbox_h_mm = (y_max_px - y_min_px + 1.0) * pixel_size_mm
        equivalent_diameter_mm = math.sqrt((4.0 * area_mm2) / math.pi)
        inscribed_radius_px = float(np.max(ndi.distance_transform_edt(comp))) if area_px > 0 else 0.0

        contour_px = _boundary_points_xy_px(comp)
        enclosing_radius_px = _ritter_circle_radius_xy_px(contour_px)
        contour_xy_mm = [(float(x * pixel_size_mm), float(y * pixel_size_mm)) for x, y in contour_px]
        score = float(np.mean(intensity_01[comp])) if area_px > 0 else 0.0
        # 过滤能量过弱的候选。
        if score < min_score:
            continue

        results.append(
            DefectResult(
                centroid_x_mm=centroid_x_px * pixel_size_mm,
                centroid_y_mm=centroid_y_px * pixel_size_mm,
                area_mm2=area_mm2,
                equivalent_diameter_mm=equivalent_diameter_mm,
                inscribed_radius_mm=inscribed_radius_px * pixel_size_mm,
                enclosing_radius_mm=enclosing_radius_px * pixel_size_mm,
                bbox_x_min_mm=x_min_px * pixel_size_mm,
                bbox_y_min_mm=y_min_px * pixel_size_mm,
                bbox_x_max_mm=x_max_px * pixel_size_mm,
                bbox_y_max_mm=y_max_px * pixel_size_mm,
                bbox_w_mm=bbox_w_mm,
                bbox_h_mm=bbox_h_mm,
                contour_xy_mm=contour_xy_mm,
                score=score,
                mode=mode,
            )
        )

    # 去除高度重叠且中心接近的重复框。
    results = _dedupe_results(results, dedupe_iou_threshold, dedupe_center_dist_mm)
    results.sort(key=lambda r: r.area_mm2, reverse=True)
    return results[:max_results]


def _bbox_iou(a: DefectResult, b: DefectResult) -> float:
    """基于 mm 坐标计算两个 bbox 的 IoU。"""
    x_left = max(a.bbox_x_min_mm, b.bbox_x_min_mm)
    y_top = max(a.bbox_y_min_mm, b.bbox_y_min_mm)
    x_right = min(a.bbox_x_max_mm, b.bbox_x_max_mm)
    y_bottom = min(a.bbox_y_max_mm, b.bbox_y_max_mm)
    iw = max(0.0, x_right - x_left)
    ih = max(0.0, y_bottom - y_top)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.bbox_x_max_mm - a.bbox_x_min_mm) * max(0.0, a.bbox_y_max_mm - a.bbox_y_min_mm)
    area_b = max(0.0, b.bbox_x_max_mm - b.bbox_x_min_mm) * max(0.0, b.bbox_y_max_mm - b.bbox_y_min_mm)
    union = max(area_a + area_b - inter, 1e-12)
    return inter / union


def _center_distance_mm(a: DefectResult, b: DefectResult) -> float:
    """缺陷中心点距离（mm）。"""
    dx = a.centroid_x_mm - b.centroid_x_mm
    dy = a.centroid_y_mm - b.centroid_y_mm
    return math.sqrt(dx * dx + dy * dy)


def _dedupe_results(
    results: list[DefectResult],
    iou_threshold: float,
    center_dist_mm: float,
) -> list[DefectResult]:
    """按 score/面积保留优质候选，抑制重复检测结果。"""
    if not results:
        return results
    # Keep stronger targets first (higher score, then larger area).
    ordered = sorted(results, key=lambda r: (r.score, r.area_mm2), reverse=True)
    kept: list[DefectResult] = []
    for cand in ordered:
        duplicated = False
        for picked in kept:
            if _bbox_iou(cand, picked) >= iou_threshold and _center_distance_mm(cand, picked) <= center_dist_mm:
                duplicated = True
                break
        if not duplicated:
            kept.append(cand)
    return kept


def detect_flaws_rule(
    image: np.ndarray,
    pixel_size_mm: float,
    cfg: RuleDetectionConfig | None = None,
) -> list[DefectResult]:
    """规则法识别：阈值分割 + 形态学 + 连通域几何量化。"""
    cfg = cfg or RuleDetectionConfig()
    display = to_display_image(image)
    norm = _normalize_01(display)
    if cfg.smooth_sigma > 0:
        norm = ndi.gaussian_filter(norm, sigma=cfg.smooth_sigma)

    strategy = cfg.threshold_strategy.lower().strip()
    if strategy == "otsu":
        thr = _otsu_threshold(norm.ravel())
    elif strategy == "percentile":
        thr = float(np.percentile(norm, cfg.threshold_percentile))
    else:
        raise ValueError("threshold_strategy must be 'otsu' or 'percentile'.")
    binary = norm >= thr

    if cfg.morph_open_px > 0:
        binary = ndi.binary_opening(binary, structure=_disk(cfg.morph_open_px))
    if cfg.morph_close_px > 0:
        binary = ndi.binary_closing(binary, structure=_disk(cfg.morph_close_px))
    if cfg.fill_holes:
        binary = ndi.binary_fill_holes(binary)

    labels, num = ndi.label(binary)
    masks: list[np.ndarray] = []
    for label_idx in range(1, num + 1):
        masks.append(labels == label_idx)

    return _results_from_masks(
        masks=masks,
        intensity_01=norm,
        pixel_size_mm=pixel_size_mm,
        min_area_mm2=cfg.min_area_mm2,
        ignore_top_mm=cfg.ignore_top_mm,
        max_results=cfg.max_results,
        max_area_ratio=cfg.max_area_ratio,
        reject_border_touch=cfg.reject_border_touch,
        border_margin_px=cfg.border_margin_px,
        min_score=cfg.min_score,
        dedupe_iou_threshold=cfg.dedupe_iou_threshold,
        dedupe_center_dist_mm=cfg.dedupe_center_dist_mm,
        mode="rule",
    )


def _predict_masks_ultralytics(display_01: np.ndarray, cfg: MlDetectionConfig) -> list[np.ndarray]:
    """调用 ultralytics 预训练模型生成候选掩膜。"""
    rgb = (np.clip(display_01, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = np.repeat(rgb[..., None], 3, axis=2)

    try:
        from ultralytics import FastSAM, SAM  # type: ignore[import-untyped]
    except Exception as exc:
        raise RuntimeError("ultralytics is not available for ML detection.") from exc

    backend = cfg.backend.lower().strip()
    if backend == "ultralytics_sam":
        model = SAM(cfg.model_name)
    elif backend == "ultralytics_fastsam":
        model = FastSAM(cfg.model_name)
    else:
        raise ValueError("backend must be 'ultralytics_sam' or 'ultralytics_fastsam'.")

    predict_kwargs: dict[str, Any] = {"verbose": False}
    if cfg.device is not None:
        predict_kwargs["device"] = cfg.device
    if backend == "ultralytics_fastsam":
        predict_kwargs["conf"] = cfg.conf_threshold
        predict_kwargs["iou"] = cfg.iou_threshold

    outputs = model(rgb, **predict_kwargs)
    if not outputs:
        return []
    result0 = outputs[0]
    if result0.masks is None or result0.masks.data is None:
        return []

    data = result0.masks.data
    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    data = np.asarray(data)
    return [data[i] > 0.5 for i in range(data.shape[0])]


def detect_flaws_ml(
    image: np.ndarray,
    pixel_size_mm: float,
    cfg: MlDetectionConfig | None = None,
) -> list[DefectResult]:
    """机器学习识别：模型给候选，规则后处理统一几何口径。"""
    cfg = cfg or MlDetectionConfig()
    display = to_display_image(image)
    norm = _normalize_01(display)

    try:
        masks = _predict_masks_ultralytics(norm, cfg)
    except Exception:
        # 模型不可用时可自动回退规则法，保证流程不中断。
        if cfg.fallback_to_rule:
            rule_cfg = RuleDetectionConfig(
                min_area_mm2=cfg.min_area_mm2,
                ignore_top_mm=cfg.ignore_top_mm,
                max_results=cfg.max_results,
            )
            fallback = detect_flaws_rule(image, pixel_size_mm, rule_cfg)
            for item in fallback:
                item.mode = "ml_fallback_rule"
            return fallback
        raise

    return _results_from_masks(
        masks=masks,
        intensity_01=norm,
        pixel_size_mm=pixel_size_mm,
        min_area_mm2=cfg.min_area_mm2,
        ignore_top_mm=cfg.ignore_top_mm,
        max_results=cfg.max_results,
        max_area_ratio=cfg.max_area_ratio,
        reject_border_touch=cfg.reject_border_touch,
        border_margin_px=cfg.border_margin_px,
        min_score=cfg.min_score,
        dedupe_iou_threshold=cfg.dedupe_iou_threshold,
        dedupe_center_dist_mm=cfg.dedupe_center_dist_mm,
        mode="ml",
    )


def detect_flaws(
    image: np.ndarray,
    pixel_size_mm: float,
    mode: str = "rule",
    rule_cfg: RuleDetectionConfig | None = None,
    ml_cfg: MlDetectionConfig | None = None,
) -> list[DefectResult]:
    """统一入口：按 mode 切换规则法或机器学习法。"""
    mode_l = mode.lower().strip()
    if mode_l == "rule":
        return detect_flaws_rule(image, pixel_size_mm, rule_cfg)
    if mode_l == "ml":
        return detect_flaws_ml(image, pixel_size_mm, ml_cfg)
    raise ValueError("mode must be 'rule' or 'ml'.")


def defects_to_dicts(defects: list[DefectResult]) -> list[dict[str, Any]]:
    """将 dataclass 结果转为可序列化字典。"""
    return [asdict(item) for item in defects]
