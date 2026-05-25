from __future__ import annotations

from os import path
from time import perf_counter
from typing import Any

from filename_matrix import generate_filename_matrix
from experiment_common import (
    apply_shallow_mask,
    build_imaging_config,
    maybe_preprocess_data,
    reconstruct_image,
    save_raw_reconstruction_image,
    save_shallow_masked_image,
    show_shallow_masked_image,
)
from flaw_detection import MlDetectionConfig, RuleDetectionConfig, defects_to_dicts, detect_flaws
from flaw_visualize import draw_defect_overlays
from my_read import my_read


def build_read_settings() -> dict[str, Any]:
    """读取多文件矩阵数据所需的参数。"""
    return {
        "data_dir": r"/mnt/d/tanshang/sbts_new",  # 多通道 MAT 文件所在目录
        "n": 8,  # 阵列每边阵元数（文件矩阵尺寸 n×n）
        "t": 2000,  # 每条 A 扫最多读取的样本点
        "variable_name": "voltage",  # MAT 文件内信号变量名
        "max_samples": 10000,  # 单通道原始读取上限（与 t 共同约束）
    }


def load_data(settings: dict[str, Any]):
    """按 n×n 文件名矩阵读取 A 扫数据，返回形状 (n, n, T)。"""
    n = settings["n"]  # 阵列规模
    names = generate_filename_matrix(n, n, suffix=".mat")  # 文件名矩阵（如 00.mat ~ 77.mat）
    files = [[path.join(settings["data_dir"], f) for f in row] for row in names]  # 绝对路径矩阵
    return my_read(
        files=files,
        t=settings["t"],
        n=n,
        variable_name=settings["variable_name"],
        max_samples=settings["max_samples"],
    )


def build_geometry_settings(n: int) -> dict[str, float]:
    """构建与阵列几何和声学常量相关的参数。"""
    l0 = 1e-3  # 相邻阵元中心间距 (m)
    c = 6451.9  # 声速 (m/s)
    t0 = 1e-8  # 采样时间间隔 (s)
    detector_diameter = 6e-3  # 阵元有效直径 (m)
    f0 = 5e6  # 中心频率 (Hz)
    d = 0.01584  # 发射阵列与接收阵列间隙 (m)
    x2 = 0.018  # 接收阵列起始参考位置 (m)
    return {
        "l0": l0,  # 阵元间距 (m)
        "c": c,  # 声速 (m/s)
        "t0": t0,  # 采样时间间隔 (s)
        "fs": 1.0 / t0,  # 采样率 (Hz)
        "a": detector_diameter / 2.0,  # 阵元半径 (m)
        "lam": c / f0,  # 波长 (m)
        "x2": x2,  # 接收阵列参考位置 (m)
        "x1": x2 + d + (n - 1) * l0,  # 发射阵列参考位置 (m)
        "detector_diameter": detector_diameter,  # 阵元直径 (m)
    }


def build_preprocess_settings() -> tuple[bool, dict[str, Any]]:
    """返回预处理开关及参数。"""
    return True, {
        "center_freq_hz": 5e6,  # 带通中心频率 (Hz)
        "bandwidth_hz": 2e6,  # 带通带宽 (Hz)
        "remove_dc": True,  # 是否去直流分量
        "mute_front_samples": 140,  # 前部静音样本数
        "subtract_common_mode": True,  # 是否减公共模式
        "common_mode_method": "median",  # 公共模式估计方式
    }


def build_imaging_settings(detector_diameter: float, use_shallow_mask: bool) -> dict[str, Any]:
    """构建重建与显示相关的配置参数。"""
    return {
        "delta": 1e-4,  # 成像网格步长 (m)
        "length": 100e-3,  # 成像区域长度 (m)
        "width": 60e-3,  # 成像区域宽度 (m)
        "subset_len": 600,  # 每条 A 扫参与重建样本数
        "show": not use_shallow_mask,  # 是否直接显示未遮罩图
        "log_progress": True,  # 是否打印重建进度
        "progress_every": 50,  # 每隔多少行打印一次进度
        "apply_filter": False,  # 是否在成像内部做带通滤波
        "use_numba": True,  # 是否启用 Numba 加速
        "use_fan_mask": True,  # 是否启用扇形掩膜
        "fan_half_angle_deg": 40.0,  # 扇形半角 (deg)
        "fan_origin_x": None,  # 扇形原点 x（None 为自动）
        "fan_origin_y": 0.0,  # 扇形原点 y (m)
        "beam_model": "none",  # 波束模型
        "aperture_apodization": "none",  # 孔径加窗方式
        "piston_diameter": detector_diameter,  # 活塞模型阵元直径 (m)
        "attenuation_db_per_m": 5,  # 衰减系数 (dB/m)(5-20 for Al, 10-50 for 碳钢, 30-100 for 不锈钢)
        "piston_use_abs": True,  # 活塞方向图是否取绝对值
        "piston_min_gain": 0,  # 活塞增益下限
        "legacy_use_abs": True,  # 旧模型增益是否取绝对值
        "legacy_min_gain": 0,  # 旧模型增益下限
        "mgb_angle_c": 3,  # MGB 角度修正系数
        "coherence_mode": "cf_hilbert",  # 相干加权模式->这个非常有用
        "coherence_gamma": 2,  # 相干加权指数
        "reduction_mode": " gated_max",  # 轨迹聚合模式->别用max,这个才正常
        "gate_center_idx": 15,  # 门控中心索引
        "gate_half_width": 15,  # 门控半宽
        "sensitivity_comp": True,  # 是否做灵敏度补偿
    }


def build_detection_settings() -> dict[str, Any]:
    """构建缺陷识别配置参数。"""
    return {
        "mode": "ml",  # 识别模式：rule | ml
        "rule_cfg": RuleDetectionConfig(
            smooth_sigma=1.0,
            threshold_strategy="otsu",
            threshold_percentile=99.0,
            min_area_mm2=1.0,
            morph_open_px=1,
            morph_close_px=2,
            fill_holes=True,
            ignore_top_mm=6.0,
            max_results=20,
            max_area_ratio=0.25,
            reject_border_touch=True,
            border_margin_px=1,
            min_score=0.15,
            dedupe_iou_threshold=0.25,
            dedupe_center_dist_mm=1.0,
        ),
        "ml_cfg": MlDetectionConfig(
            backend="ultralytics_sam",
            model_name="sam_b.pt",
            conf_threshold=0.25,
            iou_threshold=0.7,
            min_area_mm2=0.5,
            ignore_top_mm=6.0,
            fallback_to_rule=True,
            max_results=20,
            max_area_ratio=0.20,
            reject_border_touch=True,
            border_margin_px=1,
            min_score=0.1,
            dedupe_iou_threshold=0.25,
            dedupe_center_dist_mm=1.0,
        ),
    }


def print_defect_summary(defects: list[dict[str, Any]], prefix: str, total_depth_mm: float | None = None) -> None:
    print(f"[{prefix}] defects found: {len(defects)}")
    for idx, item in enumerate(defects, start=1):
        cy = item['centroid_y_mm']
        ymin = item['bbox_y_min_mm']
        ymax = item['bbox_y_max_mm']
        if total_depth_mm is not None:
            cy = total_depth_mm - cy
            ymin_new = total_depth_mm - ymax
            ymax_new = total_depth_mm - ymin
            ymin, ymax = ymin_new, ymax_new
        print(
            f"[{prefix}] #{idx}: center=({item['centroid_x_mm']:.2f}mm,{cy:.2f}mm), "
            f"area={item['area_mm2']:.2f}mm2, d_eq={item['equivalent_diameter_mm']:.2f}mm, "
            f"r={item['inscribed_radius_mm']:.2f}mm, r_out={item['enclosing_radius_mm']:.2f}mm, "
            f"bbox=({item['bbox_w_mm']:.2f}mm,{item['bbox_h_mm']:.2f}mm), "
            f"score={item['score']:.3f}, mode={item['mode']}"
        )


def main() -> None:
    t_start = perf_counter()  # 程序起始时间戳

    # 1) 读取原始数据
    read_settings = build_read_settings()  # 数据读取配置
    data = load_data(read_settings)  # 原始数据立方体 (n, n, T)

    # 2) 几何参数与信号预处理
    n = read_settings["n"]  # 阵列规模
    geometry = build_geometry_settings(n)  # 几何与物理参数字典
    use_preprocess, preprocess_settings = build_preprocess_settings()  # 预处理开关与参数
    data = maybe_preprocess_data(
        data=data,
        fs=geometry["fs"],
        use_preprocess=use_preprocess,
        preprocess_kwargs=preprocess_settings,
    )

    # 3) 成像配置、重建与可视化
    use_shallow_mask = True  # 是否在显示时遮蔽浅层区域
    shallow_mask_mm = 6.0  # 浅层遮蔽厚度 (mm)
    imaging_settings = build_imaging_settings(  # 成像重建配置
        detector_diameter=geometry["detector_diameter"],
        use_shallow_mask=use_shallow_mask,
    )
    config = build_imaging_config(  # 强类型成像配置对象
        x1=geometry["x1"],
        x2=geometry["x2"],
        a=geometry["a"],
        lam=geometry["lam"],
        c=geometry["c"],
        t0=geometry["t0"],
        n=n,
        l0=geometry["l0"],
        imaging_kwargs=imaging_settings,
    )

    image = reconstruct_image(data, config)  # 归一化重建图像
    final_image = image  # 最终用于识别与展示的图像
    if use_shallow_mask:
        final_image = apply_shallow_mask(  # 先做浅层遮罩，再进入后续识别流程
            image=image,
            delta=config.delta,
            shallow_mask_mm=shallow_mask_mm,
        )

    pixel_size_mm = config.delta * 1e3  # 像素尺寸 (mm/px)
    detection_settings = build_detection_settings()  # 识别配置
    defects = detect_flaws(
        image=final_image,
        pixel_size_mm=pixel_size_mm,
        mode=detection_settings["mode"],
        rule_cfg=detection_settings["rule_cfg"],
        ml_cfg=detection_settings["ml_cfg"],
    )
    defect_dicts = defects_to_dicts(defects)  # 识别结果（可序列化）
    total_depth_mm = image.shape[1] * pixel_size_mm
    print_defect_summary(defect_dicts, prefix="experiment", total_depth_mm=total_depth_mm)
    draw_defect_overlays(
        image=final_image,
        defects=defects,
        pixel_size_mm=pixel_size_mm,
        title=f"Flaw Detection Overlay ({detection_settings['mode']})",
        show=True,
    )

    if use_shallow_mask:
        show_shallow_masked_image(
            image=image,
            delta=config.delta,
            shallow_mask_mm=shallow_mask_mm,
            title="Normalized Imaging Result (Shallow Masked)",
        )

    # --- saveplot: 将图像保存到原始数据目录 ---
    out_dir = read_settings["data_dir"]
    save_raw_reconstruction_image(
        image=image,
        title="Normalized Imaging Result",
        save_path=path.join(out_dir, "experiment_image.png"),
    )
    draw_defect_overlays(
        image=final_image,
        defects=defects,
        pixel_size_mm=pixel_size_mm,
        title=f"Flaw Detection Overlay ({detection_settings['mode']})",
        save_path=path.join(out_dir, "experiment_overlay.png"),
        show=False,
    )
    if use_shallow_mask:
        save_shallow_masked_image(
            image=image,
            delta=config.delta,
            shallow_mask_mm=shallow_mask_mm,
            title="Normalized Imaging Result (Shallow Masked)",
            save_path=path.join(out_dir, "experiment_shallow_masked.png"),
        )
    # -------------------------------------------------

    print(f"[experiment] total runtime: {perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
