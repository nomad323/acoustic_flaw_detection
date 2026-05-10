# Acoustic Flaw Detection

超声探伤成像与缺陷识别项目（Python 实现）。

## 1. 项目目标

- 从 A 扫数据重建成像图（`A_norm` 风格归一化图像）
- 在流程内直接进行缺陷识别并输出参数
- 支持两种识别版本：
  - 无机器学习（规则法）
  - 含机器学习（预训练模型候选 + 统一后处理）
- 支持单图输入识别（离线图片推理）

## 2. 目录与关键文件

- `python/my_image.py`：核心成像重建算法（含 Numba 路径）
- `python/experiment.py`：多文件 A 扫读入实验入口
- `python/experiment_new.py`：单 MAT 读入实验入口
- `python/experiment_common.py`：公共流程函数（预处理、配置构建、shallow mask）
- `python/flaw_detection.py`：缺陷识别核心（规则法 + ML 法 + 后处理）
- `python/flaw_visualize.py`：识别轮廓与尺寸叠加可视化
- `python/flaw_infer_image.py`：单图输入识别入口

## 3. 环境依赖

基础依赖（当前代码已使用）：

- `numpy`
- `scipy`
- `matplotlib`

可选依赖：

- `numba`：加速重建核心循环
- `ultralytics`：启用 ML 版本（SAM/FastSAM）

建议安装示例：

```bash
pip install numpy scipy matplotlib
pip install numba
pip install ultralytics
```

## 4. 快速开始

### 4.1 流程内识别（A扫 -> 重建 -> 识别）

运行：

```bash
python python/experiment.py
python python/experiment_new.py
```

脚本会输出：

- 缺陷参数列表（位置、面积、等效直径、内切圆半径、外接圆半径、bbox、score）
- 轮廓叠加图（`Flaw Detection Overlay`）
- 需要时显示 shallow mask 图

### 4.2 单图识别

```bash
python python/flaw_infer_image.py your_image.png --mode rule --pixel-size-mm 1.0
python python/flaw_infer_image.py your_image.png --mode ml --pixel-size-mm 1.0
```

输出：

- 终端打印 JSON
- 同目录写入 `*_defects.json`
- 同目录写入 `*_overlay.png`

## 5. 识别模式说明

### 5.1 规则法（`mode="rule"`）

流程：

1. 归一化
2. 阈值分割（Otsu 或百分位）
3. 形态学开闭运算与填孔
4. 连通域提取与几何参数计算
5. 抑制冗余候选（大框/边缘框/低分/去重）

优点：无需模型权重，稳定可复现。

### 5.2 机器学习法（`mode="ml"`）

流程：

1. 预训练模型生成候选掩膜（ultralytics SAM/FastSAM）
2. 复用与规则法同一后处理与几何量化

说明：

- 若未安装 `ultralytics` 且 `fallback_to_rule=True`，会自动回退规则法
- 若 `fallback_to_rule=False`，会直接抛出异常

## 6. 输出参数定义（统一口径）

每个缺陷对象字段：

- `centroid_x_mm`, `centroid_y_mm`：缺陷中心坐标
- `area_mm2`：缺陷面积
- `equivalent_diameter_mm`：等效圆直径
- `inscribed_radius_mm`：内切圆半径（当前主报告半径）
- `enclosing_radius_mm`：外接圆半径（参考包络）
- `bbox_x_min_mm`, `bbox_y_min_mm`, `bbox_x_max_mm`, `bbox_y_max_mm`
- `bbox_w_mm`, `bbox_h_mm`
- `contour_xy_mm`：轮廓点
- `score`：候选平均强度评分
- `mode`：识别来源（`rule` / `ml` / `ml_fallback_rule`）

## 7. 关键参数建议

在 `build_detection_settings()` 里优先调这些：

- `min_area_mm2`：最小有效面积
- `max_area_ratio`：过滤“几乎整图大框”
- `reject_border_touch`：过滤贴边伪目标
- `min_score`：过滤弱散射噪声
- `dedupe_iou_threshold` + `dedupe_center_dist_mm`：去重强度

经验：

- 冗余框多：增大 `min_score`、减小 `max_area_ratio`
- 小目标漏检：减小 `min_area_mm2`、降低 `min_score`
- 重复目标多：降低 `dedupe_iou_threshold` 或提高 `dedupe_center_dist_mm`

## 8. Shallow Mask 顺序约定

当前已实现为：

1. 先重建图像
2. 先应用 `shallow_mask`
3. 再进行识别与轮廓叠加

确保识别结果与最终展示图一致。

## 9. 常见问题

### Q1: `ModuleNotFoundError: No module named 'ultralytics'`

- 安装：`pip install ultralytics`
- 或把配置改为 `mode="rule"`
- 或保留 `mode="ml"` 但设置 `fallback_to_rule=True`

### Q2: 识别结果出现大红框、重复框

- 调整：
  - `max_area_ratio`
  - `reject_border_touch`
  - `min_score`
  - `dedupe_iou_threshold`

### Q3: 半径该看哪个？

- 工程上建议优先看 `inscribed_radius_mm`（抗散射）
- `enclosing_radius_mm` 作为保守包络参考

