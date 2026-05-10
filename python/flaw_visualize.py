from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from flaw_detection import DefectResult, to_display_image


def draw_defect_overlays(
    image: np.ndarray,
    defects: Sequence[DefectResult],
    pixel_size_mm: float,
    title: str = "Flaw Detection Result",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """在成像图上绘制缺陷轮廓与尺寸标签。"""
    display = to_display_image(image)
    plt.figure()
    plt.imshow(display, origin="lower", aspect="equal", cmap="turbo")
    plt.colorbar()
    plt.title(title)

    for idx, item in enumerate(defects, start=1):
        if item.contour_xy_mm:
            # contour_xy_mm 存的是毫米坐标，绘图前转换回像素坐标。
            contour = np.asarray(item.contour_xy_mm, dtype=np.float64)
            contour_x = contour[:, 0] / pixel_size_mm
            contour_y = contour[:, 1] / pixel_size_mm
            plt.plot(contour_x, contour_y, "r-", linewidth=1.3)
            plt.plot(
                [contour_x[-1], contour_x[0]],
                [contour_y[-1], contour_y[0]],
                "r-",
                linewidth=1.3,
            )

        cx = item.centroid_x_mm / pixel_size_mm
        cy = item.centroid_y_mm / pixel_size_mm
        plt.plot(cx, cy, "wo", markersize=3)
        label = (
            f"#{idx} d_eq={item.equivalent_diameter_mm:.2f}mm "
            f"r={item.inscribed_radius_mm:.2f}mm "
            f"(r_out={item.enclosing_radius_mm:.2f}mm)"
        )
        plt.text(cx + 1.0, cy + 1.0, label, color="white", fontsize=8)

    if save_path:
        # 支持离线保存标注结果，便于报告归档。
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
