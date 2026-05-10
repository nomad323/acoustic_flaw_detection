from __future__ import annotations


def generate_filename_matrix(
    rows: int, cols: int, suffix: str = ".mat", pad: int | None = None
) -> list[list[str]]:
    """Generate a filename matrix like [['00.mat', ...], ...] with any size."""
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive integers")

    width = pad if pad is not None else max(len(str(rows - 1)), len(str(cols - 1)), 1)
    if width <= 0:
        raise ValueError("pad must be a positive integer")

    return [
        [f"{r:0{width}d}{c:0{width}d}{suffix}" for c in range(cols)]
        for r in range(rows)
    ]
