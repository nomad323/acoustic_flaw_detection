from __future__ import annotations

from time import perf_counter
from os import path
from my_image import my_image
from my_read import my_read


def main() -> None:
    dir = path.dirname(path.abspath(__file__))
    parent_dir = path.dirname(dir)
    data_dir = path.join(parent_dir, "test_data")
    t_start = perf_counter()
    names = [
        ["00.mat", "01.mat", "02.mat", "03.mat"],
        ["10.mat", "11.mat", "12.mat", "13.mat"],
        ["20.mat", "21.mat", "22.mat", "23.mat"],
        ["30.mat", "31.mat", "32.mat", "33.mat"],
    ]
    files = [[path.join(data_dir, f) for f in row] for row in names]
    # Multi-file read logic is unified in my_read().
    data = my_read(files, t=400000, n=4)

    l0 = 0.002
    c = 6270
    t0 = 5e-10
    a = 5e-3
    lam = 1.254e-3
    d = 1e-2
    x1 = d / 2
    x2 = 0.0448 + d / 2
    n = 4
    my_image(x1, x2, a, lam, c, t0, data, n, l0, beam_model="mgb")
    print(f"[experiment2] total runtime: {perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
