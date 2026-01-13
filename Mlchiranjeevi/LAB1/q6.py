import numpy as np

gradient = np.array([2, 3, 3])

points = [
    (0, 0, 0),
    (1, 2, 3),
    (-1, 4, 2),
    (5, -3, 1)
]

for p in points:
    print(f"At point x1={p[0]}, x2={p[1]}, x3={p[2]} -> gradient = {gradient}")
