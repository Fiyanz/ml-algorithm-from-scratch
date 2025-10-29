import numpy as np

def calculate_distance(x1, y1, x2, y2) -> float:
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

result = calculate_distance(32, 4, 25, 3)
print(result)