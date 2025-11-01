import math

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def brute_force(points):
    n = len(points)
    min_dist = float("inf")
    pair = (None, None)

    for i in range(n):
        for j in range(i + 1, n):
            d = distance(points[i], points[j])
            if d < min_dist:
                min_dist = d
                pair = (points[i], points[j])

    return min_dist, pair


def divide_and_conquer(points):
    px = sorted(points, key=lambda p: p[0])
    py = sorted(points, key=lambda p: p[1])
    return _recursive_closest(px, py)


def _recursive_closest(px, py):
    n = len(px)
    if n <= 3:
        return brute_force(px)

    mid = n // 2
    mid_x = px[mid][0]

    left_x = px[:mid]
    right_x = px[mid:]

    left_y = [p for p in py if p[0] <= mid_x]
    right_y = [p for p in py if p[0] > mid_x]

    d_left, pair_left = _recursive_closest(left_x, left_y)
    d_right, pair_right = _recursive_closest(right_x, right_y)

    d_min = d_left
    pair_min = pair_left
    if d_right < d_min:
        d_min = d_right
        pair_min = pair_right

    strip = [p for p in py if abs(p[0] - mid_x) < d_min]

    for i in range(len(strip)):
        for j in range(i + 1, min(i + 7, len(strip))):
            d = distance(strip[i], strip[j])
            if d < d_min:
                d_min = d
                pair_min = (strip[i], strip[j])

    return d_min, pair_min

