import numpy as np

color_hist = {0: 1000}

def unique_color(color):
    """
    Computes a unique value for uint8 array, e.g. for identifying the input color to make variance computation easy
    :param color: nd.array<n>
    """
    return sum([255 ** i * c for i, c in enumerate(color)])

def to_inverse_count(color):
    return 1 / color_hist.get(color, 1)


def exciting_color_score(img, x, y, w, h):
    selection = img[y:y + h, x:x + w]
    colors_codes = np.apply_along_axis(unique_color, axis=2, arr=selection)
    inverse_counts = np.vectorize(to_inverse_count)(colors_codes)
    return np.mean(inverse_counts)


def select(img, x, y, w, h):
    base_score = exciting_color_score(img, x, y, w, h)
    while w > 8:
        left_cut_score = exciting_color_score(img, x + w // 5, y, w - w // 5, h)
        if left_cut_score > base_score:
            x = x + w // 5
            w -= w // 5
            base_score = left_cut_score
        else:
            break
    while w > 8:
        right_cut_score = exciting_color_score(img, x, y, w - w // 5, h)
        if right_cut_score > base_score:
            w -= w // 5
            base_score = right_cut_score
        else:
            break
    while h > 9:
        left_cut_score = exciting_color_score(img, x, y + h // 5, w, h - h // 5)
        if left_cut_score > base_score:
            y = y + h // 5
            h -= h // 5
            base_score = left_cut_score
        else:
            break
    while h > 9:
        right_cut_score = exciting_color_score(img, x, y, w, h - h // 5)
        if right_cut_score > base_score:
            h -= h // 5
            base_score = right_cut_score
        else:
            break
    return (-1, -1, -1, -1) if base_score < 1e-3 else (x, y, w, h)


img = np.zeros((210, 160, 3))
img[10:15, 13:18] = np.array([100, 100, 100])
print(select(img, 10, 10, 40, 40))
