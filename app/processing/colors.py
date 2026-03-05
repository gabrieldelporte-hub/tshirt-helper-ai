"""Color utilities: info extraction, RGB <-> CMYK conversion."""

from PIL import Image
import numpy as np


def rgb_to_cmyk(r: int, g: int, b: int) -> tuple[float, float, float, float]:
    """Convert RGB (0-255) to CMYK (0-100 %)."""
    if r == g == b == 0:
        return 0.0, 0.0, 0.0, 100.0

    r_, g_, b_ = r / 255, g / 255, b / 255
    k = 1 - max(r_, g_, b_)
    if k == 1:
        return 0.0, 0.0, 0.0, 100.0

    c = (1 - r_ - k) / (1 - k)
    m = (1 - g_ - k) / (1 - k)
    y = (1 - b_ - k) / (1 - k)

    return round(c * 100, 1), round(m * 100, 1), round(y * 100, 1), round(k * 100, 1)


def cmyk_to_rgb(c: float, m: float, y: float, k: float) -> tuple[int, int, int]:
    """Convert CMYK (0-100 %) to RGB (0-255)."""
    c_, m_, y_, k_ = c / 100, m / 100, y / 100, k / 100
    r = round(255 * (1 - c_) * (1 - k_))
    g = round(255 * (1 - m_) * (1 - k_))
    b = round(255 * (1 - y_) * (1 - k_))
    return r, g, b


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def get_dominant_colors(
    image: Image.Image, n_colors: int = 8, ignore_transparent: bool = True
) -> list[dict]:
    """
    Extract the N most dominant colors from an image.

    Returns a list of dicts with keys: rgb, hex, cmyk, percentage.
    """
    img = image.convert("RGBA")
    data = np.array(img)

    if ignore_transparent:
        alpha = data[:, :, 3]
        pixels = data[alpha > 128, :3]
    else:
        pixels = data[:, :, :3].reshape(-1, 3)

    if len(pixels) == 0:
        return []

    # Quantize to reduce color space, then count
    quantized = (pixels // 32) * 32
    rows = quantized.view([("r", np.uint8), ("g", np.uint8), ("b", np.uint8)])
    unique, counts = np.unique(rows, return_counts=True)
    total = counts.sum()

    top_idx = np.argsort(counts)[::-1][:n_colors]
    result = []
    for idx in top_idx:
        r, g, b = int(unique[idx]["r"]), int(unique[idx]["g"]), int(unique[idx]["b"])
        result.append(
            {
                "rgb": (r, g, b),
                "hex": rgb_to_hex(r, g, b),
                "cmyk": rgb_to_cmyk(r, g, b),
                "percentage": round(counts[idx] / total * 100, 1),
            }
        )
    return result
