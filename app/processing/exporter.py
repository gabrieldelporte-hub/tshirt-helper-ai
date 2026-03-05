"""Export processed images in various formats and DPI settings."""

from pathlib import Path
from PIL import Image


PRINT_DPI = 300
SCREEN_DPI = 96

STANDARD_SIZES_CM = {
    "A4 (21x29.7 cm)": (21, 29.7),
    "A3 (29.7x42 cm)": (29.7, 42),
    "Chest print (30x30 cm)": (30, 30),
    "Full front (40x50 cm)": (40, 50),
    "Small logo (10x10 cm)": (10, 10),
}


def cm_to_px(cm: float, dpi: int) -> int:
    return round(cm * dpi / 2.54)


def export_png(
    image: Image.Image,
    output_path: str | Path,
    dpi: int = PRINT_DPI,
) -> Path:
    """Export as PNG with transparency, at the given DPI."""
    out = Path(output_path)
    img = image.convert("RGBA")
    img.save(out, format="PNG", dpi=(dpi, dpi))
    return out


def export_resized(
    image: Image.Image,
    output_path: str | Path,
    width_cm: float,
    height_cm: float,
    dpi: int = PRINT_DPI,
    keep_aspect: bool = True,
) -> Path:
    """Resize and export at print resolution."""
    out = Path(output_path)
    target_w = cm_to_px(width_cm, dpi)
    target_h = cm_to_px(height_cm, dpi)

    img = image.convert("RGBA")

    if keep_aspect:
        img.thumbnail((target_w, target_h), Image.LANCZOS)
    else:
        img = img.resize((target_w, target_h), Image.LANCZOS)

    img.save(out, format="PNG", dpi=(dpi, dpi))
    return out


def get_image_info(image: Image.Image, dpi: int | None = None) -> dict:
    """Return basic metadata about the image."""
    w, h = image.size
    effective_dpi = dpi or PRINT_DPI
    return {
        "width_px": w,
        "height_px": h,
        "width_cm": round(w / effective_dpi * 2.54, 1),
        "height_cm": round(h / effective_dpi * 2.54, 1),
        "mode": image.mode,
        "dpi": effective_dpi,
    }
