"""Background removal — flood fill, AI (rembg), or BRIA RMBG 2.0 (Replicate)."""

from PIL import Image
import numpy as np
from collections import deque


def remove_solid_background(
    image: Image.Image,
    tolerance: int = 30,
    sample_corners: bool = True,
    target_color: tuple[int, int, int] | None = None,
) -> Image.Image:
    """
    Remove a solid background using flood-fill seeded from all image edges.

    Only pixels reachable from the border (and within color tolerance) are
    removed — isolated interior regions of the same color are kept intact.
    """
    img = image.convert("RGBA")
    data = np.array(img, dtype=np.uint8)
    h, w = data.shape[:2]

    if target_color is not None:
        bg = np.array(target_color, dtype=np.int32)
    elif sample_corners:
        corners = [
            data[0, 0, :3],
            data[0, w - 1, :3],
            data[h - 1, 0, :3],
            data[h - 1, w - 1, :3],
        ]
        bg = np.array(corners, dtype=np.int32).mean(axis=0)
    else:
        bg = np.array([255, 255, 255], dtype=np.int32)

    rgb = data[:, :, :3].astype(np.int32)
    diff = np.abs(rgb - bg).max(axis=2)
    is_bg_color = diff <= tolerance

    visited = np.zeros((h, w), dtype=bool)
    queue = deque()

    for x in range(w):
        if is_bg_color[0, x] and not visited[0, x]:
            visited[0, x] = True
            queue.append((0, x))
        if is_bg_color[h - 1, x] and not visited[h - 1, x]:
            visited[h - 1, x] = True
            queue.append((h - 1, x))
    for y in range(h):
        if is_bg_color[y, 0] and not visited[y, 0]:
            visited[y, 0] = True
            queue.append((y, 0))
        if is_bg_color[y, w - 1] and not visited[y, w - 1]:
            visited[y, w - 1] = True
            queue.append((y, w - 1))

    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and is_bg_color[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    result = data.copy()
    result[visited, 3] = 0

    return Image.fromarray(result, "RGBA")


def remove_background_ai(image: Image.Image) -> Image.Image:
    """
    Remove background using rembg via a subprocess (avoids QThread/sys.exit conflicts).
    Downloads the U2Net model (~170 MB) on first use.
    """
    import subprocess, sys, io, base64
    from pathlib import Path

    worker = Path(__file__).parent / "rembg_worker.py"

    buf = io.BytesIO()
    image.convert("RGBA").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue())

    result = subprocess.run(
        [sys.executable, str(worker)],
        input=encoded,
        capture_output=True,
        timeout=120,
    )

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise RuntimeError(f"rembg a echoue :\n{stderr}")

    out_data = base64.b64decode(result.stdout)
    return Image.open(io.BytesIO(out_data)).convert("RGBA")


def remove_background_bria(image: Image.Image, api_token: str) -> Image.Image:
    """
    Remove background using BRIA RMBG 2.0 via Replicate API.
    Requires a valid Replicate API token (replicate.com).
    """
    import replicate
    import io
    import urllib.request

    client = replicate.Client(api_token=api_token)

    buf = io.BytesIO()
    image.convert("RGBA").save(buf, format="PNG")
    buf.seek(0)

    output = client.run(
        "bria-ai/bria-rmbg-2.0",
        input={"image": buf},
    )

    # output is a FileOutput object or URL string
    if hasattr(output, "read"):
        result_data = output.read()
    elif isinstance(output, str):
        with urllib.request.urlopen(output) as resp:
            result_data = resp.read()
    else:
        # replicate FileOutput
        result_data = bytes(output)

    return Image.open(io.BytesIO(result_data)).convert("RGBA")


def apply_color_hints(
    result: Image.Image,
    original: Image.Image,
    exclude_colors: list[tuple[int,int,int]],
    protect_colors: list[tuple[int,int,int]],
    tolerance: int = 25,
) -> Image.Image:
    """
    Post-process a removal result using user-defined color hints.

    - exclude_colors: pixels matching these colors in the original -> made transparent
    - protect_colors: pixels matching these colors in the original -> restored opaque
    """
    orig = np.array(original.convert("RGBA"), dtype=np.int32)
    data = np.array(result.convert("RGBA"), dtype=np.uint8)
    rgb = orig[:, :, :3]

    for color in exclude_colors:
        bg = np.array(color, dtype=np.int32)
        mask = np.abs(rgb - bg).max(axis=2) <= tolerance
        data[mask, 3] = 0

    for color in protect_colors:
        bg = np.array(color, dtype=np.int32)
        mask = np.abs(rgb - bg).max(axis=2) <= tolerance
        data[mask, 3] = 255
        data[mask, :3] = orig[mask, :3].astype(np.uint8)

    return Image.fromarray(data, "RGBA")


def refine_edges(
    image: Image.Image,
    threshold: int = 128,
    erode: int = 0,
    feather: int = 0,
) -> Image.Image:
    """
    Post-process the alpha channel of an RGBA image for cleaner edges.
    """
    from PIL import ImageFilter
    from scipy.ndimage import binary_erosion

    img = image.convert("RGBA")
    data = np.array(img, dtype=np.uint8)
    alpha = data[:, :, 3].astype(np.float32)

    if threshold > 0:
        alpha = np.where(alpha >= threshold, 255.0, 0.0)

    if erode > 0:
        mask = alpha > 127
        structure = np.ones((erode * 2 + 1, erode * 2 + 1), dtype=bool)
        mask = binary_erosion(mask, structure=structure)
        alpha = np.where(mask, alpha, 0.0)

    if feather > 0:
        alpha_img = Image.fromarray(alpha.astype(np.uint8), "L")
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=feather))
        alpha = np.array(alpha_img, dtype=np.float32)

    data[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(data, "RGBA")


def get_background_color(image: Image.Image) -> tuple[int, int, int]:
    """Sample the most likely background color from image corners."""
    img = image.convert("RGB")
    w, h = img.size
    corners = [
        img.getpixel((0, 0)),
        img.getpixel((w - 1, 0)),
        img.getpixel((0, h - 1)),
        img.getpixel((w - 1, h - 1)),
    ]
    return tuple(int(sum(c[i] for c in corners) / 4) for i in range(3))
