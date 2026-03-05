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
    Remove background using BRIA RMBG 2.0 via Replicate HTTP API.
    Uses versioned predictions endpoint (required for community models).
    """
    import io, base64, json, time, urllib.request

    # Latest version of alexgenovese/remove-background-bria-2
    VERSION_ID = "361975516c86bd0f33c31d4f2073070e5cb463318a65afb032a709c1c9804da0"

    buf = io.BytesIO()
    image.convert("RGBA").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    data_uri = f"data:image/png;base64,{b64}"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }

    # Use /v1/predictions with version ID (correct endpoint for community models)
    payload = json.dumps({
        "version": VERSION_ID,
        "input": {"image": data_uri},
    }).encode()

    req = urllib.request.Request(
        "https://api.replicate.com/v1/predictions",
        data=payload,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            prediction = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if e.code == 401:
            raise RuntimeError("Cle API Replicate invalide. Verifie ta cle dans le panneau droit.")
        if e.code == 402:
            raise RuntimeError(
                "Paiement requis sur Replicate (erreur 402).
"
                "Meme avec des credits, Replicate exige une carte bancaire enregistree.
"
                "Va sur replicate.com > Settings > Billing et ajoute une carte."
            )
        raise RuntimeError(f"Erreur Replicate {e.code} : {body[:300]}")

    # Poll until complete
    poll_url = prediction.get("urls", {}).get("get") or f"https://api.replicate.com/v1/predictions/{prediction['id']}"
    for _ in range(60):
        status = prediction.get("status")
        if status == "succeeded":
            break
        if status in ("failed", "canceled"):
            raise RuntimeError(f"BRIA a echoue : {prediction.get('error', status)}")
        time.sleep(2)
        poll_req = urllib.request.Request(poll_url, headers={"Authorization": f"Bearer {api_token}"})
        with urllib.request.urlopen(poll_req, timeout=30) as r:
            prediction = json.loads(r.read())
    else:
        raise RuntimeError("BRIA : timeout — la prediction a pris trop de temps")

    output = prediction.get("output")
    if not output:
        raise RuntimeError("BRIA : aucune sortie recue")

    output_url = output[0] if isinstance(output, list) else output
    with urllib.request.urlopen(output_url, timeout=60) as r:
        result_data = r.read()

    return Image.open(io.BytesIO(result_data)).convert("RGBA")


def apply_color_hints(
    result: Image.Image,
    original: Image.Image,
    exclude_colors: list[tuple[int,int,int]],
    protect_colors: list[tuple[int,int,int]],
    tolerance: int = 25,
) -> Image.Image:
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
    img = image.convert("RGB")
    w, h = img.size
    corners = [
        img.getpixel((0, 0)),
        img.getpixel((w - 1, 0)),
        img.getpixel((0, h - 1)),
        img.getpixel((w - 1, h - 1)),
    ]
    return tuple(int(sum(c[i] for c in corners) / 4) for i in range(3))
