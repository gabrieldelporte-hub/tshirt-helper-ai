"""Image upscaling via Real-ESRGAN on Replicate."""

from PIL import Image


def upscale_image(image: Image.Image, api_token: str, scale: int = 4) -> Image.Image:
    """
    Upscale an image using Real-ESRGAN via Replicate API.
    Model: nightmareai/real-esrgan
    Scale: 2, 4 or 8
    """
    import io, base64, json, time, urllib.request, urllib.error

    VERSION_ID = "f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa"

    # Real-ESRGAN input limit: ~1024px recommended, 1440px max
    # Downscale before sending so the x4 result stays manageable
    MAX_INPUT = 1024
    send_image = image.copy()
    if max(send_image.size) > MAX_INPUT:
        ratio = MAX_INPUT / max(send_image.size)
        new_size = (int(send_image.width * ratio), int(send_image.height * ratio))
        send_image = send_image.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    send_image.convert("RGBA").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    data_uri = f"data:image/png;base64,{b64}"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }

    payload = json.dumps({
        "version": VERSION_ID,
        "input": {
            "image": data_uri,
            "scale": scale,
            "face_enhance": False,
        },
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
            raise RuntimeError("Cle API Replicate invalide.")
        if e.code == 402:
            raise RuntimeError(
                "Paiement requis (402). "
                "Va sur replicate.com > Settings > Billing et ajoute une carte."
            )
        raise RuntimeError(f"Erreur Replicate {e.code} : {body[:300]}")

    # Poll until complete
    poll_url = prediction.get("urls", {}).get("get") or f"https://api.replicate.com/v1/predictions/{prediction['id']}"
    for _ in range(120):
        status = prediction.get("status")
        if status == "succeeded":
            break
        if status in ("failed", "canceled"):
            raise RuntimeError(f"Upscaling echoue : {prediction.get('error', status)}")
        time.sleep(3)
        poll_req = urllib.request.Request(poll_url, headers={"Authorization": f"Bearer {api_token}"})
        with urllib.request.urlopen(poll_req, timeout=30) as r:
            prediction = json.loads(r.read())
    else:
        raise RuntimeError("Upscaling : timeout apres 6 minutes.")

    output = prediction.get("output")
    if not output:
        raise RuntimeError("Upscaling : aucune sortie recue")

    output_url = output[0] if isinstance(output, list) else output
    with urllib.request.urlopen(output_url, timeout=60) as r:
        result_data = r.read()

    return Image.open(io.BytesIO(result_data)).convert("RGBA")
