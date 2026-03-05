"""Standalone script called as subprocess to run rembg safely."""
import sys
from PIL import Image
from rembg import remove, new_session
import io, base64

def main():
    # Read base64-encoded image from stdin
    raw = sys.stdin.buffer.read()
    img_data = base64.b64decode(raw)
    image = Image.open(io.BytesIO(img_data)).convert("RGBA")

    session = new_session("u2net")
    result = remove(image, session=session)

    # Write base64-encoded PNG result to stdout
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    sys.stdout.buffer.write(base64.b64encode(buf.getvalue()))

if __name__ == "__main__":
    main()
