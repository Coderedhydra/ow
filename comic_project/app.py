import os
import uuid
import base64
from typing import List, Optional, Tuple

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Google GenAI SDK
try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - allows app to import even if package missing at authoring time
    genai = None
    types = None

from PIL import Image


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(APP_ROOT, "templates")
STATIC_DIR = os.path.join(APP_ROOT, "static")
GENERATED_DIR = os.path.join(STATIC_DIR, "generated")
UPLOADS_DIR = os.path.join(APP_ROOT, "uploads")
GEMINI_KEYS_FILE = os.path.join(APP_ROOT, "gemini_keys.txt")


def ensure_directories() -> None:
    os.makedirs(GENERATED_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)


ensure_directories()

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
DEFAULT_MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-image-preview")


def read_default_keys() -> str:
    if os.path.exists(GEMINI_KEYS_FILE):
        try:
            with open(GEMINI_KEYS_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""
    return ""


def parse_api_keys(keys_text: str) -> List[str]:
    if not keys_text:
        return []
    keys: List[str] = []
    for line in keys_text.splitlines():
        token = line.strip()
        if token and not token.startswith("#"):
            keys.append(token)
    # Environment fallback
    env_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if env_key and env_key not in keys:
        keys.append(env_key)
    return keys


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def build_frame_prompt(
    characters_text: str,
    style_text: str,
    storyline_text: str,
    frame_index: int,
    total_frames: int,
    single_character: bool,
) -> str:
    parts: List[str] = []
    parts.append(
        (
            "You are generating a comic frame. Maintain visual consistency across frames. "
            "Respect the chosen art style, characters, and storyline."
        )
    )
    parts.append(f"Art style: {style_text}.")
    parts.append(f"Characters: {characters_text}.")
    parts.append(f"Storyline (overall): {storyline_text}.")
    parts.append(f"This is frame {frame_index + 1} of {total_frames}.")
    parts.append(
        (
            "Compose a clear, readable comic panel, no text balloons unless necessary for emotion cues. "
            "Use a clean background that supports the scene."
        )
    )
    if single_character:
        parts.append(
            (
                "This panel must feature a single clear character. Enforce anatomy correctness. "
                "STRICT constraints: exactly two eyes (no extra or duplicated eyes), eyes well-aligned; "
                "no duplicate or ghost faces; avoid extra limbs; maintain a single coherent subject in frame."
            )
        )
    else:
        parts.append(
            (
                "Multiple characters allowed. Avoid duplicated or floating faces; ensure anatomy looks correct."
            )
        )
    parts.append(
        (
            "Output only a final rendered image for this comic panel."
        )
    )
    return "\n".join(parts)


def build_contents_for_generation(
    prompt_text: str,
    mockup_path: Optional[str],
) -> List[object]:
    # The google-genai SDK accepts a list of Parts or strings.
    # Include the mockup image if provided to guide or edit from it.
    contents: List[object] = [prompt_text]
    if mockup_path and types is not None:
        try:
            with open(mockup_path, "rb") as f:
                data = f.read()
            mime = infer_mime_type(mockup_path)
            contents.append(types.Part.from_bytes(data=data, mime_type=mime))
        except Exception as e:
            print(f"[warn] Failed to attach mockup image: {e}")
    return contents


def infer_mime_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def extract_image_bytes_from_response(response) -> Tuple[Optional[bytes], Optional[str]]:
    """Try to extract image bytes and mime type from a google-genai response object."""
    # Try to locate any part with inline image data
    try:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None, None
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if not parts:
                continue
            for p in parts:
                # Common shape: p.inline_data.mime_type, p.inline_data.data
                inline_data = getattr(p, "inline_data", None)
                if inline_data is not None:
                    mime = getattr(inline_data, "mime_type", "image/png")
                    data = getattr(inline_data, "data", None)
                    if data is None:
                        continue
                    if isinstance(data, bytes):
                        return data, mime
                    try:
                        return base64.b64decode(data), mime
                    except Exception:
                        pass
                # Alternative: p.blob or p.file_data depending on SDK versions
                blob = getattr(p, "blob", None)
                if blob is not None:
                    mime = getattr(blob, "mime_type", "image/png")
                    data = getattr(blob, "data", None)
                    if data:
                        if isinstance(data, bytes):
                            return data, mime
                        try:
                            return base64.b64decode(data), mime
                        except Exception:
                            pass
    except Exception as e:  # Defensive
        print(f"[error] extract_image_bytes_from_response failed: {e}")
    return None, None


def try_generate_image_with_key(
    api_key: str,
    model_name: str,
    contents: List[object],
) -> Tuple[Optional[bytes], Optional[str]]:
    if genai is None or types is None:
        raise RuntimeError(
            "google-genai SDK not available. Install dependencies from requirements.txt"
        )

    client = genai.Client(api_key=api_key)

    # Build config requesting IMAGE modality. Do NOT set response_mime_type here,
    # as some backends only allow text mime types and will reject image mimes.
    modality_enum = getattr(types, "ResponseModality", None)
    if modality_enum is not None and hasattr(modality_enum, "IMAGE"):
        response_modalities = [modality_enum.IMAGE]
    else:
        response_modalities = ["IMAGE"]

    config = types.GenerateContentConfig(
        response_modalities=response_modalities,
    )

    resp = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )

    img_bytes, mime = extract_image_bytes_from_response(resp)
    if not img_bytes:
        # Some SDKs may allow direct .to_bytes()
        try:
            if hasattr(resp, "to_bytes"):
                img_bytes = resp.to_bytes()
                mime = "image/png"
        except Exception:
            pass

    return img_bytes, mime


def generate_image_with_key_rotation(
    api_keys: List[str],
    model_name: str,
    contents: List[object],
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """Try keys one by one until success. Returns (image_bytes, mime, error_message)."""
    last_error = None
    for idx, key in enumerate(api_keys):
        try:
            img_bytes, mime = try_generate_image_with_key(key, model_name, contents)
            if img_bytes:
                return img_bytes, mime, None
            last_error = f"No image returned with key index {idx}"
        except Exception as e:
            last_error = f"Key index {idx} failed: {e}"
            # Continue to next key
    return None, None, last_error


def save_png_image(image_bytes: bytes) -> str:
    """Save bytes as a PNG file under static/generated and return the relative path."""
    # Normalize to PNG using Pillow
    img = Image.open(bytearray_to_bytes_io(image_bytes)).convert("RGBA")
    filename = f"{uuid.uuid4().hex}.png"
    out_path = os.path.join(GENERATED_DIR, filename)
    img.save(out_path, format="PNG")
    rel_path = os.path.join("generated", filename)
    return rel_path


def bytearray_to_bytes_io(data: bytes):
    from io import BytesIO

    if isinstance(data, (bytes, bytearray)):
        return BytesIO(bytes(data))
    # Attempt base64 decode as last resort
    try:
        return BytesIO(base64.b64decode(data))
    except Exception:
        return BytesIO(b"")


@app.route("/", methods=["GET"])
def index():
    default_keys = read_default_keys()
    return render_template(
        "index.html",
        default_keys=default_keys,
        default_model=DEFAULT_MODEL_NAME,
    )


@app.route("/generate", methods=["POST"])
def generate():
    # Collect form data
    api_keys_text = request.form.get("api_keys", "")
    model_name = request.form.get("model_name", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
    characters_text = request.form.get("characters", "").strip()
    style_text = request.form.get("style", "").strip()
    storyline_text = request.form.get("storyline", "").strip()

    api_keys = parse_api_keys(api_keys_text)
    if not api_keys:
        flash("Please provide at least one Gemini API key.", "error")
        return redirect(url_for("index"))

    # Single-character flags for 30 frames
    total_frames = 30
    single_flags: List[bool] = [False] * total_frames
    raw_flags = request.form.getlist("single_flags")  # list of frame indices as strings
    try:
        flag_indices = {int(x) for x in raw_flags}
        for i in range(total_frames):
            single_flags[i] = i in flag_indices
    except Exception:
        pass

    # Optional mockup upload
    mockup_path: Optional[str] = None
    if "mockup" in request.files:
        file = request.files["mockup"]
        if file and file.filename:
            filename = secure_filename(file.filename)
            if allowed_file(filename):
                unique_name = f"{uuid.uuid4().hex}_{filename}"
                mockup_path = os.path.join(UPLOADS_DIR, unique_name)
                file.save(mockup_path)
            else:
                flash("Unsupported mockup file type.", "warning")

    # Generate frames
    frame_paths: List[Optional[str]] = [None] * total_frames
    failures: List[str] = []

    for i in range(total_frames):
        prompt = build_frame_prompt(
            characters_text=characters_text,
            style_text=style_text,
            storyline_text=storyline_text,
            frame_index=i,
            total_frames=total_frames,
            single_character=single_flags[i],
        )
        contents = build_contents_for_generation(prompt, mockup_path)
        print(f"[info] Generating frame {i + 1}/{total_frames} (single={single_flags[i]})...")
        img_bytes, mime, err = generate_image_with_key_rotation(api_keys, model_name, contents)
        if img_bytes:
            try:
                rel_path = save_png_image(img_bytes)
                frame_paths[i] = rel_path
                print(f"[ok] Frame {i + 1} saved to {rel_path}")
            except Exception as e:
                msg = f"Frame {i + 1} save failed: {e}"
                print(f"[error] {msg}")
                failures.append(msg)
        else:
            msg = f"Frame {i + 1} generation failed: {err or 'Unknown error'}"
            print(f"[error] {msg}")
            failures.append(msg)

    if failures:
        for m in failures:
            flash(m, "error")
    else:
        flash("All frames generated successfully!", "success")

    # Group into pages of 6 frames each (5 pages)
    pages: List[List[Optional[str]]] = []
    for p in range(5):
        start = p * 6
        end = start + 6
        pages.append(frame_paths[start:end])

    return render_template(
        "result.html",
        pages=pages,
        total_pages=5,
        frames_per_page=6,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)

