# Comic Generator (Flask + Google GenAI)

A simple Flask web app that generates a 5-page comic (30 frames total) using Google GenAI (Gemini) image generation.

## Features

- Web UI to enter one or more API keys (auto-rotate on failures)
- Optional mockup image upload (used as a guide via `types.Part.from_bytes`)
- Enter characters, style, and storyline
- Per-frame checkbox to enforce single-character constraints (eyes aligned, no duplicates, no ghost faces)
- Saves all generated frames to `static/generated/` as PNG
- Robust logging and error handling; failed frames show placeholders

## Project Structure

```
comic_project/
├─ app.py
├─ requirements.txt
├─ README.md
├─ gemini_keys.txt        # optional, not required
├─ templates/
│  ├─ index.html
│  └─ result.html
├─ static/
│  └─ generated/
└─ uploads/
```

## Prerequisites

- Python 3.10+
- A Google GenAI (Gemini) API key. You can provide multiple keys and the app will rotate between them on failures or rate limits.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the App

```bash
export FLASK_DEBUG=1  # optional
export GOOGLE_API_KEY=YOUR_GEMINI_KEY  # optional fallback if not provided in the UI
# or: export GEMINI_API_KEY=YOUR_GEMINI_KEY
python app.py
```

App will start on `http://localhost:5000` (binds to `0.0.0.0`).

## Using the UI

1. Open the app in your browser.
2. Paste one or more API keys (one per line). You can also put them in `gemini_keys.txt` (same directory as `app.py`).
3. Optionally upload a mockup image. The app will attach it to the request via `types.Part.from_bytes(...)` to guide or “edit” from the mockup.
4. Enter characters, choose a style, and describe the storyline.
5. Select which frames should be single-character frames using the 30 checkboxes.
6. Click Generate. The app will generate 30 frames, save them as PNGs in `static/generated/`, and show a 5-page result (6 frames per page).

## Model Selection

- Default model is `gemini-2.0-flash`. You can change it in the UI or by setting env var:

```bash
export GEMINI_MODEL=gemini-2.0-flash
```

- In `app.py`, generation is performed using:
  - `client.models.generate_content(model=..., contents=..., config=...)`
  - The `config` sets `response_modalities=["IMAGE"]` and requests PNG output when possible.

If your account or region supports other image-capable models, you can switch the model name accordingly.

## Where Keys Go

- Paste into the UI (recommended for quick tests)
- Put one per line into `gemini_keys.txt` (will prefill the UI)
- Or set `GOOGLE_API_KEY` or `GEMINI_API_KEY` in your environment (used as fallback)

## File Locations

- Generated frames: `static/generated/` (each saved as a unique PNG via UUID)
- Mockup uploads: `uploads/`

## Notes on Single-Character Constraints

For frames marked as single-character, the prompt includes strict instructions:
- Exactly two eyes (no extra or duplicated eyes)
- Eyes aligned
- No duplicate or ghost faces
- Clear anatomy (avoid extra limbs)

These constraints are encoded in the prompt to reduce common image artifacts.

## Troubleshooting

- If you see errors like invalid key or rate limit, add more keys or wait and try again.
- Some SDK versions return image bytes in different fields; this app attempts multiple extraction paths.
- If the app cannot import `google-genai`, ensure you installed dependencies and activated your virtualenv.

## License

MIT