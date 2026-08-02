"""CropGuard demo — upload a leaf, get a diagnosis, see what it cost.

Run:
    pip install -e ".[demo]"
    streamlit run app/streamlit_app.py

Two backends:
  Local ONNX  — needs models/cropguard.onnx. Fast (~25 ms), for trying things out.
  Live API    — the deployed Render service. Slow (~3.5 s on 0.1 vCPU), but it is the
                real system, and the latency it reports is the honest one.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import streamlit as st

DEFAULT_API = "https://cropguard-api-w9ch.onrender.com"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL = REPO_ROOT / "models" / "cropguard.onnx"
CLASSES_PATH = REPO_ROOT / "configs" / "classes.json"

# Label smoothing (eps=0.1) over K=38 classes bounds the achievable softmax output at
# (1 - eps) + eps/K. The model cannot report more than this, so the UI must not imply it can.
CONFIDENCE_CEILING = 0.9 + 0.1 / 38

st.set_page_config(page_title="CropGuard", page_icon="🌿", layout="wide")

# Tokens from the reference palette; dark values are stepped for the dark surface rather
# than flipped from the light ones.
STYLE = """
<style>
.cg {
  --surface:   #fcfcfb;
  --ink:       #0b0b0b;
  --ink-2:     #52514e;
  --muted:     #898781;
  --grid:      #e1e0d9;
  --accent:    #2a78d6;
  --rest:      #c3c2b7;
  --ring:      rgba(11,11,11,0.10);
}
@media (prefers-color-scheme: dark) {
  .cg {
    --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --accent: #3987e5; --rest: #383835;
    --ring: rgba(255,255,255,0.10);
  }
}
.cg { color: var(--ink); }
.cg-hero { font-size: 15px; color: var(--ink-2); margin: 0 0 2px 0; }
.cg-class { font-size: 30px; font-weight: 650; line-height: 1.15; margin: 0 0 10px 0; }
.cg-fig { font-size: 52px; font-weight: 680; line-height: 1; letter-spacing: -0.02em; }
.cg-unit { font-size: 17px; color: var(--ink-2); font-weight: 500; margin-left: 3px; }
.cg-note { font-size: 12.5px; color: var(--muted); margin-top: 5px; line-height: 1.45; }

.cg-row { display: grid; grid-template-columns: 1fr auto; gap: 10px;
          align-items: center; margin-bottom: 2px; }
.cg-name { font-size: 13px; color: var(--ink-2); overflow: hidden;
           text-overflow: ellipsis; white-space: nowrap; }
.cg-val  { font-size: 13px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
/* Track carries the gridline colour so an empty bar still reads as a scale. */
.cg-track { height: 9px; background: var(--grid); border-radius: 4px;
            overflow: hidden; margin-bottom: 11px; }
.cg-bar { height: 100%; border-radius: 0 4px 4px 0; }
.cg-bar-1 { background: var(--accent); }
.cg-bar-n { background: var(--rest); }

.cg-tile { border: 1px solid var(--ring); border-radius: 10px;
           padding: 12px 14px; background: var(--surface); }
.cg-tile-label { font-size: 12px; color: var(--muted); margin-bottom: 3px; }
.cg-tile-val { font-size: 21px; font-weight: 620; font-variant-numeric: tabular-nums; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_local_model():
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from cropguard.serving.model_loader import CropGuardModel

    return CropGuardModel(LOCAL_MODEL, CLASSES_PATH, "local-fp32")


def predict_local(image_bytes: bytes) -> tuple[dict, float]:
    model = load_local_model()
    start = time.perf_counter()
    result = model.predict(image_bytes, top_k=3)
    elapsed = (time.perf_counter() - start) * 1000
    result["latency_ms"] = round(elapsed, 1)
    return result, elapsed


def predict_remote(image_bytes: bytes, base_url: str) -> tuple[dict, float]:
    boundary = uuid.uuid4().hex
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="leaf.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode()
        + image_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )

    request = urllib.request.Request(
        base_url.rstrip("/") + "/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read())
    return result, (time.perf_counter() - start) * 1000


def pretty(class_name: str) -> str:
    crop, _, disease = class_name.partition("___")
    crop = crop.replace("_", " ").strip()
    disease = disease.replace("_", " ").strip() or "healthy"
    return f"{crop} — {disease}"


def bars(top_k: list[dict]) -> str:
    """Emphasis form: rank 1 in the accent hue, the rest in the recessive gray.

    Not a categorical palette — the classes are not the subject, the winner is. Every bar is
    directly labelled, so identity never depends on colour.
    """
    rows = []
    for rank, entry in enumerate(top_k):
        pct = max(entry["probability"] * 100, 0.6)  # keep a sliver visible at ~0
        fill = "cg-bar-1" if rank == 0 else "cg-bar-n"
        rows.append(
            f'<div class="cg-row"><div class="cg-name">{pretty(entry["class_name"])}</div>'
            f'<div class="cg-val">{entry["probability"]:.1%}</div></div>'
            f'<div class="cg-track"><div class="cg-bar {fill}" style="width:{pct:.2f}%"></div></div>'
        )
    return '<div class="cg">' + "".join(rows) + "</div>"


def tile(label: str, value: str) -> str:
    return (
        f'<div class="cg cg-tile"><div class="cg-tile-label">{label}</div>'
        f'<div class="cg-tile-val">{value}</div></div>'
    )


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("CropGuard")
st.sidebar.caption("38-class crop disease classifier · ResNet50 · ONNX")

local_available = LOCAL_MODEL.exists()
options = ["Live API (Render)"] + (["Local ONNX"] if local_available else [])
backend = st.sidebar.radio("Backend", options, help="Local is fast; the API is the real thing.")

api_url = DEFAULT_API
if backend.startswith("Live"):
    api_url = st.sidebar.text_input("API URL", DEFAULT_API)
    st.sidebar.caption(
        "Render's free tier runs on 0.1 vCPU, so expect ~3.5 s per image — and up to "
        "~50 s extra on the first request after 15 minutes idle, while the instance wakes."
    )
elif not local_available:
    st.sidebar.warning("models/cropguard.onnx not found — run the export step first.")

st.sidebar.divider()
if st.sidebar.button("Check API health", use_container_width=True):
    try:
        with urllib.request.urlopen(api_url.rstrip("/") + "/health", timeout=120) as r:
            st.sidebar.json(json.loads(r.read()))
    except Exception as exc:  # noqa: BLE001 — surfaced to the user, not swallowed
        st.sidebar.error(f"{type(exc).__name__}: {exc}")

# --------------------------------------------------------------------------- main
st.markdown(
    '<div class="cg"><div class="cg-class">Leaf disease diagnosis</div></div>',
    unsafe_allow_html=True,
)
st.caption(
    "Trained on PlantVillage — leaves photographed against a plain background. "
    "Field photographs are a different distribution and have not been evaluated."
)

uploaded = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded is None:
    st.info("Upload a leaf image (JPEG or PNG, at least 128×128) to run a prediction.")
    st.stop()

image_bytes = uploaded.getvalue()
left, right = st.columns([1, 1.35], gap="large")

with left:
    st.image(io.BytesIO(image_bytes), use_container_width=True)
    st.caption(f"{uploaded.name} · {len(image_bytes) / 1024:.0f} KB")

with right:
    try:
        with st.spinner("Running inference..."):
            if backend.startswith("Local"):
                result, wall_ms = predict_local(image_bytes)
                server_ms = result["latency_ms"]
            else:
                result, wall_ms = predict_remote(image_bytes, api_url)
                server_ms = result.get("latency_ms", float("nan"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        st.error(f"HTTP {exc.code} — {detail}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

    confidence = result["confidence"]
    st.markdown(
        f'<div class="cg"><div class="cg-hero">Predicted</div>'
        f'<div class="cg-class">{pretty(result["predicted_class"])}</div>'
        f'<div class="cg-fig">{confidence:.1%}<span class="cg-unit">confidence</span></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    if confidence >= CONFIDENCE_CEILING - 0.01:
        st.markdown(
            f'<div class="cg"><div class="cg-note">At the ceiling. Label smoothing '
            f"(ε=0.1, K=38) bounds confidence at {CONFIDENCE_CEILING:.1%}, so this is as "
            f"certain as the model can report — not a limit on how certain it is. "
            f"Uncalibrated; do not read as a probability.</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown(bars(result["top_k"]), unsafe_allow_html=True)

st.divider()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(tile("Inference", f"{server_ms:.0f} ms"), unsafe_allow_html=True)
with c2:
    st.markdown(tile("Round trip", f"{wall_ms:.0f} ms"), unsafe_allow_html=True)
with c3:
    st.markdown(
        tile("Uncertainty", f"{result.get('uncertainty', float('nan')):.3f}"),
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(tile("Model", result.get("model_version", "—")), unsafe_allow_html=True)

st.caption(
    "**Inference** is model time only; **round trip** adds network and request handling. "
    "**Uncertainty** is normalized predictive entropy (0 = decisive, 1 = uniform) — it "
    "measures spread in this one softmax, so it cannot flag an input that is simply unlike "
    "anything in training."
)

with st.expander("All 38 class probabilities"):
    st.dataframe(
        [
            {"class": pretty(e["class_name"]), "probability": round(e["probability"], 5)}
            for e in result["top_k"]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption("The API returns the top 3; request more with `top_k` when calling it directly.")
