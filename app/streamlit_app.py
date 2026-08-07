"""CropGuard demo — upload a leaf, pick a model, get a diagnosis and what it cost.

Run:
    pip install -e ".[demo]"          # API backend only
    pip install -e ".[demo,serve]"    # adds onnxruntime, needed for local models
    streamlit run app/streamlit_app.py

Backends come from configs/models.json plus the deployed API. Only models whose file is
actually on disk are offered, so the panel never lists a model it cannot run.

The compare mode exists because the A/B result is the interesting part of this project: the
challenger has higher macro-F1 and slightly lower accuracy, so on any single image the two
models usually agree and occasionally do not. Watching where they diverge is a better feel for
what "no significant improvement" means than reading the p-value.
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
REGISTRY_PATH = REPO_ROOT / "configs" / "models.json"
CLASSES_PATH = REPO_ROOT / "configs" / "classes.json"
COMPARISON_PATH = REPO_ROOT / "artifacts" / "ab_comparison.json"
API_CHOICE = "Live API (Render)"

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
.cg-class-sm { font-size: 19px; font-weight: 640; line-height: 1.2; margin: 0 0 6px 0; }
.cg-fig { font-size: 52px; font-weight: 680; line-height: 1; letter-spacing: -0.02em; }
.cg-fig-sm { font-size: 30px; font-weight: 670; line-height: 1; letter-spacing: -0.02em; }
.cg-unit { font-size: 17px; color: var(--ink-2); font-weight: 500; margin-left: 3px; }
.cg-unit-sm { font-size: 13px; color: var(--ink-2); font-weight: 500; margin-left: 3px; }
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

.cg-card { border: 1px solid var(--ring); border-radius: 12px;
           padding: 16px 18px; background: var(--surface); height: 100%; }
.cg-eyebrow { font-size: 11.5px; letter-spacing: 0.06em; text-transform: uppercase;
              color: var(--muted); margin-bottom: 8px; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# --------------------------------------------------------------------------- registry


@st.cache_data(show_spinner=False, ttl=600)
def hub_files(repo: str | None) -> set[str] | None:
    """Filenames in the model repo, or None if the Hub could not be reached.

    None and the empty set mean different things and must not be conflated: an unreachable Hub
    is no evidence that a file is absent, so the caller keeps trusting the registry in that
    case. A successful listing that lacks the file *is* evidence, and the model is hidden.
    The TTL lets a newly uploaded model appear without redeploying the Space.
    """
    if not repo:
        return None
    try:
        from huggingface_hub import HfApi

        return set(HfApi().list_repo_files(repo))
    except Exception:  # noqa: BLE001 — offline is a normal state for the local demo
        return None


@st.cache_data(show_spinner=False, ttl=600)
def load_registry() -> list[dict]:
    """Models declared in configs/models.json, annotated with how they can be obtained.

    `available` means runnable *now* without a download; `fetchable` means the weights are on
    the Hub and will be pulled on first use. On Spaces nothing is on disk at startup, so every
    model is fetchable and none is available — which is why the two are tracked separately
    rather than collapsed into one boolean.

    The TTL is the load-bearing part and belongs *here*, not only on `hub_files`. This function
    caches the *derived* answer, so without its own expiry it pins whatever the Hub said when
    the container booted: a model published afterwards stays invisible until the Space is
    restarted, no matter how short the inner TTL is. That is exactly what happened once.
    """
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)

    hub_repo = registry.get("hub_repo")
    published = hub_files(hub_repo)

    models = []
    for entry in registry["models"]:
        path = REPO_ROOT / entry["file"]
        hub_file = entry.get("hub_file")
        fetchable = bool(hub_repo and hub_file) and (published is None or hub_file in published)
        models.append(
            {
                **entry,
                "path": path,
                "hub_repo": hub_repo,
                "available": path.exists(),
                "fetchable": fetchable,
            }
        )
    return models


@st.cache_resource(show_spinner="Downloading weights from the Hub (first run only)...")
def resolve_weights(model_id: str) -> str:
    """Local path to the ONNX file, downloading from the Hub if it is not already here.

    Cached as a resource so the ~95 MB download happens once per container rather than once
    per session. hf_hub_download caches to disk as well, so a restarted Space that kept its
    layer cache does not re-fetch.
    """
    entry = next(m for m in load_registry() if m["id"] == model_id)
    if entry["available"]:
        return str(entry["path"])
    if not entry["fetchable"]:
        raise FileNotFoundError(
            f"{entry['name']} is not on disk and has no hub_file in configs/models.json"
        )

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=entry["hub_repo"], filename=entry["hub_file"])


@st.cache_resource(show_spinner=False)
def load_local_model(model_id: str, temperature: float):
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from cropguard.serving.model_loader import CropGuardModel

    entry = next(m for m in load_registry() if m["id"] == model_id)
    weights = resolve_weights(model_id)
    return CropGuardModel(weights, CLASSES_PATH, entry["model_version"], temperature)


def predict_local(model_id: str, temperature: float, image_bytes: bytes) -> tuple[dict, float]:
    model = load_local_model(model_id, temperature)
    start = time.perf_counter()
    result = model.predict(image_bytes, top_k=38)
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


def run_backend(choice: str, image_bytes: bytes, temperature: float, api_url: str):
    """One entry point for both backends so compare mode does not special-case them."""
    if choice == API_CHOICE:
        result, wall_ms = predict_remote(image_bytes, api_url)
        return result, wall_ms, result.get("latency_ms", float("nan"))
    result, wall_ms = predict_local(choice, temperature, image_bytes)
    return result, wall_ms, result["latency_ms"]


# --------------------------------------------------------------------------- rendering


def pretty(class_name: str) -> str:
    crop, _, disease = class_name.partition("___")
    crop = crop.replace("_", " ").strip()
    disease = disease.replace("_", " ").strip() or "healthy"
    return f"{crop} — {disease}"


def bars(top_k: list[dict], limit: int = 3) -> str:
    """Emphasis form: rank 1 in the accent hue, the rest in the recessive gray.

    Not a categorical palette — the classes are not the subject, the winner is. Every bar is
    directly labelled, so identity never depends on colour.
    """
    rows = []
    for rank, entry in enumerate(top_k[:limit]):
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


def label_for(choice: str, models: list[dict]) -> str:
    if choice == API_CHOICE:
        return API_CHOICE
    return next(m["name"] for m in models if m["id"] == choice)


# ---- the A/B verdict ------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=600)
def load_comparison() -> dict | None:
    """The A/B report: local if this is a checkout, from the Hub if this is a Space."""
    if COMPARISON_PATH.exists():
        with open(COMPARISON_PATH, encoding="utf-8") as f:
            return json.load(f)

    registry_hub = next((m["hub_repo"] for m in models if m.get("hub_repo")), None)
    if not registry_hub:
        return None
    try:
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(registry_hub, "ab_comparison.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 — the report is a bonus, never a hard dependency
        return None


def render_comparison_report() -> None:
    """The A/B verdict. Rendered whether or not an image has been uploaded — it is a result
    about the models, not about anyone's leaf, and hiding it behind the uploader buried the
    most substantive thing on the page."""
    report = load_comparison()
    if report is None:
        return

    verdict_is_win = (
        report["mcnemar"]["p_value"] < 0.05
        and report["mcnemar"]["only_b_correct"] > report["mcnemar"]["only_a_correct"]
        and not (report["bootstrap"]["ci_low"] <= 0 <= report["bootstrap"]["ci_high"])
    )
    headline = (
        "challenger promoted" if verdict_is_win else "no significant improvement — not promoted"
    )

    with st.expander(f"A/B test: ResNet50 vs ConvNeXt-Tiny — {headline}", expanded=False):
        st.markdown(
            "Both models scored on the identical holdout, image by image. The tests are "
            "**paired**, which is what makes them sensitive — and void if the two ever saw "
            "different test sets, so `compare.py` asserts that before testing anything."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                tile("Accuracy", f"{report['accuracy_a']:.4f} → {report['accuracy_b']:.4f}"),
                unsafe_allow_html=True,
            )
        with c2:
            if report.get("macro_f1_a") is not None:
                st.markdown(
                    tile("Macro-F1", f"{report['macro_f1_a']:.4f} → {report['macro_f1_b']:.4f}"),
                    unsafe_allow_html=True,
                )
        with c3:
            st.markdown(
                tile("McNemar p", f"{report['mcnemar']['p_value']:.4f}"), unsafe_allow_html=True
            )

        # Both intervals, because the two metrics move in opposite directions here and showing
        # only one would let a reader pick the flattering story.
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        intervals = [
            {
                "quantity": "accuracy difference",
                "estimate": f"{report['bootstrap']['observed_difference']:+.4f}",
                "95% CI": f"[{report['bootstrap']['ci_low']:+.4f}, "
                f"{report['bootstrap']['ci_high']:+.4f}]",
                "excludes 0": "no"
                if report["bootstrap"]["ci_low"] <= 0 <= report["bootstrap"]["ci_high"]
                else "yes",
            }
        ]
        macro = report.get("macro_f1_bootstrap")
        if macro:
            intervals.append(
                {
                    "quantity": "macro-F1 difference",
                    "estimate": f"{macro['observed_difference']:+.4f}",
                    "95% CI": f"[{macro['ci_low']:+.4f}, {macro['ci_high']:+.4f}]",
                    "excludes 0": "no" if macro["ci_low"] <= 0 <= macro["ci_high"] else "yes",
                }
            )
        st.dataframe(intervals, use_container_width=True, hide_index=True)

        mc = report["mcnemar"]
        st.markdown(
            f"**The comparison rests on {mc['only_a_correct'] + mc['only_b_correct']} images.** "
            f"{mc['both_correct']:,} both got right and {mc['both_wrong']} both got wrong; those "
            f"carry no information about which is better. Of the rest, "
            f"{mc['only_a_correct']} went to the baseline and {mc['only_b_correct']} to the "
            f"challenger — close enough to a coin flip that McNemar returns "
            f"p = {mc['p_value']:.3f}."
        )

        for note in report.get("notes", []):
            st.caption(note)
        st.json(report, expanded=False)


# --------------------------------------------------------------------------- sidebar

models = load_registry()
runnable = [m for m in models if m["available"] or m["fetchable"]]

st.sidebar.title("CropGuard")
st.sidebar.caption("38-class crop disease classifier · ONNX")

choices = [API_CHOICE] + [m["id"] for m in runnable]
mode = st.sidebar.radio("Mode", ["Single model", "Compare models"], horizontal=True)

if mode == "Single model":
    selected = [
        st.sidebar.selectbox(
            "Model",
            choices,
            format_func=lambda c: label_for(c, models),
            help="The API serves whichever model is deployed; local models run in this process.",
        )
    ]
else:
    # Prefer models already on disk, so the default selection never opens with a download.
    # The sort is stable, so registry order decides among equals.
    ready_first = sorted(runnable, key=lambda m: not m["available"])
    default = [m["id"] for m in ready_first[:2]] or [API_CHOICE]
    selected = st.sidebar.multiselect(
        "Models to compare",
        choices,
        default=default,
        format_func=lambda c: label_for(c, models),
    )

api_url = DEFAULT_API
if API_CHOICE in selected:
    api_url = st.sidebar.text_input("API URL", DEFAULT_API)
    st.sidebar.caption(
        "Render's free tier runs on 0.1 vCPU, so expect ~3.5 s per image — and up to "
        "~50 s extra on the first request after 15 minutes idle, while the instance wakes."
    )

to_fetch = [m["name"] for m in runnable if not m["available"]]
if to_fetch:
    st.sidebar.caption(
        "Downloaded from the Hub on first use: "
        + ", ".join(to_fetch)
        + ". The first prediction with one of these pays for the download; later ones do not."
    )

unavailable = [m["name"] for m in models if not (m["available"] or m["fetchable"])]
if unavailable:
    st.sidebar.caption(
        "Not runnable here: " + ", ".join(unavailable) + " — no local file and not on the Hub."
    )
    # An escape hatch, because "wait for a cache to expire" is a terrible answer to "I just
    # published that model and it is not showing".
    if st.sidebar.button("Re-check the Hub", use_container_width=True):
        hub_files.clear()
        load_registry.clear()
        load_comparison.clear()
        st.rerun()

st.sidebar.divider()

# Calibration is a serving-time decision, not a property of the weights, so it belongs in the
# UI. Off by default would be dishonest (the deployed API applies it); a toggle lets someone
# see the ~9-point confidence gap that temperature scaling closes.
calibrated = st.sidebar.toggle(
    "Apply temperature calibration",
    value=True,
    help="Divides logits by the T fitted on that model's validation split. Changes confidence, "
    "never the predicted class — a positive scalar cannot reorder logits.",
)

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

if not selected:
    st.warning("Pick at least one model in the sidebar.")
    st.stop()

if uploaded is None:
    st.info("Upload a leaf image (JPEG or PNG, at least 128×128) to run a prediction.")

    with st.expander("What is in the registry"):
        for entry in models:
            test = entry.get("test")
            measured = (
                f"accuracy {test['accuracy']:.4f} · macro-F1 {test['macro_f1']:.4f} (n={test['n']})"
                if test
                else "no full-holdout evaluation"
            )
            if entry["available"]:
                state = "on disk"
            elif entry["fetchable"]:
                state = f"fetched from {entry['hub_repo']} on first use"
            else:
                state = "not runnable here"
            st.markdown(
                f"**{entry['name']}** — {entry['role']} · {state}  \n"
                f"{entry['architecture']}  \n"
                f"{measured}  \n"
                f"_{entry['note']}_"
            )
    render_comparison_report()
    st.stop()

image_bytes = uploaded.getvalue()


def temperature_for(choice: str) -> float:
    if not calibrated or choice == API_CHOICE:
        return 1.0  # the API applies its own T server-side; doing it again would double-scale
    return next(m["temperature"] for m in models if m["id"] == choice)


# ---- single model ---------------------------------------------------------------------

if mode == "Single model":
    choice = selected[0]
    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.image(io.BytesIO(image_bytes), use_container_width=True)
        st.caption(f"{uploaded.name} · {len(image_bytes) / 1024:.0f} KB")

    with right:
        try:
            with st.spinner("Running inference..."):
                result, wall_ms, server_ms = run_backend(
                    choice, image_bytes, temperature_for(choice), api_url
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            st.error(f"HTTP {exc.code} — {detail}")
            st.stop()
        except ImportError:
            st.error('Local models need onnxruntime: pip install -e ".[demo,serve]"')
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

        if not result.get("calibrated") and confidence >= CONFIDENCE_CEILING - 0.01:
            st.markdown(
                f'<div class="cg"><div class="cg-note">At the ceiling. Label smoothing '
                f"(ε=0.1, K=38) bounds raw confidence at {CONFIDENCE_CEILING:.1%}, so this is "
                f"as certain as the uncalibrated model can report — not a limit on how certain "
                f"it is. Turn on calibration to see the corrected figure.</div></div>",
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

    entry = (
        next((m for m in models if m["role"] == "production"), None)
        if choice == API_CHOICE
        else next(m for m in models if m["id"] == choice)
    )
    if entry and entry.get("test"):
        st.caption(
            f"**{entry['name']}** on the 8,125-image holdout: accuracy "
            f"{entry['test']['accuracy']:.4f}, macro-F1 **{entry['test']['macro_f1']:.4f}**. "
            f"Read macro-F1 — the dataset is imbalanced ~36×, so accuracy is dominated by the "
            f"largest classes. One image tells you nothing about either number."
        )
    elif entry:
        st.caption(
            f"**{entry['name']}** has no full-holdout evaluation, so no accuracy is claimed "
            f"for it. {entry['note']}"
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

# ---- compare --------------------------------------------------------------------------

else:
    head_left, head_right = st.columns([1, 2.2], gap="large")
    with head_left:
        st.image(io.BytesIO(image_bytes), use_container_width=True)
        st.caption(f"{uploaded.name} · {len(image_bytes) / 1024:.0f} KB")

    outcomes = []
    for choice in selected:
        try:
            with st.spinner(f"Running {label_for(choice, models)}..."):
                result, wall_ms, server_ms = run_backend(
                    choice, image_bytes, temperature_for(choice), api_url
                )
            outcomes.append((choice, result, wall_ms, server_ms, None))
        except ImportError:
            outcomes.append(
                (choice, None, 0.0, 0.0, "needs onnxruntime: pip install -e '.[demo,serve]'")
            )
        except Exception as exc:  # noqa: BLE001
            outcomes.append((choice, None, 0.0, 0.0, f"{type(exc).__name__}: {exc}"))

    with head_right:
        succeeded = [o for o in outcomes if o[1] is not None]
        predicted = {o[1]["predicted_class"] for o in succeeded}
        if len(succeeded) < 2:
            st.info("Select two or more working backends to see agreement.")
        elif len(predicted) == 1:
            st.success(f"All {len(succeeded)} agree: **{pretty(next(iter(predicted)))}**")
            st.caption(
                "Agreement on one image is not evidence of equivalence — the models disagreed "
                "on 95 of 8,125 holdout images, so most images look like this one."
            )
        else:
            st.warning("The models disagree on this image.")
            st.caption(
                "This is one of the ~1.2% of holdout images where they diverge. Neither is "
                "known to be right here; the holdout says one is right 49 times and the other "
                "46 times out of those disagreements."
            )

    st.divider()
    columns = st.columns(len(outcomes), gap="large")

    for column, (choice, result, wall_ms, server_ms, error) in zip(columns, outcomes, strict=True):
        with column:
            st.markdown(
                f'<div class="cg"><div class="cg-eyebrow">{label_for(choice, models)}</div></div>',
                unsafe_allow_html=True,
            )
            if error is not None:
                st.error(error)
                continue

            st.markdown(
                f'<div class="cg"><div class="cg-class-sm">'
                f"{pretty(result['predicted_class'])}</div>"
                f'<div class="cg-fig-sm">{result["confidence"]:.1%}'
                f'<span class="cg-unit-sm">confidence</span></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            st.markdown(bars(result["top_k"]), unsafe_allow_html=True)
            st.markdown(
                f'<div class="cg"><div class="cg-note">{server_ms:.0f} ms inference · '
                f"{wall_ms:.0f} ms round trip · uncertainty "
                f"{result.get('uncertainty', float('nan')):.3f}<br>"
                f"{result.get('model_version', '—')}</div></div>",
                unsafe_allow_html=True,
            )

    st.caption(
        "Local models run in this process on this CPU; the API runs on Render's 0.1 vCPU. "
        "Comparing their latencies compares two machines, not two models."
    )

    # ---- measured accuracy, next to the live guess -------------------------------------
    #
    # The prediction above is one image. These are the numbers that actually rank the models,
    # and putting them side by side is the point: a visitor who watches two models agree on
    # their leaf should not conclude anything about which is better.
    st.divider()
    st.markdown(
        '<div class="cg"><div class="cg-eyebrow">Measured on the 8,125-image holdout</div></div>',
        unsafe_allow_html=True,
    )

    rows = []
    for choice, result, _, server_ms, _error in outcomes:
        if choice == API_CHOICE:
            entry = next((m for m in models if m["role"] == "production"), None)
        else:
            entry = next(m for m in models if m["id"] == choice)
        test = (entry or {}).get("test")
        rows.append(
            {
                "model": label_for(choice, models),
                "test accuracy": f"{test['accuracy']:.4f}" if test else "not measured",
                "test macro-F1": f"{test['macro_f1']:.4f}" if test else "not measured",
                "this image": pretty(result["predicted_class"]) if result else "—",
                "inference": f"{server_ms:.0f} ms" if result else "—",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "**Macro-F1 is the number to read**, not accuracy: the dataset is imbalanced ~36×, so "
        "accuracy is dominated by the largest classes. The INT8 build has no full-holdout "
        "figure — it was checked against fp32 on a 3,000-image subset (9 disagreements) and "
        "that is not the same measurement, so nothing is claimed for it here."
    )

render_comparison_report()
