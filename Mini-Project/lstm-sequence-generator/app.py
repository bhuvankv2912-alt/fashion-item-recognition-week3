"""
app.py: Production-Ready Streamlit Frontend for LSTM Sequence Generation.

Features:
- Responsive, premium dark/glassmorphism UI layout.
- Real-time inference with temperature and top-k controls.
- One-click Seed Prompt presets.
- Interactive token probability distribution visualization.
- Model architecture inspector and live in-app model training tab.
"""

import os
import time
import subprocess
import streamlit as st
import pandas as pd

# Lazy import generator after checking model presence
from generator import TextGenerator


# ============================================================================
# 1. Streamlit Page Configuration & Styling
# ============================================================================

st.set_page_config(
    page_title="LSTM Sequence Generator | PyTorch & Streamlit",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern visual aesthetics
st.markdown(
    """
    <style>
    /* Global Styles & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 50%, #4ade80 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        margin-right: 0.5rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-pytorch { background-color: rgba(238, 76, 44, 0.15); color: #f87171; border: 1px solid rgba(238, 76, 44, 0.3); }
    .badge-lstm { background-color: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-streamlit { background-color: rgba(255, 75, 75, 0.15); color: #ff6b6b; border: 1px solid rgba(255, 75, 75, 0.3); }

    .output-card {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #e2e8f0;
        white-space: pre-wrap;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }

    .seed-highlight {
        color: #38bdf8;
        font-weight: 600;
        text-decoration: underline;
        text-decoration-color: rgba(56, 189, 248, 0.4);
    }

    .generated-highlight {
        color: #34d399;
    }

    .stat-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
    .stat-val {
        font-size: 1.3rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #94a3b8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# 2. Model Cache Loader
# ============================================================================

MODEL_PATH = os.path.join("models", "model.pt")
VOCAB_PATH = os.path.join("models", "vocab.json")

@st.cache_resource(show_spinner="Loading PyTorch LSTM weights into memory...")
def load_generator(model_path: str, vocab_path: str) -> TextGenerator:
    """Load and cache the inference engine."""
    return TextGenerator(model_path=model_path, vocab_path=vocab_path)


# ============================================================================
# 3. Sidebar UI: Controls & Parameters
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Generation Controls")

    # Temperature Slider
    temperature = st.slider(
        "Temperature (Softmax Scaling)",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.05,
        help=(
            "Controls sampling randomness:\n"
            "• Lower (< 0.5): Conservative, repetitive, highly confident.\n"
            "• Balanced (0.6 - 0.8): Creative, cohesive poetry.\n"
            "• Higher (> 1.0): Adventurous, unpredictable, higher entropy."
        ),
    )

    # Dynamic temperature status pill
    if temperature < 0.4:
        st.caption("🔒 **Mode:** Conservative & Deterministic")
    elif temperature <= 0.85:
        st.caption("✨ **Mode:** Balanced & Poetic (Recommended)")
    else:
        st.caption("🔥 **Mode:** Creative & High Entropy")

    # Max Length Slider
    max_length = st.slider(
        "Max Length (Tokens / Characters)",
        min_value=50,
        max_value=600,
        value=220,
        step=10,
        help="Number of characters to generate following the seed prompt.",
    )

    # Top-K Sampling
    top_k = st.slider(
        "Top-K Sampling Filter",
        min_value=0,
        max_value=30,
        value=0,
        step=1,
        help="0 to disable. Restricts candidate pool to the top-K highest scoring tokens.",
    )

    st.markdown("---")
    st.markdown("### 💡 Seed Inspiration Presets")
    preset_prompts = [
        "Shall I compare thee to a summer's day?",
        "Once upon a midnight dreary,",
        "Tyger Tyger, burning bright,",
        "Do not go gentle into that good night,",
        "The sea is calm tonight,",
    ]

    selected_preset = st.selectbox(
        "Select a classic preset prompt:",
        ["Custom Input"] + preset_prompts,
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📦 Model Status")
    model_exists = os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH)

    if model_exists:
        st.success("✅ Model checkpoint & vocab loaded")
    else:
        st.warning("⚠️ No trained checkpoint found in `models/`")
        if st.button("🚀 Train Quick Model Now (15 Epochs)", use_container_width=True):
            with st.spinner("Training LSTM model on poetry corpus..."):
                try:
                    res = subprocess.run(
                        ["python", "train.py", "--epochs", "15", "--batch_size", "32"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    st.success("Training finished successfully! Reloading...")
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")


# ============================================================================
# 4. Main Page Header & Hero
# ============================================================================

st.markdown('<div class="main-title">⚡ Deep Sequence Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Autoregressive recurrent neural network language generation powered by PyTorch LSTM and Streamlit'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <span class="badge badge-pytorch">PyTorch 2.x</span>
    <span class="badge badge-lstm">Multi-Layer LSTM</span>
    <span class="badge badge-streamlit">Streamlit UI</span>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# Determine default text for the prompt area
default_prompt = (
    selected_preset if selected_preset != "Custom Input"
    else "Shall I compare thee to a summer's day?"
)

seed_prompt = st.text_area(
    "Enter Seed Prompt (Context for LSTM):",
    value=default_prompt,
    height=85,
    help="Type any sentence or verse. The model will condition on these characters to continue the sequence.",
)

col1, col2 = st.columns([1, 4])
with col1:
    generate_btn = st.button("✨ Generate Sequence", type="primary", use_container_width=True)

# ============================================================================
# 5. Generation Execution & Results
# ============================================================================

if generate_btn:
    if not model_exists:
        st.error("Model checkpoint not found! Click 'Train Quick Model Now' in the sidebar first.")
    else:
        generator = load_generator(MODEL_PATH, VOCAB_PATH)

        with st.spinner("🧠 Propagating recurrent hidden states & sampling tokens..."):
            t0 = time.time()
            result = generator.generate(
                seed_prompt=seed_prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
            )
            elapsed_sec = max(time.time() - t0, 1e-4)

        tokens_gen = result["token_count"]
        speed = tokens_gen / elapsed_sec

        # Store in session state for tab inspection
        st.session_state["last_result"] = result
        st.session_state["elapsed"] = elapsed_sec
        st.session_state["speed"] = speed

# Display results if available in session state
if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    elapsed = st.session_state["elapsed"]
    speed = st.session_state["speed"]

    # Metrics row
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="stat-box"><div class="stat-val">{res["token_count"]}</div><div class="stat-label">Characters Generated</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="stat-box"><div class="stat-val">{elapsed * 1000:.1f} ms</div><div class="stat-label">Inference Latency</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="stat-box"><div class="stat-val">{speed:.0f} chars/s</div><div class="stat-label">Throughput</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="stat-box"><div class="stat-val">{res["temperature"]:.2f}</div><div class="stat-label">Sampling Temp</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Tabs
    tab_gen, tab_dist, tab_arch, tab_math = st.tabs([
        "📜 Generated Sequence",
        "📊 Probability Distribution",
        "🧠 Model Architecture",
        "📐 How Temperature Sampling Works",
    ])

    with tab_gen:
        st.markdown("#### Model Output")
        st.markdown(
            f'<div class="output-card">'
            f'<span class="seed-highlight">{res["seed_prompt"]}</span>'
            f'<span class="generated-highlight">{res["new_text"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption("Blue underlined: User Seed Prompt | Green: Autoregressive Generated Sequence")

        # Copy-friendly plain text area
        st.text_area("Raw Text (Copyable):", value=res["generated_text"], height=160)

    with tab_dist:
        st.markdown("#### Next-Token Probability Distribution at Current Step")
        st.write(
            f"These were the top candidate characters considered for the final token at **Temperature = {temperature}**:"
        )

        candidates = res.get("top_candidates", [])
        if candidates:
            df_candidates = pd.DataFrame(candidates)
            st.bar_chart(
                data=df_candidates.set_index("token")["probability"],
                color="#38bdf8",
                use_container_width=True,
            )
            st.dataframe(df_candidates, hide_index=True, use_container_width=True)

    with tab_arch:
        st.markdown("#### Neural Network Layer Breakdown")
        generator = load_generator(MODEL_PATH, VOCAB_PATH)
        summary = generator.model.get_summary()

        col_a, col_b = st.columns(2)
        with col_a:
            st.json(summary)
        with col_b:
            st.markdown(
                """
                **Layer Composition:**
                1. **Embedding Layer**: Translates discrete character token IDs into 128-dimensional dense vectors.
                2. **LSTM Stack**: 2-layer Recurrent Neural Network with 256 hidden units per cell and dropout regularization.
                3. **Dropout Layer**: Regularization preventing co-adaptation of recurrent states.
                4. **Linear / Projection**: Fully-connected dense layer projecting hidden vectors back to vocabulary logits.
                """
            )

        # Plot loss curve if history is present
        if generator.history:
            st.markdown("#### Training & Validation Loss History")
            history_df = pd.DataFrame(generator.history)
            st.line_chart(
                data=history_df.set_index("epoch")[["train_loss", "val_loss"]],
                use_container_width=True,
            )

    with tab_math:
        st.markdown("#### Mathematical Formulation of Temperature Sampling")
        st.markdown(
            r"""
            Given the model's raw unnormalized logit output vector $\mathbf{z} \in \mathbb{R}^{V}$ where $V$ is vocabulary size:

            $$P(w_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^{V} \exp(z_j / T)}$$

            - **When $T \to 0$**: The highest logit dominates; distribution approaches an argmax (greedy / deterministic generation).
            - **When $T = 1.0$**: Standard unscaled softmax probability distribution as learned by the network.
            - **When $T > 1.0$**: Smooths the distribution, increasing the relative probabilities of lower-ranked tokens, yielding higher entropy and creative surprises.
            """
        )
