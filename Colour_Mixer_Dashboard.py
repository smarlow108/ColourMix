"""
================================================================================
MODULE NAME   : Colour_Mixer_Dashboard.py
DESCRIPTION   : Tkinter GUI for outreach game involving syringe pumps and a webcam to reproduce colours.  
================================================================================

AUTHOR        : Dr. Sushila Marlow
EMAIL         : s.marlow@imperial.ac.uk
AFFILIATION   : DIGIBAT / Chemical Engineering / Imperial College London
DATE CREATED  : 2026-01-15
LAST MODIFIED : 2026-05-26
VERSION       : 1.3.0

LICENSE       : BSD 3-clause licence
                Copyright © 2026, Imperial College, London. All rights reserved.
"""




# Colour_Mixer_Dashboard.py
# ================================================================
# Streamlit Dashboard for AI Colour Matching System
# Run with: streamlit run "Colour_Mixer_Dashboard.py"
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from PIL import Image

# ================================================================
# ⚙️  CONFIGURATION — must match your main script
# ================================================================
OUTPUT_FOLDER = "colour_mixer_data"
LOG_FILE      = f"{OUTPUT_FOLDER}/experiment_log.csv"
EXCEL_FILE    = f"{OUTPUT_FOLDER}/experiment_log.xlsx"
IMAGES_FOLDER = f"{OUTPUT_FOLDER}/images"
SIGNAL_FILE   = f"{OUTPUT_FOLDER}/dashboard_signal.json"
MAX_TRIES     = 3

# ================================================================

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title = "AI Colour Mixer Dashboard",
    page_icon  = "🧪",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2E4057;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #2E4057;
        margin-bottom: 1rem;
    }
    .score-excellent { color: #00C851; font-weight: bold; font-size: 1.3rem; }
    .score-good      { color: #FFD700; font-weight: bold; font-size: 1.3rem; }
    .score-fair      { color: #FF8800; font-weight: bold; font-size: 1.3rem; }
    .score-poor      { color: #FF4444; font-weight: bold; font-size: 1.3rem; }
    .colour-swatch {
        width: 100%;
        height: 60px;
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

@st.cache_data(ttl=2)
def load_data():
    """Load experiment log — cached for 2 seconds for live refresh."""
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    df = pd.read_csv(LOG_FILE)
    return df


def rgb_to_hex(r, g, b):
    r, g, b = int(max(0, min(255, r))), \
              int(max(0, min(255, g))), \
              int(max(0, min(255, b)))
    return f"#{r:02X}{g:02X}{b:02X}"


def score_label(score):
    if score >= 90: return "🟢 Excellent"
    if score >= 70: return "🟡 Good"
    if score >= 50: return "🟠 Fair"
    return "🔴 Poor"


def score_colour_hex(score):
    if score >= 90: return "#00C851"
    if score >= 70: return "#FFC107"
    if score >= 50: return "#FF8800"
    return "#FF4444"


def get_last_updated():
    if os.path.exists(SIGNAL_FILE):
        with open(SIGNAL_FILE) as f:
            data = json.load(f)
        return data.get("last_updated", "Unknown")
    return "No data yet"


def get_images_for_colour(colour_id):
    """Return sorted list of image paths for a given colour ID."""
    pattern = os.path.join(IMAGES_FOLDER, f"colour{colour_id}_*.jpg")
    return sorted(glob.glob(pattern))


def colour_swatch_html(hex_colour, label, rgb_text):
    """Render an inline HTML colour swatch."""
    brightness = sum(int(hex_colour[i:i+2], 16) * w
                     for i, w in zip([1, 3, 5], [0.299, 0.587, 0.114]))
    text_col = "#000" if brightness > 128 else "#fff"
    return f"""
    <div style="
        background-color: {hex_colour};
        border-radius: 10px;
        padding: 18px 10px;
        text-align: center;
        border: 1px solid #ccc;
        margin: 4px 0;
    ">
        <span style="color:{text_col}; font-weight:bold;
                     font-size:0.85rem;">{label}</span><br>
        <span style="color:{text_col}; font-size:0.75rem;">{rgb_text}</span><br>
        <span style="color:{text_col}; font-size:0.7rem;">{hex_colour}</span>
    </div>
    """


# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/"
             "thumb/b/b8/Colour_icon_blue.svg/240px-Colour_icon_blue.svg.png",
             width=60)
    st.markdown("## 🧪 AI Colour Mixer")
    st.markdown("---")

    # Auto-refresh toggle
    auto_refresh = st.toggle("🔄 Auto-refresh (5s)", value=True)
    if auto_refresh:
        st.markdown(
            '<meta http-equiv="refresh" content="5">',
            unsafe_allow_html=True
        )

    st.markdown(f"**Last updated:**  \n`{get_last_updated()}`")
    st.markdown("---")

    # Filter by colour ID
    df_all = load_data()
    if not df_all.empty:
        colour_ids = sorted(df_all["colour_id"].unique().tolist())
        selected_colour = st.selectbox(
            "🎨 Filter by Colour ID",
            options=["All"] + [str(c) for c in colour_ids]
        )
        st.markdown("---")
        st.markdown(f"**Total experiments:** {len(df_all)}")
        st.markdown(f"**Unique colours:** {df_all['colour_id'].nunique()}")
        if not df_all.empty:
            best = df_all["match_score_pct"].max()
            avg  = df_all["match_score_pct"].mean()
            st.markdown(f"**Best score:** {best:.1f}%")
            st.markdown(f"**Avg score:** {avg:.1f}%")
    else:
        selected_colour = "All"

    st.markdown("---")
    # Download button for Excel
    if os.path.exists(EXCEL_FILE):
        with open(EXCEL_FILE, "rb") as f:
            st.download_button(
                label     = "📥 Download Excel Log",
                data      = f,
                file_name = "experiment_log.xlsx",
                mime      = "application/vnd.openxmlformats-"
                            "officedocument.spreadsheetml.sheet"
            )


# ================================================================
# MAIN CONTENT
# ================================================================

st.markdown('<div class="main-header">🧪 AI Colour Matching Dashboard</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time results from the '
            'syringe pump colour mixing system</div>',
            unsafe_allow_html=True)

# Load data
df = load_data()

if df.empty:
    st.info("⏳ No experiment data yet. Run the colour mixer script "
            "and capture some results to see them here!")
    st.stop()

# Apply colour filter
if selected_colour != "All":
    df_view = df[df["colour_id"] == int(selected_colour)].copy()
else:
    df_view = df.copy()

# ================================================================
# SECTION 1 — KPI METRICS
# ================================================================
st.markdown("## 📊 Overview")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("🔬 Total Experiments", len(df))
with col2:
    st.metric("🎨 Colours Tested", df["colour_id"].nunique())
with col3:
    best_score = df["match_score_pct"].max()
    st.metric("🏆 Best Score", f"{best_score:.1f}%")
with col4:
    avg_score = df["match_score_pct"].mean()
    st.metric("📈 Avg Score", f"{avg_score:.1f}%")
with col5:
    excellent = (df["match_score_pct"] >= 90).sum()
    st.metric("🟢 Excellent Matches", excellent)

st.markdown("---")

# ================================================================
# SECTION 2 — LATEST RESULT SPOTLIGHT
# ================================================================
st.markdown("## 🔍 Latest Result")

latest = df.iloc[-1]
lc1, lc2, lc3 = st.columns([1, 1, 2])

with lc1:
    t_hex = rgb_to_hex(latest["target_r"],
                       latest["target_g"],
                       latest["target_b"])
    st.markdown(colour_swatch_html(
        t_hex,
        "🎯 Target Colour",
        f"R:{latest['target_r']:.0f} "
        f"G:{latest['target_g']:.0f} "
        f"B:{latest['target_b']:.0f}"
    ), unsafe_allow_html=True)

with lc2:
    m_hex = rgb_to_hex(latest["machine_r"],
                       latest["machine_g"],
                       latest["machine_b"])
    st.markdown(colour_swatch_html(
        m_hex,
        "🤖 Machine Colour",
        f"R:{latest['machine_r']:.0f} "
        f"G:{latest['machine_g']:.0f} "
        f"B:{latest['machine_b']:.0f}"
    ), unsafe_allow_html=True)

with lc3:
    score = latest["match_score_pct"]
    s_col = score_colour_hex(score)
    st.markdown(f"""
    <div style="background:#f8f9fa; border-radius:10px;
                padding:1rem; border-left:4px solid {s_col};">
        <h4 style="margin:0; color:#333;">Colour ID {int(latest['colour_id'])}
        — Try {int(latest['try_number'])}/{MAX_TRIES}</h4>
        <p style="font-size:2rem; font-weight:bold;
                  color:{s_col}; margin:0.3rem 0;">
            {score:.1f}% {score_label(score)}
        </p>
        <p style="margin:0; color:#555;">
            RGB Distance: {latest['rgb_distance']:.2f}<br>
            Pump A: {latest['pump_a']:.3f}  | 
            Pump B: {latest['pump_b']:.3f}  | 
            Pump C: {latest['pump_c']:.3f}<br>
            Time: {latest['timestamp']}
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ================================================================
# SECTION 3 — CHARTS
# ================================================================
st.markdown("## 📈 Learning Progress")

chart1, chart2 = st.columns(2)

with chart1:
    fig, ax = plt.subplots(figsize=(7, 4))
    colours_plot = [score_colour_hex(s)
                    for s in df_view["match_score_pct"]]
    ax.plot(df_view["match_score_pct"].values,
            color="mediumseagreen", linewidth=2, zorder=1)
    ax.scatter(range(len(df_view)),
               df_view["match_score_pct"].values,
               c=colours_plot, s=60, zorder=2, edgecolors='white')
    ax.axhline(90, color='green',  linestyle='--',
               alpha=0.5, label='90% target')
    ax.axhline(70, color='orange', linestyle='--',
               alpha=0.3, label='70% threshold')
    ax.set_title("Match Score Over Experiments", fontweight='bold')
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Match Score (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with chart2:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_view["pump_a"].values, marker='o',
            label="Pump A", color='tomato',    linewidth=2)
    ax.plot(df_view["pump_b"].values, marker='s',
            label="Pump B", color='steelblue', linewidth=2)
    ax.plot(df_view["pump_c"].values, marker='^',
            label="Pump C", color='goldenrod', linewidth=2)
    ax.set_title("Pump Ratios Over Experiments", fontweight='bold')
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# Best score per colour bar chart
if df["colour_id"].nunique() > 1:
    st.markdown("### 🏆 Best Score Per Target Colour")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    best_per = df.groupby("colour_id")["match_score_pct"].max()
    bar_cols = [score_colour_hex(s) for s in best_per.values]
    bars = ax.bar(best_per.index.astype(str),
                  best_per.values,
                  color=bar_cols, edgecolor='white', linewidth=0.8)
    ax.axhline(90, color='green', linestyle='--',
               alpha=0.5, label='90% target')
    for bar, val in zip(bars, best_per.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}%", ha='center', va='bottom',
                fontsize=8, fontweight='bold')
    ax.set_title("Best Match Score Per Target Colour", fontweight='bold')
    ax.set_xlabel("Colour ID")
    ax.set_ylabel("Best Score (%)")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ================================================================
# SECTION 4 — COLOUR GALLERY (per colour ID)
# ================================================================
st.markdown("## 🎨 Colour Gallery")

colour_ids_to_show = (
    [int(selected_colour)]
    if selected_colour != "All"
    else sorted(df["colour_id"].unique().tolist())
)

for cid in colour_ids_to_show:
    c_data = df[df["colour_id"] == cid].sort_values("try_number")

    with st.expander(f"🎨 Colour ID {cid} — "
                     f"{len(c_data)} try/tries  |  "
                     f"Best: {c_data['match_score_pct'].max():.1f}%",
                     expanded=(cid == colour_ids_to_show[-1])):

        for _, row in c_data.iterrows():
            tc1, tc2, tc3, tc4 = st.columns([1, 1, 1, 2])

            t_hex = rgb_to_hex(row["target_r"],
                               row["target_g"], row["target_b"])
            m_hex = rgb_to_hex(row["machine_r"],
                               row["machine_g"], row["machine_b"])

            with tc1:
                st.markdown(colour_swatch_html(
                    t_hex, "🎯 Target",
                    f"R:{row['target_r']:.0f} "
                    f"G:{row['target_g']:.0f} "
                    f"B:{row['target_b']:.0f}"
                ), unsafe_allow_html=True)

            with tc2:
                st.markdown(colour_swatch_html(
                    m_hex, "🤖 Machine",
                    f"R:{row['machine_r']:.0f} "
                    f"G:{row['machine_g']:.0f} "
                    f"B:{row['machine_b']:.0f}"
                ), unsafe_allow_html=True)

            with tc3:
                score = row["match_score_pct"]
                s_col = score_colour_hex(score)
                st.markdown(f"""
                <div style="text-align:center; padding:10px;
                            border-radius:8px; background:#f8f9fa;
                            border: 1px solid #ddd; height:100%;">
                    <div style="font-size:0.8rem;
                                color:#555;">Try {int(row['try_number'])}</div>
                    <div style="font-size:1.6rem; font-weight:bold;
                                color:{s_col};">{score:.1f}%</div>
                    <div style="font-size:0.75rem;
                                color:#777;">{score_label(score)}</div>
                    <div style="font-size:0.7rem; color:#999;
                                margin-top:4px;">
                        Δ {row['rgb_distance']:.1f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with tc4:
                # Show pump ratios as a horizontal bar
                fig, ax = plt.subplots(figsize=(4, 0.8))
                pumps = [row["pump_a"], row["pump_b"], row["pump_c"]]
                labels = ["A", "B", "C"]
                colours = ["tomato", "steelblue", "goldenrod"]
                left = 0
                for val, lbl, col in zip(pumps, labels, colours):
                    ax.barh(0, val, left=left, color=col,
                            height=0.5, label=f"{lbl}: {val:.2f}")
                    ax.text(left + val / 2, 0, f"{lbl}\n{val*100:.0f}%",
                            ha='center', va='center',
                            fontsize=7, color='white', fontweight='bold')
                    left += val
                ax.set_xlim(0, 1)
                ax.axis('off')
                ax.legend(loc='upper right', fontsize=6,
                          bbox_to_anchor=(1.0, 1.8))
                fig.tight_layout(pad=0)
                st.pyplot(fig, use_container_width=True)
                plt.close()

        # ── Images for this colour ───────────────────────────
        images = get_images_for_colour(cid)
        if images:
            st.markdown("**📸 Captured Images:**")
            img_cols = st.columns(min(len(images), 4))
            for i, img_path in enumerate(images):
                with img_cols[i % 4]:
                    try:
                        img = Image.open(img_path)
                        label = os.path.basename(img_path)\
                                  .replace(".jpg", "")\
                                  .replace(f"colour{cid}_", "")
                        st.image(img, caption=label,
                                 use_container_width=True)
                    except Exception:
                        st.warning(f"Could not load {img_path}")

st.markdown("---")

# ================================================================
# SECTION 5 — RAW DATA TABLE
# ================================================================
st.markdown("## 📋 Raw Experiment Log")

display_df = df_view.copy()
display_df["match_score_pct"] = display_df["match_score_pct"].apply(
    lambda x: f"{x:.2f}%"
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "match_score_pct" : st.column_config.TextColumn("Score"),
        "pump_a"          : st.column_config.NumberColumn("Pump A", format="%.3f"),
        "pump_b"          : st.column_config.NumberColumn("Pump B", format="%.3f"),
        "pump_c"          : st.column_config.NumberColumn("Pump C", format="%.3f"),
        "target_r"        : st.column_config.NumberColumn("T_R", format="%d"),
        "target_g"        : st.column_config.NumberColumn("T_G", format="%d"),
        "target_b"        : st.column_config.NumberColumn("T_B", format="%d"),
        "machine_r"       : st.column_config.NumberColumn("M_R", format="%d"),
        "machine_g"       : st.column_config.NumberColumn("M_G", format="%d"),
        "machine_b"       : st.column_config.NumberColumn("M_B", format="%d"),
    }
)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "AI Colour Matching System — Sushila Marlow</div>",
    unsafe_allow_html=True
)