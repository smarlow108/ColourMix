# colour_mixer_ai_v2.py
# ================================================================
# AI Colour Matching System — v2
# - Max 3 attempts per target colour
# - Continuous learning across all colours
# - Excel logging for backend review
# - RGB match score (%) displayed after each trial
# ================================================================

import cv2
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ================================================================
# ⚙️  CONFIGURATION
# ================================================================

CAMERA_INDEX      = 1
OUTPUT_FOLDER     = "colour_mixer_data"
MODEL_FILE        = "colour_mixer_data/ai_model.pkl"
SCALER_FILE       = "colour_mixer_data/scaler.pkl"
LOG_FILE          = "colour_mixer_data/experiment_log.csv"
EXCEL_FILE        = "colour_mixer_data/experiment_log.xlsx"
FRAME_WIDTH       = 1920
FRAME_HEIGHT      = 1080
WARMUP_SECONDS    = 1.5
MAX_TRIES         = 3          # Maximum attempts per target colour

ROI_TOP_FRAC      = 0.3
ROI_BOTTOM_FRAC   = 0.7
ROI_INNER_FRAC    = 0.1

# ================================================================


# ----------------------------------------------------------------
# SETUP
# ----------------------------------------------------------------
def setup_folders():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/images", exist_ok=True)


# ----------------------------------------------------------------
# RGB MATCH SCORE
# ----------------------------------------------------------------
def rgb_match_score(rgb1, rgb2):
    """
    Returns a match score from 0% (worst) to 100% (perfect).
    Based on Euclidean distance in RGB space.
    Max possible distance = sqrt(3 * 255^2) ≈ 441.67
    """
    MAX_DIST = np.sqrt(3 * (255 ** 2))
    dist     = np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))
    score    = (1 - dist / MAX_DIST) * 100
    return round(float(score), 2), round(float(dist), 2)


# ----------------------------------------------------------------
# SCORE COLOUR — for display
# ----------------------------------------------------------------
def score_colour(score):
    """Returns a BGR colour based on score for overlay display."""
    if score >= 90:
        return (0, 255, 0)       # Green  — excellent
    elif score >= 70:
        return (0, 200, 255)     # Yellow — good
    elif score >= 50:
        return (0, 140, 255)     # Orange — fair
    else:
        return (0, 0, 255)       # Red    — poor


# ----------------------------------------------------------------
# EXTRACT AVERAGE RGB FROM ROI
# ----------------------------------------------------------------
def extract_avg_rgb(frame, x_start, x_end, top_frac, bottom_frac):
    h, w   = frame.shape[:2]
    y1     = int(h * top_frac)
    y2     = int(h * bottom_frac)
    roi    = frame[y1:y2, x_start:x_end]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    avg_r  = float(np.mean(roi_rgb[:, :, 0]))
    avg_g  = float(np.mean(roi_rgb[:, :, 1]))
    avg_b  = float(np.mean(roi_rgb[:, :, 2]))
    return avg_r, avg_g, avg_b

# Add this function to your existing script
def signal_dashboard():
    """Write a small JSON signal file so Streamlit knows to refresh."""
    import json
    signal = {
        "last_updated"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_experiments" : len(log_df) if 'log_df' in dir() else 0
    }
    with open(f"{OUTPUT_FOLDER}/dashboard_signal.json", "w") as f:
        json.dump(signal, f)

# ----------------------------------------------------------------
# DRAW OVERLAY
# ----------------------------------------------------------------
def draw_overlay(frame, target_rgb, machine_rgb, pump_ratios,
                 attempt, tries, last_score, last_dist):
    h, w  = frame.shape[:2]
    mid   = w // 2
    inner = int(w * ROI_INNER_FRAC)
    y1    = int(h * ROI_TOP_FRAC)
    y2    = int(h * ROI_BOTTOM_FRAC)

    # ROI boxes
    cv2.line(frame, (mid, 0), (mid, h), (255, 255, 255), 2)
    cv2.rectangle(frame, (0, y1),           (mid - inner, y2), (0, 255, 0),    2)
    cv2.rectangle(frame, (mid + inner, y1), (w, y2),           (0, 200, 255),  2)

    # Section labels
    cv2.putText(frame, "TARGET (Left)",   (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "MACHINE (Right)", (mid + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # RGB readouts
    if target_rgb:
        r, g, b = target_rgb
        cv2.putText(frame, f"RGB: ({r:.0f}, {g:.0f}, {b:.0f})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if machine_rgb:
        r, g, b = machine_rgb
        cv2.putText(frame, f"RGB: ({r:.0f}, {g:.0f}, {b:.0f})",
                    (mid + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # Pump ratios
    if pump_ratios:
        a, b_p, c = pump_ratios
        cv2.putText(frame, f"Pumps → A:{a:.2f}  B:{b_p:.2f}  C:{c:.2f}",
                    (10, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    # Tries remaining
    tries_left = MAX_TRIES - tries
    tries_col  = (0, 255, 0) if tries_left > 1 else (0, 100, 255)
    cv2.putText(frame, f"Tries left for this colour: {tries_left}/{MAX_TRIES}",
                (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tries_col, 2)

    # Score display
    if last_score is not None:
        s_col = score_colour(last_score)
        cv2.putText(frame,
                    f"Last Score: {last_score:.1f}%  |  Delta: {last_dist:.1f}",
                    (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, s_col, 2)

    # Total attempts
    cv2.putText(frame, f"Total experiments: {attempt}",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Controls
    cv2.putText(frame,
                "[T] Target  [M] Mix  [C] Capture  [N] New Colour  [P] Plot  [Q] Quit",
                (10, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return frame


# ----------------------------------------------------------------
# EXCEL EXPORT — with colour-coded cells
# ----------------------------------------------------------------
def export_to_excel(log_df):
    """
    Writes experiment log to a formatted Excel file.
    Colour-codes the RGB sample cells and score cells.
    """
    # Write raw data first
    log_df.to_excel(EXCEL_FILE, index=False, sheet_name="Experiment Log")

    wb = load_workbook(EXCEL_FILE)
    ws = wb["Experiment Log"]

    # ── Header styling ──────────────────────────────────────────
    header_fill   = PatternFill("solid", fgColor="2E4057")
    header_font   = Font(color="FFFFFF", bold=True)
    header_align  = Alignment(horizontal="center", vertical="center")
    thin_border   = Border(
        left   = Side(style="thin"),
        right  = Side(style="thin"),
        top    = Side(style="thin"),
        bottom = Side(style="thin"),
    )

    for col_idx, cell in enumerate(ws[1], start=1):
        cell.fill      = header_fill
        cell.font      = header_font
        cell.alignment = header_align
        cell.border    = thin_border
        ws.column_dimensions[get_column_letter(col_idx)].width = 16

    # ── Column index map ────────────────────────────────────────
    headers = [cell.value for cell in ws[1]]
    col     = {name: idx + 1 for idx, name in enumerate(headers)}

    # ── Row styling ─────────────────────────────────────────────
    for row_idx, row in enumerate(ws.iter_rows(min_row=2), start=2):

        # Alternate row background
        row_fill = PatternFill("solid",
                               fgColor="F0F4F8" if row_idx % 2 == 0 else "FFFFFF")

        for cell in row:
            cell.fill      = row_fill
            cell.alignment = Alignment(horizontal="center")
            cell.border    = thin_border

        # ── Target RGB colour swatch ────────────────────────────
        try:
            tr = int(ws.cell(row_idx, col["target_r"]).value or 0)
            tg = int(ws.cell(row_idx, col["target_g"]).value or 0)
            tb = int(ws.cell(row_idx, col["target_b"]).value or 0)
            hex_t = f"{tr:02X}{tg:02X}{tb:02X}"
            swatch_cell_t = ws.cell(row_idx, col.get("target_swatch",
                                    col["target_b"] + 1))
            # Write swatch in a dedicated notes column
        except Exception:
            pass

        # ── Score colour coding ──────────────────────────────────
        try:
            score_val = float(ws.cell(row_idx,
                              col["match_score_pct"]).value or 0)
            if score_val >= 90:
                score_hex = "00C851"    # Green
            elif score_val >= 70:
                score_hex = "FFD700"    # Yellow
            elif score_val >= 50:
                score_hex = "FF8800"    # Orange
            else:
                score_hex = "FF4444"    # Red
            ws.cell(row_idx, col["match_score_pct"]).fill = \
                PatternFill("solid", fgColor=score_hex)
            ws.cell(row_idx, col["match_score_pct"]).font = \
                Font(bold=True,
                     color="FFFFFF" if score_val < 70 else "000000")
        except Exception:
            pass

        # ── Target RGB cell background ───────────────────────────
        try:
            tr = int(ws.cell(row_idx, col["target_r"]).value or 0)
            tg = int(ws.cell(row_idx, col["target_g"]).value or 0)
            tb = int(ws.cell(row_idx, col["target_b"]).value or 0)
            hex_t = f"{tr:02X}{tg:02X}{tb:02X}"
            for c_name in ["target_r", "target_g", "target_b"]:
                ws.cell(row_idx, col[c_name]).fill = \
                    PatternFill("solid", fgColor=hex_t)
                # Make text readable on dark backgrounds
                brightness = (tr * 0.299 + tg * 0.587 + tb * 0.114)
                ws.cell(row_idx, col[c_name]).font = \
                    Font(color="000000" if brightness > 128 else "FFFFFF")
        except Exception:
            pass

        # ── Machine RGB cell background ──────────────────────────
        try:
            mr = int(ws.cell(row_idx, col["machine_r"]).value or 0)
            mg = int(ws.cell(row_idx, col["machine_g"]).value or 0)
            mb = int(ws.cell(row_idx, col["machine_b"]).value or 0)
            hex_m = f"{mr:02X}{mg:02X}{mb:02X}"
            for c_name in ["machine_r", "machine_g", "machine_b"]:
                ws.cell(row_idx, col[c_name]).fill = \
                    PatternFill("solid", fgColor=hex_m)
                brightness = (mr * 0.299 + mg * 0.587 + mb * 0.114)
                ws.cell(row_idx, col[c_name]).font = \
                    Font(color="000000" if brightness > 128 else "FFFFFF")
        except Exception:
            pass

    # ── Freeze top row ───────────────────────────────────────────
    ws.freeze_panes = "A2"

    # ── Auto-fit row height ──────────────────────────────────────
    for row in ws.iter_rows():
        ws.row_dimensions[row[0].row].height = 20

    wb.save(EXCEL_FILE)
    print(f"  📊 Excel log updated → {os.path.abspath(EXCEL_FILE)}")


# ----------------------------------------------------------------
# AI MODEL
# ----------------------------------------------------------------
def load_or_create_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        print("  📂 Loading existing AI model...")
        model  = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        print("  ✅ Model loaded.")
        return model, scaler, True
    else:
        print("  🆕 Creating new AI model...")
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=3000,
            warm_start=True,
            random_state=42
        )
        scaler = StandardScaler()
        print("  ✅ New model ready.")
        return model, scaler, False


def predict_ratios(model, scaler, target_rgb, is_trained):
    if not is_trained:
        print("  ⚠️  No training data yet — using equal ratios as initial guess.")
        return (1/3, 1/3, 1/3)
    X        = np.array([[target_rgb[0], target_rgb[1], target_rgb[2]]])
    X_scaled = scaler.transform(X)
    ratios   = model.predict(X_scaled)[0]
    ratios   = np.clip(ratios, 0.01, 0.98)
    ratios   = ratios / ratios.sum()
    return tuple(ratios)


def update_model(model, scaler, log_df, is_trained):
    if len(log_df) < 2:
        print("  ⚠️  Need at least 2 data points to train. Keep going!")
        return model, scaler, is_trained

    X        = log_df[["target_r", "target_g", "target_b"]].values
    y        = log_df[["pump_a", "pump_b", "pump_c"]].values
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    model.fit(X_scaled, y)
    is_trained = True
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"  🧠 Model updated on {len(log_df)} total observations.")
    return model, scaler, is_trained


# ----------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------
def log_experiment(log_df, colour_id, try_num, target_rgb,
                   machine_rgb, pump_ratios, score, dist):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = {
        "timestamp"       : timestamp,
        "colour_id"       : colour_id,
        "try_number"      : try_num,
        "target_r"        : round(target_rgb[0], 1),
        "target_g"        : round(target_rgb[1], 1),
        "target_b"        : round(target_rgb[2], 1),
        "machine_r"       : round(machine_rgb[0], 1),
        "machine_g"       : round(machine_rgb[1], 1),
        "machine_b"       : round(machine_rgb[2], 1),
        "pump_a"          : round(pump_ratios[0], 4),
        "pump_b"          : round(pump_ratios[1], 4),
        "pump_c"          : round(pump_ratios[2], 4),
        "rgb_distance"    : round(dist, 2),
        "match_score_pct" : round(score, 2),
    }
    log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
    log_df.to_csv(LOG_FILE, index=False)
    return log_df


# ----------------------------------------------------------------
# PLOT PROGRESS
# ----------------------------------------------------------------
def plot_progress(log_df):
    if len(log_df) < 2:
        print("  ⚠️  Not enough data to plot yet.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("AI Colour Mixer — Learning Progress", fontsize=14, fontweight='bold')

    # Plot 1: Match score over all experiments
    axes[0].plot(log_df["match_score_pct"].values, marker='o',
                 color='mediumseagreen', linewidth=2)
    axes[0].axhline(y=90, color='green',  linestyle='--',
                    alpha=0.5, label='90% target')
    axes[0].set_title("Match Score Over All Experiments")
    axes[0].set_xlabel("Experiment #")
    axes[0].set_ylabel("Match Score (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Pump ratios over all experiments
    axes[1].plot(log_df["pump_a"].values, marker='o',
                 label="Pump A", color='tomato',   linewidth=2)
    axes[1].plot(log_df["pump_b"].values, marker='s',
                 label="Pump B", color='steelblue', linewidth=2)
    axes[1].plot(log_df["pump_c"].values, marker='^',
                 label="Pump C", color='goldenrod', linewidth=2)
    axes[1].set_title("Pump Ratios Over All Experiments")
    axes[1].set_xlabel("Experiment #")
    axes[1].set_ylabel("Ratio (sums to 1.0)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Best score per colour
    best_per_colour = log_df.groupby("colour_id")["match_score_pct"].max()
    axes[2].bar(best_per_colour.index.astype(str),
                best_per_colour.values,
                color='mediumpurple', edgecolor='white')
    axes[2].axhline(y=90, color='green', linestyle='--',
                    alpha=0.5, label='90% target')
    axes[2].set_title("Best Score Per Target Colour")
    axes[2].set_xlabel("Colour ID")
    axes[2].set_ylabel("Best Match Score (%)")
    axes[2].set_ylim(0, 105)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = f"{OUTPUT_FOLDER}/progress_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"  📊 Plot saved → {plot_path}")


# ================================================================
# MAIN
# ================================================================
def main():
    setup_folders()

    print("=" * 65)
    print("  🧪 AI Colour Matching System — v2")
    print("=" * 65)
    print(f"  Max tries per colour : {MAX_TRIES}")
    print(f"  Data folder          : {os.path.abspath(OUTPUT_FOLDER)}")
    print()
    print("  Controls:")
    print("    [T] → Capture TARGET colour from LEFT vial")
    print("    [M] → Get AI pump MIX recommendation")
    print("    [C] → Capture MACHINE colour + log result")
    print("    [N] → Move to NEXT colour (new target)")
    print("    [P] → Plot learning progress")
    print("    [Q] → Quit")
    print("=" * 65)

    # Load model and log
    model, scaler, is_trained = load_or_create_model()

    if os.path.exists(LOG_FILE):
        log_df   = pd.read_csv(LOG_FILE)
        colour_id = int(log_df["colour_id"].max()) + 1 \
                    if len(log_df) > 0 else 1
        print(f"  📂 Loaded {len(log_df)} previous experiment(s). "
              f"Starting at Colour ID {colour_id}.")
    else:
        log_df    = pd.DataFrame(columns=[
            "timestamp", "colour_id", "try_number",
            "target_r", "target_g", "target_b",
            "machine_r", "machine_g", "machine_b",
            "pump_a", "pump_b", "pump_c",
            "rgb_distance", "match_score_pct"
        ])
        colour_id = 1

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # State
    target_rgb   = None
    machine_rgb  = None
    pump_ratios  = None
    tries        = 0          # Tries for current colour
    total_exp    = len(log_df)
    last_score   = None
    last_dist    = None

    print("\n✅ Camera open. Preview active.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w  = frame.shape[:2]
            mid   = w // 2
            inner = int(w * ROI_INNER_FRAC)

            display = frame.copy()
            display = draw_overlay(display, target_rgb, machine_rgb,
                                   pump_ratios, total_exp, tries,
                                   last_score, last_dist)
            cv2.imshow("AI Colour Mixer v2", display)

            key = cv2.waitKey(1) & 0xFF

            # ── [T] Capture target ────────────────────────────────
            if key == ord('t') or key == ord('T'):
                if tries >= MAX_TRIES:
                    print(f"\n⚠️  Max tries ({MAX_TRIES}) reached for this colour.")
                    print("    Press [N] to move to a new colour.")
                else:
                    r, g, b = extract_avg_rgb(
                        frame, 0, mid - inner,
                        ROI_TOP_FRAC, ROI_BOTTOM_FRAC
                    )
                    target_rgb  = (r, g, b)
                    machine_rgb = None
                    pump_ratios = None
                    last_score  = None
                    last_dist   = None
                    print(f"\n🎯 [Colour {colour_id}] Target set → "
                          f"R:{r:.1f}  G:{g:.1f}  B:{b:.1f}")
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(
                        f"{OUTPUT_FOLDER}/images/"
                        f"colour{colour_id}_target_{ts}.jpg", frame
                    )

            # ── [M] Get mix recommendation ────────────────────────
            elif key == ord('m') or key == ord('M'):
                if target_rgb is None:
                    print("\n⚠️  Capture a target colour first [T].")
                elif tries >= MAX_TRIES:
                    print(f"\n⚠️  Max tries reached. Press [N] for next colour.")
                else:
                    pump_ratios = predict_ratios(
                        model, scaler, target_rgb, is_trained
                    )
                    a, b_p, c = pump_ratios
                    print(f"\n💉 [Colour {colour_id} | Try {tries + 1}/{MAX_TRIES}]"
                          f" Pump ratios:")
                    print(f"   Pump A : {a:.4f}  ({a * 100:.1f}%)")
                    print(f"   Pump B : {b_p:.4f}  ({b_p * 100:.1f}%)")
                    print(f"   Pump C : {c:.4f}  ({c * 100:.1f}%)")
                    print(f"   → Run pumps, then press [C] to capture result.")

            # ── [C] Capture machine result ────────────────────────
            elif key == ord('c') or key == ord('C'):
                if target_rgb is None:
                    print("\n⚠️  Set a target first [T].")
                elif pump_ratios is None:
                    print("\n⚠️  Get a mix recommendation first [M].")
                elif tries >= MAX_TRIES:
                    print(f"\n⚠️  Max tries reached. Press [N] for next colour.")
                else:
                    r, g, b = extract_avg_rgb(
                        frame, mid + inner, w,
                        ROI_TOP_FRAC, ROI_BOTTOM_FRAC
                    )
                    machine_rgb = (r, g, b)
                    tries      += 1
                    total_exp  += 1

                    score, dist = rgb_match_score(target_rgb, machine_rgb)
                    last_score  = score
                    last_dist   = dist

                    s_col = "🟢" if score >= 90 else \
                            "🟡" if score >= 70 else \
                            "🟠" if score >= 50 else "🔴"

                    print(f"\n🤖 [Colour {colour_id} | Try {tries}/{MAX_TRIES}]"
                          f" Machine → R:{r:.1f}  G:{g:.1f}  B:{b:.1f}")
                    print(f"   {s_col} Match Score : {score:.2f}%")
                    print(f"   📏 RGB Distance : {dist:.2f}")

                    # Save image
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(
                        f"{OUTPUT_FOLDER}/images/"
                        f"colour{colour_id}_try{tries}_{ts}.jpg", frame
                    )

                    # Log experiment
                    log_df = log_experiment(
                        log_df, colour_id, tries,
                        target_rgb, machine_rgb,
                        pump_ratios, score, dist
                    )

                    # Export to Excel
                    export_to_excel(log_df)
                    signal_dashboard()   # ← add this line

                    # Retrain model
                    model, scaler, is_trained = update_model(
                        model, scaler, log_df, is_trained
                    )

                    # Check if max tries reached
                    if tries >= MAX_TRIES:
                        best = log_df[
                            log_df["colour_id"] == colour_id
                        ]["match_score_pct"].max()
                        print(f"\n{'=' * 50}")
                        print(f"  ✅ Colour {colour_id} complete!")
                        print(f"  🏆 Best score for this colour: {best:.2f}%")
                        print(f"  Press [N] to start a new target colour.")
                        print(f"{'=' * 50}")
                    else:
                        remaining = MAX_TRIES - tries
                        print(f"   ℹ️  {remaining} try/tries remaining "
                              f"for this colour.")

            # ── [N] New colour ────────────────────────────────────
            elif key == ord('n') or key == ord('N'):
                if target_rgb is None:
                    print("\n⚠️  No colour was set. Use [T] to set a target first.")
                else:
                    colour_id  += 1
                    tries       = 0
                    target_rgb  = None
                    machine_rgb = None
                    pump_ratios = None
                    last_score  = None
                    last_dist   = None
                    print(f"\n🔄 Moving to Colour ID {colour_id}. "
                          f"Press [T] to set new target.")

            # ── [P] Plot ──────────────────────────────────────────
            elif key == ord('p') or key == ord('P'):
                plot_progress(log_df)

            # ── [Q] Quit ──────────────────────────────────────────
            elif key == ord('q') or key == ord('Q'):
                print("\n👋 Exiting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if len(log_df) > 0:
            export_to_excel(log_df)
            signal_dashboard()   # ← add this line
        print(f"\n✅ Session complete. {total_exp} total experiments logged.")
        print(f"   Excel file → {os.path.abspath(EXCEL_FILE)}")


if __name__ == "__main__":
    main()