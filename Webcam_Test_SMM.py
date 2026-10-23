# lenovo_webcam.py
# -----------------------------------------------
# Lenovo FHD Webcam - Photo Capture & Save Script
# -----------------------------------------------
# Controls:
#   SPACE  → Take a photo and save it to the output folder
#   Q      → Quit the application
# -----------------------------------------------

import cv2
import os
import time
from datetime import datetime

# ============================================================
# ⚙️  CONFIGURATION — Edit these settings as needed
# ============================================================

CAMERA_INDEX   = 1          # Change to 0, 1, 2 etc. based on find_camera.py output
OUTPUT_FOLDER  = r"C:\Users\smarlow\Box\Sushila Marlow\Python Scripts\colour_mixer_data"   # Folder where photos will be saved
IMAGE_FORMAT   = "jpg"      # Options: 'jpg' or 'png'
JPEG_QUALITY   = 92         # JPEG quality (1-100). Ignored if IMAGE_FORMAT is 'png'
FRAME_WIDTH    = 1920       # Requested capture width  (FHD = 1920)
FRAME_HEIGHT   = 1080       # Requested capture height (FHD = 1080)
WARMUP_SECONDS = 1.0        # Seconds to warm up camera before showing preview

# ============================================================


def find_working_camera(preferred_index: int) -> int:
    """Try the preferred index first; fall back to scanning 0-5."""
    for idx in [preferred_index] + [i for i in range(6) if i != preferred_index]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return idx
    raise RuntimeError(
        "No working camera found. Check USB connection and permissions."
    )


def warm_up_camera(cap: cv2.VideoCapture, seconds: float) -> None:
    """Read and discard frames to allow auto-exposure/focus to settle."""
    print(f"  Warming up camera for {seconds}s...")
    deadline = time.time() + seconds
    while time.time() < deadline:
        cap.read()   # discard frame


def save_photo(frame, folder: str, fmt: str, quality: int) -> str:
    """Save a captured frame to the output folder with a timestamp filename."""
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename  = f"photo_{timestamp}.{fmt}"
    filepath  = os.path.join(folder, filename)

    if fmt.lower() == "jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success = cv2.imwrite(filepath, frame, params)
    else:
        success = cv2.imwrite(filepath, frame)

    if not success:
        raise RuntimeError(f"Failed to save image to: {filepath}")

    return filepath


def main():
    print("=" * 55)
    print("  Lenovo FHD Webcam — Photo Capture Tool")
    print("=" * 55)
    print(f"  Output folder : {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"  Format        : {IMAGE_FORMAT.upper()}")
    print(f"  Controls      : [SPACE] Take photo  |  [Q] Quit")
    print("=" * 55)

    # --- Locate camera ---
    print(f"\n🔍 Looking for camera at index {CAMERA_INDEX}...")
    try:
        cam_index = find_working_camera(CAMERA_INDEX)
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        return

    print(f"  ✅ Camera ready at index {cam_index}")

    # --- Open camera ---
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ Could not open camera. Check permissions or USB connection.")
        return

    # --- Set FHD resolution (request; driver may override) ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  📐 Actual resolution: {actual_w}x{actual_h}")

    # --- Warm up ---
    warm_up_camera(cap, WARMUP_SECONDS)

    photo_count = 0
    print("\n✅ Preview window open. Press SPACE to capture, Q to quit.\n")

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("⚠️  Warning: Failed to read frame. Retrying...")
                time.sleep(0.1)
                continue

            # Show live preview
            display = frame.copy()
            cv2.putText(
                display,
                f"Photos taken: {photo_count}  |  SPACE=Capture  Q=Quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Lenovo FHD Webcam — Live Preview", display)

            key = cv2.waitKey(1) & 0xFF

            # --- SPACE: take photo ---
            if key == ord(" "):
                try:
                    filepath = save_photo(frame, OUTPUT_FOLDER, IMAGE_FORMAT, JPEG_QUALITY)
                    photo_count += 1
                    print(f"  📸 Photo {photo_count} saved → {filepath}")
                except RuntimeError as e:
                    print(f"  ❌ Save error: {e}")

            # --- Q: quit ---
            elif key == ord("q") or key == ord("Q"):
                print("\n👋 Exiting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Done. {photo_count} photo(s) saved to: {os.path.abspath(OUTPUT_FOLDER)}")


if __name__ == "__main__":
    main()