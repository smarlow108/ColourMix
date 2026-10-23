import cv2

print("Scanning for available cameras...")
for index in range(6):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  ✅ Camera found at index: {index} | Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
        else:
            print(f"  ⚠️  Index {index} opens but cannot read frames.")
        cap.release()
    else:
        print(f"  ❌ No camera at index {index}")