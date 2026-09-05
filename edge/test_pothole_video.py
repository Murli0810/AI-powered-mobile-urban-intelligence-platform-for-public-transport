import sys
import os
from ultralytics import YOLO

WEIGHTS_PATH = "models/pothole_detector/weights/best.pt"


def test_on_video(video_path: str):
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"No weights file at '{WEIGHTS_PATH}' — place best.pt there first.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No video at '{video_path}'.")

    model = YOLO(WEIGHTS_PATH)

    results = model(video_path, save=True, conf=0.4)

    frames_with_detection = sum(1 for r in results if len(r.boxes) > 0)
    total_detections = sum(len(r.boxes) for r in results)

    print(f"\nTotal frames processed: {len(results)}")
    print(f"Frames with at least one pothole detected: {frames_with_detection}")
    print(f"Total pothole detections (across all frames): {total_detections}")
    print(f"Annotated output video saved under: runs/detect/ (check the newest 'predict' folder)")


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/Asish3.mp4"
    test_on_video(video_path)