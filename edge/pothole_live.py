"""
Phase 3 (live pipeline version): Pothole detection, frame-by-frame,
sending an event to the backend for each detection.

This replaces test_pothole_video.py's bulk model(video) call with the
same frame-by-frame loop pattern used in vehicle_density.py,
pedestrian_zone.py, and incident_detection.py -- needed because sending
one event per detection requires processing frame-by-frame, not handing
the whole video to the model in one call.
"""

from ultralytics import YOLO
import cv2

from event_client import send_event
from gps_matcher import simulate_route

WEIGHTS_PATH = "models/pothole_detector/weights/best.pt"

# Placeholder route coordinates -- replace with real GPS log matching
# (see gps_matcher.match_gps_to_frame) once real GPS data is available.
ROUTE_START = (22.8046, 86.2029)
ROUTE_END = (22.8060, 86.2050)


def process_video(video_path: str, output_path: str = "pothole_output.mp4", conf_threshold: float = 0.4,
                   bus_id: str = "bus_01", send_events: bool = True):
    model = YOLO(WEIGHTS_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at '{video_path}' — check the path.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    total_detections = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, conf=conf_threshold)[0]

        for box in results.boxes:
            confidence = float(box.conf[0])
            total_detections += 1

            if send_events:
                lat, lon = simulate_route(*ROUTE_START, *ROUTE_END, frame_idx, total_frames)
                send_event(
                    event_type="pothole",
                    confidence=confidence,
                    gps_lat=lat, gps_lon=lon,
                    bus_id=bus_id,
                )

        annotated = results.plot()  # Ultralytics' built-in box-drawing for quick visual output
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Processed {frame_idx} frames.")
    print(f"Total pothole detections sent: {total_detections}")
    return total_detections


if __name__ == "__main__":
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample1.mp4"
    process_video(video_path)