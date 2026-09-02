"""
Phase 4: Vulnerable pedestrian detection.

Concept:
- "person" is COCO class 0 -- YOLOv8n already knows this, no fine-tuning needed.
- The PS wants "vulnerable pedestrian situations" specifically (e.g. children
  crossing), not just "a pedestrian existed somewhere in frame". True age
  classification needs a specialized model we don't have time to build for
  MVP -- so we simplify: flag a pedestrian as an ALERT only when their
  position falls inside a defined "danger zone" (near/on the road ahead),
  vs. just a normal detection when they're on a distant sidewalk.
- The danger zone is a polygon in pixel coordinates. For MVP this is
  manually defined (a rough trapezoid representing "the road ahead of the
  vehicle"). Later this could be made dynamic (e.g. lane detection), but a
  fixed zone is a reasonable simplification for a dashcam-style fixed
  camera angle.
"""

from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np

PERSON_CLASS_ID = 0  # COCO class id for "person"


def point_in_zone(point, zone_polygon):
    """Check if a point (x, y) is inside the danger-zone polygon."""
    return cv2.pointPolygonTest(zone_polygon, point, False) >= 0


def get_default_zone(frame_width, frame_height):
    """
    Defines a trapezoid roughly covering 'the road ahead' in a typical
    dashcam frame -- wider near the bottom (close to vehicle), narrower
    near the middle of the frame (further down the road).
    Tune these fractions against YOUR footage -- this is a starting point.
    """
    zone = np.array([
        [int(frame_width * 0.15), frame_height],        # bottom-left
        [int(frame_width * 0.85), frame_height],        # bottom-right
        [int(frame_width * 0.65), int(frame_height * 0.55)],  # upper-right
        [int(frame_width * 0.35), int(frame_height * 0.55)],  # upper-left
    ], dtype=np.int32)
    return zone


def process_video(video_path: str, output_path: str = "pedestrian_output.mp4", conf_threshold: float = 0.4):
    model = YOLO("yolov8n.pt")
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at '{video_path}' — check the path.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    zone = get_default_zone(w, h)
    alert_events = []  # collects {frame, tracker_id} for each alert -- this is what would become the JSON event in phase later
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        mask = [
            (cls_id == PERSON_CLASS_ID) and (conf >= conf_threshold)
            for cls_id, conf in zip(detections.class_id, detections.confidence)
        ]
        detections = detections[mask]
        detections = tracker.update_with_detections(detections)

        labels = []
        for i, (tid, box) in enumerate(zip(detections.tracker_id, detections.xyxy)):
            x1, y1, x2, y2 = box
            foot_point = (int((x1 + x2) / 2), int(y2))  # bottom-center of box = where the person is standing
            in_danger_zone = point_in_zone(foot_point, zone)
            if in_danger_zone:
                alert_events.append({"frame": frame_idx, "tracker_id": int(tid)})
                labels.append(f"ALERT #{tid}")
            else:
                labels.append(f"person #{tid}")

        annotated = frame.copy()
        cv2.polylines(annotated, [zone], isClosed=True, color=(0, 255, 255), thickness=2)  # draw danger zone
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Processed {frame_idx} frames.")
    print(f"Total danger-zone alert events: {len(alert_events)}")
    return alert_events


if __name__ == "__main__":
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample1.mp4"
    process_video(video_path)