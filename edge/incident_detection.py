"""
Phase 5: Incident detection (rash driving proxy) + ANPR.

Concept:
- Reuses the same detect+track loop from vehicle_density.py.
- "Rash driving" proxy: track each vehicle's bounding-box AREA across
  frames. A rapidly growing box area means the vehicle is approaching the
  camera unusually fast (tailgating / sudden acceleration toward us).
  This is a heuristic, not true behavior classification -- disclose this
  simplification when presenting.
- Once a vehicle is flagged, crop its bounding box and run OCR directly on
  it to attempt plate extraction. No dedicated plate-detector model here
  (that would need its own fine-tuning, like the pothole model) -- this is
  a v1 simplification.
"""

from ultralytics import YOLO
import supervision as sv
import cv2
import easyocr

from event_client import send_event
from gps_matcher import simulate_route

VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Placeholder route coordinates -- replace with real GPS log matching
# (see gps_matcher.match_gps_to_frame) once real GPS data is available.
ROUTE_START = (22.8046, 86.2029)
ROUTE_END = (22.8060, 86.2050)

# How fast a box's area can grow between consecutive frames before we
# flag it as "approaching too fast". Tune this against real footage --
# there's no universal correct value, it depends on camera FPS and typical
# following distance.
AREA_GROWTH_THRESHOLD = 1.4  # e.g. 1.4 = area grew 40% in one frame


def process_video(video_path: str, output_path: str = "incident_output.mp4", conf_threshold: float = 0.4,
                   bus_id: str = "bus_01", send_events: bool = True):
    model = YOLO("yolov8n.pt")
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    ocr_reader = easyocr.Reader(["en"], gpu=False)  # set gpu=True on a CUDA machine for speed

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at '{video_path}'.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    previous_areas = {}   # tracker_id -> last frame's box area
    incidents = []         # collected incident events
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        mask = [
            (cls_id in VEHICLE_CLASS_IDS) and (conf >= conf_threshold)
            for cls_id, conf in zip(detections.class_id, detections.confidence)
        ]
        detections = detections[mask]
        detections = tracker.update_with_detections(detections)

        labels = []
        for tid, box, cls_id in zip(detections.tracker_id, detections.xyxy, detections.class_id):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            flagged = False

            if tid in previous_areas and previous_areas[tid] > 0:
                growth = area / previous_areas[tid]
                if growth >= AREA_GROWTH_THRESHOLD:
                    flagged = True

            previous_areas[tid] = area

            if flagged:
                # Crop the vehicle and attempt plate OCR
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                ocr_results = ocr_reader.readtext(crop) if crop.size > 0 else []
                # Pick the OCR result with highest confidence, if any text found
                plate_text, plate_conf = None, 0.0
                if ocr_results:
                    best = max(ocr_results, key=lambda r: r[2])
                    plate_text, plate_conf = best[1], best[2]

                incidents.append({
                    "frame": frame_idx,
                    "tracker_id": int(tid),
                    "vehicle_type": VEHICLE_CLASS_IDS[cls_id],
                    "plate_text": plate_text,
                    "plate_confidence": round(plate_conf, 3),
                })
                labels.append(f"INCIDENT #{tid} {plate_text or '(no plate read)'}")

                if send_events:
                    lat, lon = simulate_route(*ROUTE_START, *ROUTE_END, frame_idx, total_frames)
                    send_event(
                        event_type="incident",
                        confidence=0.9,  # confidence in the "incident" flag itself, not the plate read
                        gps_lat=lat, gps_lon=lon,
                        bus_id=bus_id,
                        extra={
                            "vehicle_type": VEHICLE_CLASS_IDS[cls_id],
                            "tracker_id": int(tid),
                            "plate_number": plate_text,
                            "plate_confidence": round(plate_conf, 3),
                        },
                    )
            else:
                labels.append(f"#{tid} {VEHICLE_CLASS_IDS[cls_id]}")

        annotated = box_annotator.annotate(frame.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Processed {frame_idx} frames.")
    print(f"Total incident events flagged: {len(incidents)}")
    for inc in incidents[:10]:
        print(f"  Frame {inc['frame']}: vehicle #{inc['tracker_id']} ({inc['vehicle_type']}) "
              f"plate={inc['plate_text']} conf={inc['plate_confidence']}")
    return incidents


if __name__ == "__main__":
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample1.mp4"
    process_video(video_path)