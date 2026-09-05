from ultralytics import YOLO
import supervision as sv
import cv2

from event_client import send_event
from gps_matcher import simulate_route

VEHICLE_CLASS_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

ROUTE_START = (22.8046, 86.2029)
ROUTE_END = (22.8060, 86.2050)


def process_video(video_path: str, output_path: str = "output.mp4", conf_threshold: float = 0.4,
                   bus_id: str = "bus_01", send_events: bool = True):
    model = YOLO("yolov8n.pt")  
    tracker = sv.ByteTrack() 
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    seen_ids = set()  
    class_counts = {name: 0 for name in VEHICLE_CLASS_IDS.values()}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at '{video_path}' — check the path.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

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

        for tid, cls_id in zip(detections.tracker_id, detections.class_id):
            if tid not in seen_ids:
                seen_ids.add(tid)
                class_counts[VEHICLE_CLASS_IDS[cls_id]] += 1

                if send_events:
                    lat, lon = simulate_route(*ROUTE_START, *ROUTE_END, frame_idx, total_frames)
                    send_event(
                        event_type="vehicle_density",
                        confidence=float(detections.confidence[list(detections.tracker_id).index(tid)]),
                        gps_lat=lat, gps_lon=lon,
                        bus_id=bus_id,
                        extra={"vehicle_type": VEHICLE_CLASS_IDS[cls_id], "tracker_id": int(tid)},
                    )

        labels = [
            f"#{tid} {VEHICLE_CLASS_IDS[cls_id]} {conf:.2f}"
            for tid, cls_id, conf in zip(detections.tracker_id, detections.class_id, detections.confidence)
        ]
        annotated = box_annotator.annotate(frame.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels)
        cv2.putText(annotated, f"Distinct vehicles so far: {len(seen_ids)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Processed {frame_idx} frames.")
    print(f"Total distinct vehicles: {len(seen_ids)}")
    print(f"By class: {class_counts}")
    return {"total": len(seen_ids), "by_class": class_counts}


if __name__ == "__main__":
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample1.mp4"
    process_video(video_path)