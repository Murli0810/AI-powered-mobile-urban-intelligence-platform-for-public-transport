"""
Phase 3: Fine-tune YOLOv8 to detect potholes.

Concept:
- yolov8n.pt (used in phase 2) was pretrained on COCO -- it knows "car",
  "person", "bus", but has NO concept of "pothole". It's not that its
  confidence is low; the class doesn't exist in its output space at all.
- Fine-tuning takes that same pretrained model (which already knows
  general shapes/edges/textures from COCO) and continues training it on
  OUR labeled pothole images. This is much faster than training from
  scratch, because the model isn't learning "what an edge is" from zero --
  just "this specific shape pattern = pothole".
- Output is a NEW weights file (e.g. runs/detect/train/weights/best.pt)
  which replaces yolov8n.pt specifically for defect detection. You'll end
  up running TWO models in the edge pipeline eventually: one for vehicles
  (original yolov8n.pt) and one for defects (this fine-tuned version).
"""

from ultralytics import YOLO


def train_pothole_model(data_yaml: str = "data/pothole_dataset/data.yaml",
                         epochs: int = 50,
                         imgsz: int = 640):
    # Start from the pretrained nano model -- NOT from scratch
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,      # 50 is a reasonable starting point for a small dataset
        imgsz=imgsz,
        batch=16,           # lower this (e.g. 4 or 8) if you don't have a GPU / run out of memory
        patience=10,        # stop early if validation loss stops improving
        project="models",
        name="pothole_detector",
    )
    # Best weights are saved automatically at:
    # models/pothole_detector/weights/best.pt
    return results


def test_on_image(weights_path: str, image_path: str):
    """Quick sanity check: run the fine-tuned model on a single image."""
    model = YOLO(weights_path)
    results = model(image_path)
    results[0].show()      # opens an annotated preview window
    results[0].save("pothole_test_output.jpg")
    return results


if __name__ == "__main__":
    train_pothole_model()