import os
import glob

def count_images(folder):
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))])


def inspect_labels(labels_folder, n=5):
    
    if not os.path.exists(labels_folder):
        print(f"  Labels folder not found: {labels_folder}")
        return
    label_files = glob.glob(os.path.join(labels_folder, "*.txt"))[:n]
    if not label_files:
        print(f"  No label files found in {labels_folder}")
        return
    for lf in label_files:
        with open(lf) as f:
            lines = f.readlines()
        print(f"  {os.path.basename(lf)}:")
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x, y, w, h = parts
                w, h = float(w), float(h)
                flag = "  <-- covers (almost) the whole image!" if (w > 0.9 and h > 0.9) else ""
                print(f"    class={cls}  w={w:.3f}  h={h:.3f}{flag}")


def check_training_results(run_folder="models/pothole_detector"):
    csv_path = os.path.join(run_folder, "results.csv")
    if not os.path.exists(csv_path):
        print(f"  No results.csv found at {csv_path} -- can't check training curve.")
        return
    with open(csv_path) as f:
        lines = [l.strip() for l in f.readlines()]
    header = lines[0].split(",")
    last_row = lines[-1].split(",")
    print("  Final epoch metrics:")
    for h, v in zip(header, last_row):
        if "mAP" in h or "loss" in h.lower():
            print(f"    {h.strip()}: {v.strip()}")


if __name__ == "__main__":
    print("=== Dataset size ===")
    train_count = count_images("data/pothole_dataset/train/images")
    valid_count = count_images("data/pothole_dataset/valid/images")
    print(f"  Train images: {train_count}")
    print(f"  Valid images: {valid_count}")
    if train_count < 100:
        print("  ⚠ Under 100 training images is very small for YOLO fine-tuning.")

    print("\n=== Sample training labels (checking for full-image boxes) ===")
    inspect_labels("data/pothole_dataset/train/labels")

    print("\n=== Training results ===")
    check_training_results()