from ultralytics import YOLO


def train_pothole_model(data_yaml: str = "data/pothole_dataset/data.yaml",
                         epochs: int = 50,
                         imgsz: int = 640):
    
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,      
        imgsz=imgsz,
        batch=16,          
        patience=10,       
        project="models",
        name="pothole_detector",
    )
   
    return results


def test_on_image(weights_path: str, image_path: str):
    """Quick sanity check: run the fine-tuned model on a single image."""
    model = YOLO(weights_path)
    results = model(image_path)
    results[0].show()      
    results[0].save("pothole_test_output.jpg")
    return results


if __name__ == "__main__":
    train_pothole_model()