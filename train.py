from ultralytics import YOLO

# Load the YOLOv8s model
model = YOLO("yolov8s.pt")  # or use yolov8m.pt / yolov8l.pt

# Train the model
results = model.train(
    data="data.yaml",       # path to your data.yaml
    epochs=100,
    imgsz=640,
    batch=16,
    name="trash-v8s",
    project="trash-runs",
    cache=True,
    device=0,               # GPU index (0 = first GPU)
    verbose=True
)

# Optional: Print final metrics
print("Training complete.")
print(f"Best model saved at: {model.ckpt_path}")
