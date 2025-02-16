from ultralytics import YOLO

# Load the model for training
model = YOLO('E:/BirdDetection/yolo11s.pt')

# Train the model and save it
model.train(
    data='E:/BirdDetection/data.yaml',  # Path to your dataset config
    epochs=200,  # Number of epochs
    imgsz=640,   # Image size
    batch=16,    # Batch size
    save_period=1,  # Save model every epoch
    project='E:/BirdDetection/weights/bird_detection',  # Project directory
    name='yolo11s_trained_model',  # Name of the model folder
    exist_ok=True  # Overwrite if the directory exists
)

# Save the trained model
model.export(format='pt', weights='E:/BirdDetection/weights/bird_detection/yolo11s_trained_model.pt')

print("Model saved at E:/BirdDetection/weights/bird_detection/yolo11s_trained_model.pt")
