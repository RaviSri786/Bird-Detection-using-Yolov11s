from PIL import Image, ImageDraw
from IPython.display import display
import os
import random
from ultralytics import YOLO
import cv2

# Path to the images folder (update the dataset path accordingly)
images_folder = '/content/enter your dataset name here/train/images'

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Choose four random images from the list
random_images = random.sample(image_files, 4)

# Use a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Loop through the selected images and show prediction results
for image_file in random_images:
    # Construct the full path to the image
    image_path = os.path.join(images_folder, image_file)
    
    # Use the model to detect objects (birds in this case)
    result_predict = model.predict(source=image_path, imgsz=(416))
    
    # Display the original image
    original_image = Image.open(image_path)
    display(original_image)
    
    # Display the prediction results
    for result in result_predict:
        plot = result.plot()
        plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
        display(Image.fromarray(plot))
