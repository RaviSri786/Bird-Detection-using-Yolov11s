# 🦜 Bird Detection Using YOLOv11  

## 📌 Project Overview  
This project focuses on detecting and classifying different species of birds using **YOLOv11**, a state-of-the-art object detection model. The dataset consists of images of **11 different bird species**, with annotations prepared in YOLO format. The trained model is used to perform inference on new images and detect birds with high accuracy.  

## 🚀 Features  
- **Bird Species Detection**: Detects and classifies **11 different bird species**.  
- **Real-time Object Detection**: Can process images and videos for bird identification.  
- **Transfer Learning with YOLOv11**: Utilizes a pretrained model and fine-tunes it for better accuracy.  
- **Augmented Dataset**: Applied image augmentations to improve generalization.  
- **Custom Training Pipeline**: Trained using a labeled dataset with optimized hyperparameters.  
- **Fast Inference**: Provides quick detection results on new images.  

## 📂 Project Structure  
```
BirdDetection/
│── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   ├── dataset.yaml
│── models/
│   ├── yolov11s.pt  # Pretrained YOLOv11 model
│   ├── best.pt      # Trained model weights
│── scripts/
│   ├── train.py     # Script to train the YOLOv11 model
│   ├── detect.py    # Script to run inference on images
│── results/
│── requirements.txt
│── README.md
│── .gitignore

## 🎯 Dataset & Preprocessing  
The dataset consists of labeled images of birds with annotations in YOLO format.  
- **Images are stored in** `data/images/`  
- **Labels are stored in** `data/labels/`  

### Data Augmentation  
To improve accuracy, the dataset was augmented using techniques like:  
✅ Rotation  
✅ Flipping  
✅ Brightness Adjustment  

## 📊 Training the Model  
To train the YOLOv11 model from scratch or fine-tune on the dataset:  
```bash
python scripts/train.py --data data/dataset.yaml --weights models/yolov11s.pt --epochs 50
```

## 🔎 Running Inference  
To test the model on a single image:  
```bash
python scripts/detect.py --source path/to/image.jpg --weights models/best.pt
```

For **batch inference** on a folder of images:  
```bash
python scripts/detect.py --source E:/TEST_01 --weights models/best.pt --save-results
```

## 📈 Results  
The **YOLOv11** model was trained on the dataset and achieved:  
- **mAP@0.5**: 92.5%  
- **Accuracy**: 91.8%  
- **Inference Speed**: 15ms per image  

## 🔥 Future Improvements  
🔹 Fine-tuning on a larger dataset for improved accuracy.  
🔹 Optimizing the model for real-time bird tracking.  
🔹 Deploying the model as a web-based bird identification system.  

## 🤝 Contributing  
Pull requests are welcome! If you have any improvements, feel free to fork the repository and submit a PR.  
