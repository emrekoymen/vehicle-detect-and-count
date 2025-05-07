# Real-Time Object Detection and Counting of Vehicles in Urban Traffic Scenes

## Objective
Develop a real-time object detection and counting system to identify and track different vehicle types in urban traffic footage using deep learning. You will integrate object detection with tracking and counting logic to assess traffic flow dynamically from video streams.

## Dataset
Use an annotated public dataset like UA-DETRAC or Open Images Dataset – Vehicles subset, or any open-source dataset, which include classes such as cars, buses, trucks, and motorcycles with bounding boxes.

## Project Tasks
- Explore the dataset, understand the structure, labeling conventions, and class diversity.  
- Preprocess the images for model input: resize, normalize, augment with noise or occlusion to simulate real traffic conditions.  
- Train two object detectors: Faster R-CNN and YOLOv8 (or latest YOLO version), using pretrained weights as a starting point.  
- Implement object tracking across frames using DeepSORT or ByteTrack to enable counting.  
- Evaluate detection accuracy using metrics like precision, recall, IoU, and mAP. Assess tracking with metrics like MOTA (Multiple Object Tracking Accuracy).  
- Integrate everything into a real-time video processing application that detects, tracks, and counts vehicles from webcam or traffic camera footage.  
- Train for at least 50 epochs and include validation loss monitoring. Apply regularization or data augmentation if overfitting is detected.

## Deliverables
- A technical report detailing methodology, model architectures, training process, evaluations, and conclusions.  
- A GitHub repository with well-documented code, datasets (or links), and trained models.  
- A short video or live demonstration of the system processing real-time footage and displaying counts and classifications.  