# Project Setup on Ubuntu Server with CUDA

This guide details the steps to set up the vehicle detection and counting project on a fresh Ubuntu server with CUDA pre-configured. You will need terminal access to the server.

## 1. System Update and Basic Tools

First, ensure your system is up-to-date and install some essential tools:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3-venv python3-pip unzip
```

## 2. Clone the Repository

Clone the project repository from GitHub (replace `<your-repo-url>` with the actual URL):

```bash
git clone <your-repo-url> vehicle-detect-and-count
cd vehicle-detect-and-count
```

## 3. Download and Prepare Dataset

You will download the dataset from the provided Google Drive link.

*   **Install `gdown`:**
    ```bash
    pip install gdown
    ```

*   **Download the dataset:**
    The provided link is to a folder: `https://drive.google.com/drive/folders/1bn9dFIO-Ua3DyMTLg6aC42CVJGesvctS`
    You will need to download the contents of this folder. If it's a single zip file within that folder, find its direct download ID. For this example, let's assume there's a `dataset.zip` file in that Google Drive folder. You'll need to find its specific ID or shareable link that `gdown` can use.

    If you have a direct file ID, use:
    ```bash
    # Replace <file-id> with the actual ID of the dataset zip file from Google Drive
    gdown --id <file-id> -O dataset.zip 
    ```
    Alternatively, if you make the folder downloadable as a zip, `gdown` might be able to download it if you have the folder ID. This can be tricky with folders. The most reliable way is to get a direct download link for a zip file of the dataset.

*   **Create dataset directory structure and extract:**
    The project expects a specific directory structure, typically `datasets/yolo/...`. The `dataset.yaml` refers to `../datasets/yolo`.
    Assuming your `dataset.zip` contains the `yolo` folder (with `images` and `labels` subdirectories):

    ```bash
    # Create the target directory if it doesn't exist, from the workspace root (vehicle-detect-and-count)
    mkdir -p datasets
    unzip dataset.zip -d datasets/ 
    # Ensure the final structure is vehicle-detect-and-count/datasets/yolo/images/... and vehicle-detect-and-count/datasets/yolo/labels/...
    # You might need to move files/folders around after unzipping depending on the zip file's structure.
    # For example, if dataset.zip extracts to datasets/dataset_folder_name/yolo/...
    # mv datasets/dataset_folder_name/yolo datasets/yolo
    # rm -rf datasets/dataset_folder_name
    ```
    Verify that `vehicle-detect-and-count/datasets/yolo` exists and contains your `images` and `labels` folders. The `traffic-pipeline/dataset.yaml` specifies `path: ../datasets/yolo`, so the `yolo` directory should be directly inside `vehicle-detect-and-count/datasets/`.

## 4. Set Up Python Virtual Environment

Create and activate a virtual environment within the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
```
You should see `(.venv)` at the beginning of your terminal prompt.

## 5. Install Python Dependencies

Install the required Python libraries. Ensure you have CUDA installed and PyTorch can see it.
The server template should have CUDA. PyTorch installation commands below will pick the correct CUDA version.

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (visit https://pytorch.org/ for the latest command for your CUDA version)
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Example for CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other core dependencies
pip install ultralytics opencv-python pandas PyYAML shapely tqdm
pip install cjm-byte-track
pip install matplotlib # Often useful for vision tasks and ultralytics plots
pip install scipy # Often a dependency for scientific computing packages

# For COCO evaluation (used by evaluate-detectors.py)
pip install pycocotools
```

## 6. Obtain Torchvision Utility Files (for Faster R-CNN)

The `train-fasterrcnn.py` and `evaluate-detectors.py` scripts use utility files from the `torchvision/references/detection/` repository. You need to download `utils.py`, `engine.py`, and `transforms.py` into your `traffic-pipeline/` directory.

```bash
cd traffic-pipeline

# Download utils.py
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py

# Download engine.py
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py

# Download transforms.py
wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py

cd .. 
# You are now back in the project root: vehicle-detect-and-count/
```
**Note:** Ensure these files are placed directly inside the `traffic-pipeline/` directory, alongside your training and evaluation scripts.

## 7. Project Execution Workflow

Refer to `PROCESS.md` for a detailed explanation of each step. The general workflow is:

*   **Prepare Data (Convert YOLO to COCO for Faster R-CNN):**
    Make sure `traffic-pipeline/dataset.yaml` correctly points to your dataset path (`../datasets/yolo`) and has the correct class names.
    ```bash
    python traffic-pipeline/prepare-data.py --yaml_path traffic-pipeline/dataset.yaml --output_coco_path traffic-pipeline/datasets/yolo_coco_from_yolo.json
    ```

*   **Train Models:**
    *   **YOLOv8:**
        ```bash
        # Adjust --device if needed, e.g., --device 0 for the first GPU
        python traffic-pipeline/train-yolov8.py --data traffic-pipeline/dataset.yaml --model yolov8m.pt --epochs 50 --batch 32 --device cuda 
        ```
    *   **Faster R-CNN:**
        The `num_classes` argument should be your number of object classes + 1 (for the background). Based on `dataset.yaml`, you have 4 classes, so `num_classes` will be 5.
        ```bash
        # Adjust --device if needed
        python traffic-pipeline/train-fasterrcnn.py --coco_json traffic-pipeline/datasets/yolo_coco_from_yolo.json --num_classes 5 --epochs 50 --batch_size 4
        ```
        Training outputs will be saved in `traffic-pipeline/runs/yolov8/` and `traffic-pipeline/runs/fasterrcnn/`.

*   **Evaluate Models:**
    Use the paths to your best trained weights.
    ```bash
    python traffic-pipeline/evaluate-detectors.py \
        --yolo_weights traffic-pipeline/runs/yolov8/weights/best.pt \
        --dataset_yaml traffic-pipeline/dataset.yaml \
        --fasterrcnn_weights traffic-pipeline/runs/fasterrcnn/model_e49.pth \
        --coco_json_val traffic-pipeline/datasets/yolo_coco_from_yolo.json \
        --num_classes_frcnn 5 \
        --device cuda # or mps, or cpu
    ```
    Metrics will be saved to `traffic-pipeline/reports/detection_metrics.csv`.

*   **Run Real-Time Application:**
    Use your trained YOLOv8 weights and a video source (e.g., a video file or webcam ID '0').
    ```bash
    # Using a video file:
    python traffic-pipeline/realtime-app.py --weights traffic-pipeline/runs/yolov8/weights/best.pt --source path/to/your/video.mp4 --device cuda

    # Using webcam (if available on the server, typically not for remote servers):
    # python traffic-pipeline/realtime-app.py --weights traffic-pipeline/runs/yolov8/weights/best.pt --source 0 --device cuda
    ```
    Counts will be logged to a CSV file in `traffic-pipeline/logs/`.

## 8. Deactivate Virtual Environment (When Done)

```bash
deactivate
```

This setup guide should help you get the project running on your rented GPU server. Remember to replace placeholders like `<your-repo-url>` and `<file-id>` with actual values. 