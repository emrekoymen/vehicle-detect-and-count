from ultralytics import YOLO
import argparse, os
import torch

def main():
    ap = argparse.ArgumentParser()
    # Updated default path for dataset.yaml to be relative to workspace root if script is in traffic-pipeline/
    ap.add_argument("--data", default="traffic-pipeline/dataset.yaml", 
                        help="Path to the dataset.yaml file.")
    ap.add_argument("--model", default="yolov8m.pt", 
                        help="YOLOv8 model to train (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt).")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32, help="Batch size.")
    ap.add_argument("--imgsz", type=int, default=640, help="Image size.")
    # Changed default device to 'mps' as per previous state, but allow user to override.
    # The script will try to use 'mps', then 'cuda', then 'cpu' if not specified or if mps is unavailable.
    ap.add_argument("--device", default=None, 
                        help="Device to use, e.g., 'cpu', 'cuda', '0' or '0,1,2,3' for CUDA devices, or 'mps'. Defaults to MPS if available, then CUDA, then CPU.")
    ap.add_argument("--project_name", default="traffic-pipeline/runs/yolov8",
                        help="Directory to save training runs.")
    ap.add_argument("--name", default=None, help="Subdirectory name for this run (e.g., 'exp'). Defaults to auto-generated.")
    args = ap.parse_args()

    # Determine device
    if args.device:
        device_to_use = args.device
    elif torch.cuda.is_available():
        device_to_use = 'cuda' # This will use all available CUDA devices by default
                               # Or specify e.g., '0' for the first CUDA device
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built(): # Check if MPS is available and built
        device_to_use = 'mps'
    else:
        device_to_use = 'cpu'
    print(f"Using device: {device_to_use}")

    model = YOLO(args.model) # Load a pretrained model (recommended for training)
    
    # Ensure the model is loaded onto the correct device before training, if necessary
    # model.to(device_to_use) # YOLO train call usually handles device transfer

    print(f"Starting YOLOv8 training with data: {args.data}, epochs: {args.epochs}, project: {args.project_name}")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device_to_use,
        lr0=1e-3, # Example: initial learning rate
        weight_decay=5e-4, # Example: weight decay
        project=args.project_name, # Main directory for saving runs
        name=args.name, # Specific run name, defaults to exp, exp2, etc.
        exist_ok=False, # True to overwrite existing run with same name
        patience=20, # Example: epochs to wait for no observable improvement before early stopping
        verbose=True
    )
    print(f"Finished YOLOv8 training. Results saved in {args.project_name}")

if __name__ == "__main__":
    # Need torch for device check, ensure it's imported
    import torch 
    main()