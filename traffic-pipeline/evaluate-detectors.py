"""
Loads best YOLOv8 + Faster‑RCNN weights, runs on val set,
reports precision / recall / mAP to CSV for quick comparison.
"""
import pandas as pd, subprocess, json, torch, torchvision
from ultralytics import YOLO
from pathlib import Path
import argparse

# Import from our modified training script
from train_fasterrcnn import CocoDet
from coco_eval import CocoEvaluator
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 and Faster R-CNN models.")
    # YOLOv8 arguments
    parser.add_argument("--yolo_weights", type=str, default="traffic-pipeline/runs/yolov8/weights/best.pt",
                        help="Path to trained YOLOv8 model weights.")
    parser.add_argument("--dataset_yaml", type=str, default="traffic-pipeline/dataset.yaml",
                        help="Path to the YOLO dataset.yaml file.")
    # Faster R-CNN arguments
    parser.add_argument("--fasterrcnn_weights", type=str, default="traffic-pipeline/runs/fasterrcnn/model_e49.pth",
                        help="Path to trained Faster R-CNN model weights (e.g., model_e49.pth for epoch 49).")
    parser.add_argument("--coco_json_val", type=str, default="traffic-pipeline/datasets/yolo_coco_from_yolo.json",
                        help="Path to the COCO JSON annotation file for validation.")
    parser.add_argument("--num_classes_frcnn", type=int, default=5, 
                        help="Number of classes for Faster R-CNN (including background).")
    # General arguments
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for YOLOv8 evaluation.")
    parser.add_argument("--device", type=str, default="mps", help="Device to use (e.g., 'mps', 'cuda', 'cpu').")
    parser.add_argument("--output_csv", type=str, default="traffic-pipeline/reports/detection_metrics.csv",
                        help="Path to save the detection metrics CSV file.")

    args = parser.parse_args()

    rows = []
    current_device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {current_device}")

    # ---- YOLOv8 eval -----------------------------------------------------------
    print(f"\nEvaluating YOLOv8 model: {args.yolo_weights}")
    if Path(args.yolo_weights).exists() and Path(args.dataset_yaml).exists():
        try:
            yolo = YOLO(args.yolo_weights)
            # Ensure the model is on the correct device for evaluation if not handled by .val()
            yolo.to(current_device) 
            metrics = yolo.val(data=args.dataset_yaml, imgsz=args.imgsz, plots=False, device=current_device, split='val')
            if metrics and hasattr(metrics, 'box') and metrics.box:
                rows.append({"model":"yolov8", "mAP50":metrics.box.map50, "mAP50‑95":metrics.box.map, "precision": metrics.box.mp, "recall": metrics.box.mr})
                print(f"YOLOv8 Metrics: mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}")
            else:
                print("Could not retrieve YOLOv8 metrics or metrics.box is None/empty.")
        except Exception as e:
            print(f"Error during YOLOv8 evaluation: {e}")
    else:
        print(f"Skipping YOLOv8 evaluation. Weights or dataset YAML not found.")
        print(f"  Weights path checked: {args.yolo_weights}")
        print(f"  Dataset YAML checked: {args.dataset_yaml}")

    # ---- Faster‑RCNN eval ------------------------------------------------------
    print(f"\nEvaluating Faster R-CNN model: {args.fasterrcnn_weights}")
    if Path(args.fasterrcnn_weights).exists() and Path(args.coco_json_val).exists():
        try:
            # CocoDet is imported from train_fasterrcnn and uses its updated image root path
            val_ds = CocoDet(json_path=args.coco_json_val, subset="val")
            if len(val_ds) == 0:
                print("Faster R-CNN validation dataset is empty. Check COCO JSON and subset 'val'. Skipping evaluation.")
            else:
                print(f"Loaded {len(val_ds)} images for Faster R-CNN validation.")
                model_frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=args.num_classes_frcnn)
                model_frcnn.load_state_dict(torch.load(args.fasterrcnn_weights, map_location=current_device))
                model_frcnn.to(current_device).eval()

                evaluator = CocoEvaluator(val_ds.coco_api if hasattr(val_ds, 'coco_api') else val_ds, ["bbox"])
                if not hasattr(val_ds, 'coco'):
                     print("Error: val_ds.coco is not available. CocoDet in train_fasterrcnn.py needs to provide a COCO API object for the validation subset.")
                     print("Skipping Faster R-CNN evaluation due to missing COCO API object.")
                else:
                    evaluator = CocoEvaluator(val_ds.coco, ["bbox"])
                    data_loader_val = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=utils.collate_fn if 'utils' in globals() else lambda x: tuple(zip(*x)))

                    for images, targets in data_loader_val:
                        images = [img.to(current_device) for img in images]
                        targets = [{k: v.to(current_device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                        with torch.no_grad():
                            outputs = model_frcnn(images)
                        
                        outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]
                        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
                        evaluator.update(res)
                    
                    evaluator.accumulate()
                    evaluator.summarize()
                    map_50_95 = evaluator.coco_eval['bbox'].stats[0]
                    map_50 = evaluator.coco_eval['bbox'].stats[1]
                    rows.append({"model":"fasterrcnn", "mAP50":map_50, "mAP50‑95":map_50_95, "precision": "N/A", "recall": "N/A"})
                    print(f"Faster R-CNN Metrics: mAP50={map_50:.4f}, mAP50-95={map_50_95:.4f}")

        except Exception as e:
            print(f"Error during Faster R-CNN evaluation: {e}")
    else:
        print(f"Skipping Faster R-CNN evaluation. Weights or COCO JSON for validation not found.")
        print(f"  Weights path checked: {args.fasterrcnn_weights}")
        print(f"  COCO JSON val checked: {args.coco_json_val}")

    # Ensure reports directory exists
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"\n✅ Detection metrics written to {args.output_csv}")
    else:
        print("\nNo evaluation results to write to CSV.")

if __name__ == '__main__':
    # This script needs utils.collate_fn from torchvision references detection folder.
    # For standalone execution, ensure utils.py is in PYTHONPATH or same directory.
    # We will try to import it, if not, use a basic lambda collate.
    try:
        import utils # Try to import utils
    except ImportError:
        print("Warning: utils.py not found. Using basic collate_fn for Faster R-CNN DataLoader.")
        # Define a simple collate_fn if utils is not available
        def basic_collate_fn(batch):
            return tuple(zip(*batch))
        # Assign to global utils to avoid NameError if utils is not imported
        # This is a bit of a hack; ideally utils.py is correctly located.
        import sys
        class UtilsMock:
            collate_fn = basic_collate_fn
        sys.modules['utils'] = UtilsMock()
        import utils # Now this should work

    main()