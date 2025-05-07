"""
Convert an existing YOLO format dataset to COCO JSON format.
Reads dataset structure and class names from a dataset.yaml file.
Assumes dataset.yaml path key points to the root of the YOLO dataset
(e.g., ../datasets/yolo relative to the yaml file).

Input:
    datasets/yolo/ (structured as per dataset.yaml)
        images/{train,val,test}/
        labels/{train,val,test}/
    traffic-pipeline/dataset.yaml

Output:
    traffic-pipeline/datasets/yolo_coco_from_yolo.json
"""
import json
import yaml # Requires PyYAML
import cv2  # Requires OpenCV-python
from pathlib import Path
import tqdm
import argparse

def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    """Converts YOLO bbox (center_x, center_y, width, height) [all relative]
    to COCO bbox (x_min, y_min, width, height) [all absolute]."""
    x_center_abs = yolo_bbox[0] * img_width
    y_center_abs = yolo_bbox[1] * img_height
    width_abs = yolo_bbox[2] * img_width
    height_abs = yolo_bbox[3] * img_height
    x_min = x_center_abs - (width_abs / 2)
    y_min = y_center_abs - (height_abs / 2)
    return [round(x_min, 2), round(y_min, 2), round(width_abs, 2), round(height_abs, 2)]

def convert_yolo_to_coco(yaml_file_path_str, output_coco_json_path_str):
    yaml_file_path = Path(yaml_file_path_str)
    output_coco_json_path = Path(output_coco_json_path_str)
    output_coco_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(yaml_file_path, 'r') as f:
        dataset_yaml = yaml.safe_load(f)

    yolo_dataset_root = (yaml_file_path.parent / Path(dataset_yaml['path'])).resolve()
    
    print(f"Reading YOLO dataset from: {yolo_dataset_root}")
    print(f"Using classes: {dataset_yaml['names']}")

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": int(k) + 1, "name": v, "supercategory": "vehicle"} 
                       for k, v in dataset_yaml['names'].items()]
    }

    image_id_counter = 1
    annotation_id_counter = 1

    splits = {}
    if 'train' in dataset_yaml and dataset_yaml['train']:
        splits['train'] = Path(dataset_yaml['train'])
    if 'val' in dataset_yaml and dataset_yaml['val']:
        splits['val'] = Path(dataset_yaml['val'])
    if 'test' in dataset_yaml and dataset_yaml['test']:
        splits['test'] = Path(dataset_yaml['test'])
    
    if not splits:
        print("Error: No train, val, or test splits defined in dataset.yaml under 'train', 'val', or 'test' keys.")
        return

    for split_name, relative_images_dir_in_yaml in splits.items():
        actual_images_dir = yolo_dataset_root / relative_images_dir_in_yaml
        # Assumes labels are in a parallel structure relative to yolo_dataset_root, e.g., images/train -> labels/train
        actual_labels_dir = yolo_dataset_root / "labels" / relative_images_dir_in_yaml.name

        print(f"Processing split: {split_name}")
        print(f"  Image directory: {actual_images_dir}")
        print(f"  Label directory: {actual_labels_dir}")

        if not actual_images_dir.exists():
            print(f"Warning: Image directory not found for split '{split_name}': {actual_images_dir}")
            continue
        if not actual_labels_dir.exists():
            print(f"Warning: Label directory not found for split '{split_name}': {actual_labels_dir}")
            continue

        image_files = sorted(list(actual_images_dir.glob('*.jpg')) +
                             list(actual_images_dir.glob('*.png')) +
                             list(actual_images_dir.glob('*.jpeg')))

        for img_path in tqdm.tqdm(image_files, desc=f"Processing {split_name} images"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Warning: Could not read image {img_path}. Skipping.")
                    continue
                img_height, img_width = img.shape[:2]
            except Exception as e:
                print(f"Warning: Error reading image {img_path}: {e}. Skipping.")
                continue

            coco_image_file_name = f"{split_name}/{img_path.name}"
            
            coco_output["images"].append({
                "id": image_id_counter,
                "width": img_width,
                "height": img_height,
                "file_name": coco_image_file_name,
            })

            label_file_path = actual_labels_dir / f"{img_path.stem}.txt"
            if label_file_path.exists():
                with open(label_file_path, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if not parts: continue # Skip empty lines
                        try:
                            class_id_yolo = int(parts[0])
                            yolo_bbox = [float(p) for p in parts[1:5]]
                            if len(yolo_bbox) < 4:
                                print(f"Warning: Malformed bbox in {label_file_path} for image {img_path.name}. Line: '{line.strip()}'. Skipping annotation.")
                                continue
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Error parsing line in {label_file_path} for image {img_path.name}. Line: '{line.strip()}'. Error: {e}. Skipping annotation.")
                            continue
                        
                        coco_bbox = yolo_to_coco_bbox(yolo_bbox, img_width, img_height)
                        
                        coco_output["annotations"].append({
                            "id": annotation_id_counter,
                            "image_id": image_id_counter,
                            "category_id": class_id_yolo + 1, 
                            "bbox": coco_bbox,
                            "area": coco_bbox[2] * coco_bbox[3],
                            "iscrowd": 0,
                            "segmentation": [] 
                        })
                        annotation_id_counter += 1
            image_id_counter += 1

    with open(output_coco_json_path, 'w') as f:
        json.dump(coco_output, f, indent=4)
    
    print(f"\n✅ COCO JSON created at: {output_coco_json_path}")
    print(f"Total images: {len(coco_output['images'])}")
    print(f"Total annotations: {len(coco_output['annotations'])}")

if __name__ == "__main__":
    default_yaml_path = "traffic-pipeline/dataset.yaml"
    default_coco_output_path = "traffic-pipeline/datasets/yolo_coco_from_yolo.json"
    
    parser = argparse.ArgumentParser(description="Convert YOLO dataset to COCO JSON format.")
    parser.add_argument("--yaml_path", type=str, default=default_yaml_path, 
                        help="Path to the dataset.yaml file.")
    parser.add_argument("--output_coco_path", type=str, default=default_coco_output_path,
                        help="Path to save the output COCO JSON file.")
    args = parser.parse_args()

    if not Path(args.yaml_path).exists():
        print(f"Error: dataset.yaml not found at {args.yaml_path}")
        print("Please ensure the --yaml_path argument is correct or the file exists at the default location.")
    else:
      convert_yolo_to_coco(args.yaml_path, args.output_coco_path)