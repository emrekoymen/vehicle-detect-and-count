"""
Fine‑tune torchvision Faster‑RCNN on COCO‑style JSON.
Uses torch.utils.data.DataLoader + torchvision's references/detection engine.
"""
import argparse, torch, torchvision, utils, transforms as T, json, numpy as np, cv2, pathlib
from engine import train_one_epoch, evaluate  # copy from torchvision references
from pathlib import Path

class CocoDet(torch.utils.data.Dataset):
    def __init__(self, json_path, subset):
        with open(json_path) as f:
            ann = json.load(f)
        
        # Filter images based on the 'file_name' starting with the subset
        # e.g., if subset is 'train', file_name should be 'train/image_name.jpg'
        self.imgs = [i for i in ann["images"] if i["file_name"].startswith(f"{subset}/")]
        self.anns = ann["annotations"] # Keep all annotations, they will be filtered by image_id later
        
        # Create a map of image_id to its annotations for quick lookup
        self.img_to_anns = {}
        for a in self.anns:
            img_id = a["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(a)

        self.cat2id = {c["id"]: c["name"] for c in ann["categories"]}
        # Image root path updated to point to the common 'images' directory within 'datasets/yolo/'
        self.root = pathlib.Path("datasets/yolo/images") 

    def __getitem__(self, idx):
        img_info = self.imgs[idx] # This img_info is already filtered for the correct subset
        
        # img_info["file_name"] is like "train/img001.jpg". self.root is "datasets/yolo/images"
        # So, the full path will be "datasets/yolo/images/train/img001.jpg"
        img_full_path = self.root / img_info["file_name"] 
        
        img = cv2.imread(str(img_full_path))
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_full_path}. Check COCO json file_name and dataset structure.")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize and convert to tensor (basic transform)
        # For more advanced transforms, replace T.ToTensor() with a T.Compose([...])
        img = torch.from_numpy(img / 255.).permute(2,0,1).float() # Basic ToTensor equivalent for numpy array

        # Get annotations for the current image_id
        # img_info["id"] is the COCO image_id
        current_image_id = img_info["id"]
        anns_for_image = self.img_to_anns.get(current_image_id, [])
        
        boxes = torch.tensor([a["bbox"] for a in anns_for_image], dtype=torch.float32)
        # Convert COCO bbox (x,y,w,h) to (x1,y1,x2,y2)
        boxes[:,2:] += boxes[:,:2] 
        labels = torch.tensor([a["category_id"] for a in anns_for_image], dtype=torch.int64)
        
        # Ensure target tensors are not empty if there are no annotations
        if not anns_for_image:
            # If no annotations, provide empty tensors of the correct type and shape for collation
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([current_image_id])} # Use COCO image_id
        return img, target
    
    def __len__(self): 
        return len(self.imgs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4, help="Batch size for training and validation")
    ap.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    ap.add_argument("--coco_json", type=str, 
                        default="traffic-pipeline/datasets/yolo_coco_from_yolo.json",
                        help="Path to the COCO format annotation file.")
    ap.add_argument("--num_classes", type=int, default=5, 
                        help="Number of classes including background (e.g., 4 classes + 1 background = 5).")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # num_classes should be number of your objects + 1 (for background)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT", num_classes=args.num_classes)
    model.to(device)

    train_ds = CocoDet(json_path=args.coco_json, subset="train")
    val_ds   = CocoDet(json_path=args.coco_json, subset="val")
    
    print(f"Found {len(train_ds)} training images and {len(val_ds)} validation images.")
    if len(train_ds) == 0:
        print("Error: Training dataset is empty. Check COCO JSON and subset name ('train').")
        return
    if len(val_ds) == 0:
        print("Warning: Validation dataset is empty. Check COCO JSON and subset name ('val'). Evaluation might fail or be meaningless.")


    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=utils.collate_fn, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=utils.collate_fn, num_workers=2
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.1)

    save_dir_root = Path("traffic-pipeline/runs/fasterrcnn")
    save_dir_root.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_one_epoch(model, opt, train_loader, device, epoch, print_freq=10) # print_freq can be adjusted
        lr_sched.step()
        if val_loader and len(val_ds) > 0:
             evaluate(model, val_loader, device=device)
        else:
            print("Skipping evaluation as validation loader is not available or validation set is empty.")
        
        # Save model checkpoint
        model_save_path = save_dir_root / f"model_e{epoch}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model checkpoint to {model_save_path}")

if __name__ == "__main__":
    main()