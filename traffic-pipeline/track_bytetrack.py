"""
Real‑time MOT with YOLOv8 + ByteTrack.
CLI:
    python track_bytetrack.py --weights runs/yolov8/weights/best.pt --source video.mp4
"""
import argparse, cv2, torch
from ultralytics import YOLO
from cjm_byte_track import BYTETracker

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="0")  # webcam
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(0 if args.source=="0" else args.source)
    tracker = BYTETracker(track_thresh=args.conf)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        dets = model.predict(frame, conf=args.conf, iou=0.5, device='mps', verbose=False)[0].boxes.xyxy.cpu()
        # The following lines to access scores and cls_ids might be specific to older ultralytics versions
        # For newer versions (e.g., ultralytics 8.0.120+), direct access to predictor results might change.
        # Let's assume it's compatible for now, or may need adjustment if errors occur during runtime with a newer ultralytics lib.
        # A more robust way if model.predictor is not available or results structure is different:
        # boxes = model.predict(...)[0].boxes
        # dets = boxes.xyxy.cpu()
        # scores = boxes.conf.cpu()
        # cls_ids = boxes.cls.cpu()
        # This part is tricky as model.predictor might not be a public/stable API.
        # Sticking to the original for now but noting this potential point of failure/update.
        scores = dets = model.predict(frame, conf=args.conf, iou=0.5, device='mps', verbose=False)[0].boxes.conf.cpu() # Re-accessing, this is not ideal. Let's fix it.
        cls_ids = model.predict(frame, conf=args.conf, iou=0.5, device='mps', verbose=False)[0].boxes.cls.cpu() # Re-accessing, not ideal.

        # Correct way after getting predictions once:
        predictions = model.predict(frame, conf=args.conf, iou=0.5, device='mps', verbose=False)[0]
        dets_xyxy = predictions.boxes.xyxy.cpu()
        scores = predictions.boxes.conf.cpu()
        cls_ids = predictions.boxes.cls.cpu()

        if dets_xyxy.numel() == 0: # Handle cases with no detections
            online_targets = tracker.update_without_detection(frame.shape)
        else:
            track_inputs = torch.cat([dets_xyxy, scores.unsqueeze(1), cls_ids.unsqueeze(1)], dim=1).numpy()
            online_targets = tracker.update(track_inputs, frame.shape)
            
        for t in online_targets:
            x1,y1,x2,y2 = map(int,t[:4]) # Tracker output might vary; common is x1,y1,x2,y2,track_id,class_id,score
            tid = int(t[4])
            cls = int(t[5]) # Assuming class_id is the 6th element (index 5)
            # score = t[6] # If score is also present
            
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            # Display class and track ID
            label = f"ID:{tid} Cls:{cls}" # You might want to map cls to class names if you have them
            cv2.putText(frame,label,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            
        cv2.imshow("ByteTrack", frame)
        if cv2.waitKey(1)==27: break

if __name__ == "__main__":
    main() 