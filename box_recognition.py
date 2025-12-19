import cv2
from ultralytics import YOLO

MODEL_PATH = "models/recognition.pt"
rec_model = YOLO(MODEL_PATH)


def recognize_box(box, det_results, img):
    cls_idx = int(box.cls[0])
    cls_name = det_results.names[cls_idx]

    if cls_name not in ("container_number_h", "container_number_v"):
        return None

    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    rec_results = rec_model(
        roi,
        imgsz=640,
        conf=0.3,
        verbose=False
    )[0]

    char_boxes = []
    for cbox in rec_results.boxes:
        c_cls_idx = int(cbox.cls[0])
        char_label = rec_results.names[c_cls_idx]
        cx1, cy1, cx2, cy2 = map(int, cbox.xyxy[0].tolist())
        char_boxes.append((cx1, cy1, char_label))

    if not char_boxes:
        return None

    if cls_name == "container_number_h":
        char_boxes.sort(key=lambda t: t[0])
    else:
        char_boxes.sort(key=lambda t: t[1])

    recognized = "".join(char for _, _, char in char_boxes)
    return recognized
