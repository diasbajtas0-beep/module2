from ultralytics import YOLO

for p in ["models/detection.pt", "models/recognition.pt"]:
    m = YOLO(p)
    print("\n====", p, "====")
    print("task:", m.task)
    print("nc:", len(m.names))
    print("names:", m.names)
    # самое главное:
    print("yaml:", m.model.yaml)      # тут depth_multiple/width_multiple
    print("args:", getattr(m, "args", None))  # иногда есть train args
