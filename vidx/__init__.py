"""Video indexing helpers shared by run.py and the tools/ scripts."""

# Target classes in the order the brief lists them, with their COCO ids in yolo11l.pt.
CLASS_NAMES = ["person", "car", "dog", "laptop", "cell phone"]
COCO_IDS = [0, 2, 16, 63, 67]
PERSON = 0  # index into CLASS_NAMES

CONF = 0.4  # grader threshold
IMGSZ = 640
IOU = 0.7
