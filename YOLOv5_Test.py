from ultralytics import YOLO
import json
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# === PATHS ===
image_folder = Path("images")
json_output_folder = Path("output_json")
graph_output_folder = Path("output_graphs")
json_output_folder.mkdir(exist_ok=True)
graph_output_folder.mkdir(exist_ok=True)

# === LOAD YOLOv5 MODEL ===
model = YOLO("yolov5s.pt")  # Make sure the model file is available

# === TRAFFIC CLASSES TO KEEP ===
TRAFFIC_CLASSES = {
    "car", "motorcycle", "bus", "truck", "traffic light",
    "fire hydrant", "stop sign", "bicycle", "parking meter", "person", "crosswalk"
}

# === RELATIONSHIP ID MAPPING ===
RELATION_ID_MAP = {
    "on": 0,
    "above": 1,
    "below": 2,
    "next to": 3,
    "in front of": 4,
    "behind": 5,
    "inside": 6,
    "stopped at": 7,
    "waiting at": 8,
    "crossing": 9,
    "riding": 10,
    "parked on": 11,
    "moving toward": 12,
    "moving away from": 13,
    "turning at": 14,
    "attached to": 15,
    "mounted on": 16,
    "part of": 17,
    "blocking": 18,
    "covering": 19
}

# === HELPER FUNCTIONS ===
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def get_area(box):
    x1, y1, x2, y2 = box
    return max(1, (x2 - x1) * (y2 - y1))

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea / float(get_area(boxA) + get_area(boxB) - interArea)

def get_spatial_relationship(sub, obj, cls_s, cls_o):
    cx_s, cy_s = get_center(sub)
    cx_o, cy_o = get_center(obj)

    # Traffic-specific relationships
    if cls_s == "car" and cls_o == "traffic light" and iou(sub, obj) < 0.1 and abs(cx_s - cx_o) < 80:
        return "stopped at"
    if cls_s == "person" and cls_o == "crosswalk" and iou(sub, obj) > 0.2:
        return "crossing"
    if cls_s == "person" and cls_o == "bicycle" and iou(sub, obj) > 0.15:
        return "riding"

    # Core geometric relationships
    if abs(cy_s - cy_o) < 30 and cx_s < cx_o:
        return "next to"
    elif cy_s < obj[1]:
        return "above"
    elif cy_s > obj[3]:
        return "below"
    elif abs(cx_s - cx_o) < 50 and cy_s < cy_o:
        return "in front of"
    elif abs(cx_s - cx_o) < 50 and cy_s > cy_o:
        return "behind"
    elif (sub[0] >= obj[0] and sub[1] >= obj[1] and sub[2] <= obj[2] and sub[3] <= obj[3]):
        return "inside"
    else:
        return "on"

# === PROCESS EACH IMAGE ===
for image_path in image_folder.glob("*.jpg"):
    try:
        img = Image.open(image_path).convert("RGB")
        results = model(img, conf=0.5)[0]
        boxes = results.boxes
        class_names = model.names

        if not boxes:
            print(f"No objects detected in {image_path.name}")
            continue

        objects = []
        relationships = []
        box_map = {}

        for i, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            bbox = list(map(int, box.xyxy[0].tolist()))
            cls_name = class_names[cls_id]

            if cls_name not in TRAFFIC_CLASSES:
                continue  # Skip irrelevant classes

            objects.append({"id": i, "name": cls_name, "bbox": bbox})
            box_map[i] = (bbox, cls_name)

        for i in box_map:
            for j in box_map:
                if i != j:
                    box_i, cls_i = box_map[i]
                    box_j, cls_j = box_map[j]
                    rel = get_spatial_relationship(box_i, box_j, cls_i, cls_j)
                    if rel in RELATION_ID_MAP:
                        relationships.append({
                            "subject_id": i,
                            "object_id": j,
                            "predicate": rel,
                            "predicate_id": RELATION_ID_MAP[rel]
                        })

        # === SAVE JSON ===
        output = {
            "image_id": image_path.name,
            "objects": objects,
            "relationships": relationships
        }

        with open(json_output_folder / f"{image_path.stem}.json", "w") as f:
            json.dump(output, f, indent=2)

        # === DRAW BOUNDING BOXES ON IMAGE ===
        img_cv = cv2.imread(str(image_path))
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            label = f"{obj['name']} {obj['id']}"
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        annotated_image_path = graph_output_folder / f"{image_path.stem}_boxed.jpg"
        cv2.imwrite(str(annotated_image_path), img_cv)

        # === DRAW SCENE GRAPH ===
        G = nx.DiGraph()
        for obj in objects:
            G.add_node(obj['id'], label=obj['name'])
        for rel in relationships:
            G.add_edge(rel['subject_id'], rel['object_id'], label=rel['predicate'])

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True,
                labels=nx.get_node_attributes(G, 'label'),
                node_color='skyblue', node_size=2000, font_size=10)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

        plt.title(f"Scene Graph: {image_path.name}")
        plt.tight_layout()
        plt.savefig(graph_output_folder / f"{image_path.stem}_graph.png")
        plt.close()

        print(f"Processed: {image_path.name}")

    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")

print("All scene graphs, JSONs, and annotated images generated.")
