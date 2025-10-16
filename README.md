# 🚦 YOLO_SceneGraph: Scene Graph Generation for Traffic Images

**YOLO_SceneGraph** is a Python project that uses the YOLO object detection model to detect objects in traffic images and generate **scene graphs** based on spatial and semantic relationships between those objects.

This tool can help visualize traffic scenarios, understand urban environments, or support downstream AI applications like autonomous driving, smart city analytics, and behavioral modeling.

---

## 🧠 Features

- 🏙️ **Object Detection** using YOLOv5
- 🧭 **Spatial Relationship Reasoning** between detected traffic entities
- 🧩 **Scene Graph Generation** (with `networkx`)
- 🖼️ **Annotated Images** with labeled bounding boxes
- 🗂️ **JSON Output** of objects and relationships
- 📊 **Graph Visualization** as PNG images

---

## 📁 Folder Structure
```bash
YOLO_SCENEGRAPH/
├── images/                 # Input traffic scene images (.jpg)
├── output_graphs/          # Output: scene graph images (.png) and annotated images
├── output_json/            # Output: structured scene graph data in JSON
├── yolov5/                 # YOLOv5 directory (excluded from Git via .gitignore)

├── .gitattributes
├── .gitignore              # yolov5 is ignored here

├── car_relations.json      # Example: output JSON with car-focused relationships
├── output_boxes.json       # Example: detection output with bounding boxes
├── output_with_boxes.jpg   # Example: annotated image

├── YOLOv5_Test.py          # Main Python script for scene graph generation
├── yolov5s.pt              # YOLOv5 small model
├── yolov5su.pt             # (Optional) YOLOv5 custom/updated model
└── README.md               # Project documentation (this file)
```

## 📦 Requirements

Install dependencies via pip:

```bash
pip install ultralytics opencv-python pillow networkx matplotlib
```
