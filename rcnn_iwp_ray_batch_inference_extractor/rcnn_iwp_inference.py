#Import required libraries
from detectron2.utils.visualizer import Visualizer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets.coco import register_coco_instances
from detectron2.data import MetadataCatalog
from omegaconf import OmegaConf
import torch

import cv2
import json

# Function to load the model and weights
def load_model(model_config, dataset_config, model_weights):
    cfg = LazyConfig.load(model_config)
    # merge dataset specific config
    dataset_cfg = LazyConfig.load(dataset_config)
    cfg = OmegaConf.merge(cfg, dataset_cfg)

    model = instantiate(cfg.model)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(model_weights)
    model.eval()
    return model

def get_metadata(dataset_name, dataset_path):
    register_coco_instances(
        dataset_name,
        {},
        dataset_path,
        "",
    )
    metadata = MetadataCatalog.get(dataset_name)
    return metadata
    

# Function to run inference on the image
def run_inference(model, image_path, metadata, threshold=0.6, output_file="output.jpg"):
    image = cv2.imread(image_path)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)
    batched_inputs = [{"image": image_tensor}]
    
    outputs = model(batched_inputs)
    instances = outputs[0]["instances"]
    mask = instances.scores >= threshold
    filtered_instances = instances[mask]
    
    # Extract bounding box data in COCO format
    coco_boxes = extract_coco_boxes(filtered_instances, image_path)
    
    # Visualize and save the output image (original functionality)
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
    out = v.draw_instance_predictions(filtered_instances.to("cpu"))
    cv2.imwrite(output_file, out.get_image()[:, :, ::-1])
    
    return coco_boxes

def extract_coco_boxes(instances, image_path):
    """
    Extract only bounding box information in COCO format
    
    Args:
        instances: Detectron2 Instances object with detection results
        image_path: Path to the input image
    
    Returns:
        List of dictionaries with bounding box annotations in COCO format
    """
    import os
    
    # Move to CPU for easier processing
    instances = instances.to("cpu")
    
    # Initialize results list
    coco_boxes = []
    

    # Extract bounding boxes, scores and classes
    if hasattr(instances, "pred_boxes") and len(instances.pred_boxes) > 0:
        # Use detach() to remove gradient requirements before converting to numpy
        boxes = instances.pred_boxes.tensor.detach().numpy()
        scores = instances.scores.detach().numpy()
        classes = instances.pred_classes.detach().numpy()
        
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
            # Convert from [x1, y1, x2, y2] to [x, y, width, height] format for COCO
            x1, y1, x2, y2 = box
            coco_box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            
            # Calculate area
            area = float((x2 - x1) * (y2 - y1))
            
            annotation = {
                "id": i + 1,
                "category_id": int(class_id) + 1,  # COCO uses 1-indexed category IDs
                "bbox": coco_box,
                "area": area,
                "score": float(score)
            }
            
            coco_boxes.append(annotation)
    
    return coco_boxes

if __name__ == "__main__":
    model = load_model("configs/mask_rcnn_vitdet.py", "configs/datasets/iwp.py", "model_weights.pth")
    metadata = get_metadata("iwp_test", "configs/metadata/iwp_test.json")

    coco_boxes = run_inference(model, "FID_434_Polygon_2.jpg", metadata, threshold=0.6)
    
    # Save COCO bounding box annotations to file
    with open("coco_boxes.json", "w") as f:
        json.dump(coco_boxes, f, indent=2)
    