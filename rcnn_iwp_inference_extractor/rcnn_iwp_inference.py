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
def run_inference(model, image_path, metadata, threshold=0.6):
    image = cv2.imread(image_path)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)
    batched_inputs = [{"image": image_tensor}]
    
    # TODO: Instead of printing, tag the image with the metadata of boxes and classes.
    outputs = model(batched_inputs)
    print(outputs[0]["instances"].pred_classes)
    print(outputs[0]["instances"].pred_boxes)

    instances = outputs[0]["instances"]
    mask = instances.scores >= threshold
    filtered_instances = instances[mask]

    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
    out = v.draw_instance_predictions(filtered_instances.to("cpu"))
    cv2.imwrite("output.jpg", out.get_image()[:, :, ::-1])



if __name__ == "__main__":

    model = load_model("configs/mask_rcnn_vitdet.py", "configs/datasets/iwp.py", "model_final.pth")
    metadata = get_metadata("iwp_test", "configs/metadata/iwp_test.json")

    run_inference(model, "FID_434_Polygon_2.jpg", metadata, threshold=0.6)
    