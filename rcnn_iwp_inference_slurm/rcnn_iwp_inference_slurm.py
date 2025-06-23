
import logging
import subprocess
import json
import argparse
from typing import Dict, List
import shutil
import ray
from ray.util import ActorPool
import os
import sys
import gc

from rcnn_iwp_inference import load_model, get_metadata, run_inference



@ray.remote(num_cpus=1, memory=4 * 1024 * 1024 * 1024)
class ImageInferenceActor:
    def __init__(self, model_ref, metadata_ref):
        self.model = model_ref
        self.metadata = metadata_ref

    def process_image(self, image_path, threshold, output_folder, worker_id):
        try:
            print(f"Worker {worker_id} running inference on {image_path}")
            
            image_file = os.path.basename(image_path)
            
            # Create unique temporary output file using worker_id and image name
            temp_output_file = f"output_{worker_id}_{image_file}.jpg"
            
            # Run inference with custom output path
            coco_mask_metadata = run_inference(
                self.model, 
                image_path, 
                self.metadata, 
                threshold=threshold, 
                output_file=temp_output_file
            )
            
            # Create output file path
            output_file = os.path.join(output_folder, f"{image_file.split('.')[0]}_masked.jpg")
            
            # Copy the output image to the output folder
            shutil.copy(temp_output_file, output_file)
            os.remove(temp_output_file)
            
            # Force garbage collection to free memory
            gc.collect()
            
            return {"file": image_file, "metadata": coco_mask_metadata, "output_file": output_file}
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Force garbage collection even on error
            gc.collect()
            return {"file": os.path.basename(image_path), "error": str(e)}


# Load the model

model_file = "./model_weights.pth"
model = load_model("configs/mask_rcnn_vitdet.py", "configs/datasets/iwp.py", model_file)
metadata = get_metadata("iwp_test", "data/iwp/iwp_test.json")



# Initialize Ray
HEAD_NODE = os.environ["HEAD_NODE"]
RAY_PORT = os.environ["RAY_PORT"]

ray.init(address=f"{os.environ['HEAD_NODE']}:{os.environ['RAY_PORT']}",_node_ip_address=os.environ['HEAD_NODE'],
         ignore_reinit_error=True)


print(ray.available_resources())




#  Add to store
model_ref = ray.put(model)
metadata_ref = ray.put(metadata)



output_folder_name = f"output_images_masked"
output_folder = os.path.join(os.getcwd(), output_folder_name)
os.makedirs(output_folder, exist_ok=True)


dataset_folder = 'test_images'
dataset_folder = os.path.join(os.getcwd(), dataset_folder)
image_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder)]

print(image_files)


NUM_ACTORS = 4
actors = [ImageInferenceActor.remote(model_ref, metadata_ref) for _ in range(NUM_ACTORS)]
actors_pool = ActorPool(actors)


# Create task inputs with unique IDs
tasks = [
    (image_path, 0.6, output_folder, i) 
    for i, image_path in enumerate(image_files)
]

# Process all images using the actor pool
results = list(actors_pool.map(lambda actor, task: actor.process_image.remote(*task), tasks))


print(results)





