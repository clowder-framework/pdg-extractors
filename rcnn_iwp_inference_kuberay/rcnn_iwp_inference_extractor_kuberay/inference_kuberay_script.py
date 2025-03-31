#!/usr/bin/env python
import ray
from ray.util import ActorPool
import os
import sys
import gc
import logging
import requests
import json
import posixpath
import pyclowder.files
import pyclowder.datasets

from clowder_utils import get_folder_files 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcnn_iwp_inference import load_model, get_metadata, run_inference

# @ray.remote()
@ray.remote(num_cpus=1, memory=4 * 1024 * 1024 * 1024)
class ImageInferenceActor:
    def __init__(self, model_ref, metadata_ref, file_dict_ref):
        self.model = model_ref
        self.metadata = metadata_ref
        self.file_dict = file_dict_ref

    def process_image(self, file_id, threshold, submission_id, worker_id ):
        # Get the image path from the file_dict
        image_path = self.file_dict[file_id]['path']
        
        try:
            print(f"Worker {worker_id} running inference on {image_path}")

            output_file_name = f"{file_dict[file_id]['name']}_masked.jpg"
            output_file_path = os.path.join(os.getenv("MINIO_MOUNTED_PATH"), submission_id, output_file_name)

            # Run inference with custom output path
            coco_mask_metadata = run_inference(
                self.model, 
                image_path, 
                self.metadata, 
                threshold=threshold, 
                output_file_path=output_file_path
            )

            # Force garbage collection to free memory
            gc.collect()

            print(f"Worker {worker_id} completed inference on {image_path}")
            # Return the image file data and metadata
            return {
                "file_id": file_id,
                "coco_mask_metadata": coco_mask_metadata,
                "output_file_name": output_file_name
            }
        
        except Exception as e:
            print(f"Error processing image {file_id}: {e}")
            # Force garbage collection even on error
            gc.collect()
            return {
                "file_id": file_id,
                "error": str(e)
            }
        
      

if __name__ == "__main__":
    args = sys.argv[1:]
    host = args[0]
    key = args[1]
    dataset_id = args[2]
    dataset_folder_id = args[3]
    model_file_id = args[4]
    confidence_threshold = float(args[5])
    submission_id = args[6]

    logger = logging.getLogger(__name__)

    # List the files in the dataset folder, limited to 3000 files
    files = get_folder_files(host, key, dataset_id, dataset_folder_id)

    # Create a dictionary of file_id to file_name and file_path
    file_dict = {file['id']: {'name': file['name'], 'path': os.path.join(os.getenv("MINIO_MOUNTED_PATH"), file['id'])} for file in files}



    model_file_path = os.path.join(os.getenv("MINIO_MOUNTED_PATH"), model_file_id)
    # Load the model and metadata
    model = load_model("configs/mask_rcnn_vitdet.py", "configs/datasets/iwp.py", model_file_path)
    metadata = get_metadata("iwp_test", "data/iwp/iwp_test.json")

    # Add to the ray store
    model_ref = ray.put(model)
    metadata_ref = ray.put(metadata)
    file_dict_ref = ray.put(file_dict)

    # Create a new folder for the output images
    output_folder_path = os.path.join(os.getenv("MINIO_MOUNTED_PATH"), submission_id)
    os.makedirs(output_folder_path, exist_ok=True)

    NUM_ACTORS = os.getenv("NUM_ACTORS", 2)
    actors = [ImageInferenceActor.remote(model_ref, metadata_ref, file_dict_ref) for _ in range(NUM_ACTORS)]
    actor_pool = ActorPool(actors)

    # Create task inputs with unique IDs
    tasks = [
        (file_id, confidence_threshold, submission_id, i) 
        for i, file_id in enumerate(file_dict.keys())
    ]
    
    # Process all images using the actor pool
    logger.info(f"Starting inference on {len(tasks)} images using {NUM_ACTORS} actors")
    results = list(actor_pool.map(lambda actor, task: actor.process_image.remote(*task), tasks))
    logger.info(f"Completed inference on all images")

    
    # Process results
    # Create a json file with the results to the file_id as a dict and save it to MINIO_MOUNTED_PATH
    results_json = {}
    
    for result in results:
        if "error" in result:
            print(f"Error processing image {result['file_id']}: {result['error']}")
            results_json[result["file_id"]] = {
                "error": result["error"]
            }
            continue
        
        results_json[result["file_id"]] = {
            "coco_mask_metadata": result["coco_mask_metadata"],
            "output_file_name": result["output_file_name"]
        }
        
    # Save the results to the MINIO_MOUNTED_PATH
    results_json_path = os.path.join(os.getenv("MINIO_MOUNTED_PATH"), f"iwp_detection_results_{submission_id}.json")
    with open(results_json_path, 'w') as f:
        json.dump(results_json, f)

        

    


