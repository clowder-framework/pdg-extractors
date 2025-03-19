#!/usr/bin/env python

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

from pyclowder.extractors import Extractor
import pyclowder.files
import pyclowder.datasets

from pyclowder.utils import CheckMessage
from clowder_utils import create_symlink_folder, create_symlink_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

class PDGInferenceExtractor(Extractor):
    """Extractor that runs inference with a trained model on a given image to detect IWPs."""
    # TODO: This extractor is hard coded to use a specific model and dataset metadata. 
    # We need to make it more flexible
    def __init__(self):
        Extractor.__init__(self)
        self.setup()

        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)
        
    
    def process_message(self, connector, host, secret_key, resource, parameters):
        # Process the file and upload the results
        logger = logging.getLogger(__name__)

        # Initialize Ray with memory limits and other optimizations
        ray.init(
            # Set memory threshold higher to avoid killing tasks too aggressively
            _system_config={
                "object_spilling_threshold": 0.8,  # Spill objects to disk at 80% memory usage
                "memory_usage_threshold": 0.98,  # Kill workers only at 95% memory usage 
            },
            # Configure object store memory - adjust based on your system
            object_store_memory=2 * 1024 * 1024 * 1024,  # 2GB for object store,
            ignore_reinit_error=True
            
        )
        
        file_path = resource["local_paths"][0]
        file_id = resource['id']
        file_name = parameters['filename']
        dataset_id = resource['parent']['id']

        # Load user-defined params from the GUI.
        MODEL_FILE_ID = ""
        CONFIDENCE_THRESHOLD = 0.6
        params = None

        if "parameters" in parameters:
            try:
                params = json.loads(parameters['parameters'])
            except TypeError as e:
                print(f"Failed to load parameters, it's not compatible with json.loads().\nError:{e}")
                if type(parameters == Dict):
                    params = parameters['parameters']

        if "MODEL_FILE" in params:
            model_metadata = json.loads(params["MODEL_FILE"])
            MODEL_FILE_ID = model_metadata["selectionID"]
        else:
            raise ValueError("MODEL_FILE is not provided in the parameters.")
        
        if "CONFIDENCE_THRESHOLD" in params:
            CONFIDENCE_THRESHOLD = params["CONFIDENCE_THRESHOLD"]
        else:
            raise ValueError("CONFIDENCE_THRESHOLD is not provided in the parameters.")
        
        if "DATASET_FOLDER" in params:
            dataset_folder_metadata = json.loads(params["DATASET_FOLDER"])
        else:
            raise ValueError("DATASET_FOLDER is not provided in the parameters.")

        dataset_folder, file_id_dict = create_symlink_folder(
            host, 
            secret_key, 
            dataset_folder_metadata["datasetId"], 
            dataset_folder_metadata["selectionID"], 
            dataset_folder_metadata["selectionName"], 
            logger=logger
        )
        folder_name = dataset_folder_metadata["selectionName"]
        print(file_id_dict)

        # Download the model file
        model_file = pyclowder.files.download(connector, host, secret_key, fileid=MODEL_FILE_ID)

        # Load the model and metadata
        model = load_model("configs/mask_rcnn_vitdet.py", "configs/datasets/iwp.py", model_file)
        metadata = get_metadata("iwp_test", "data/iwp/iwp_test.json")


        #  Add to store
        model_ref = ray.put(model)
        metadata_ref = ray.put(metadata)

        # Create a new folder for the output images
        output_folder_name = f"{folder_name}_masked"
        output_folder = os.path.join(os.getcwd(), output_folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        # Collect all image files in the dataset folder
        image_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder)]
        
        # Create a pool of actors for inference
        NUM_ACTORS = 4 
        actors = [ImageInferenceActor.remote(model_ref, metadata_ref) for _ in range(NUM_ACTORS)]
        actors_pool = ActorPool(actors)
        
        # Create task inputs with unique IDs
        tasks = [
            (image_path, CONFIDENCE_THRESHOLD, output_folder, i) 
            for i, image_path in enumerate(image_files)
        ]
        
        # Process all images using the actor pool
        logger.info(f"Starting inference on {len(image_files)} images using {NUM_ACTORS} actors")
        results = list(actors_pool.map(lambda actor, task: actor.process_image.remote(*task), tasks))
        logger.info(f"Completed inference on all images")
        
        # Process results
        for result in results:
            if "error" in result:
                logger.error(f"Error processing {result['file']}: {result['error']}")
                continue
                
            image_file = result["file"]
            coco_mask_metadata = result["metadata"]
            
            # Update the metadata of the file
            metadata = self.get_metadata({"COCO Bounding Boxes": coco_mask_metadata}, 'file', file_id_dict[image_file], host)
            pyclowder.files.upload_metadata(connector, host, secret_key, file_id_dict[image_file], metadata)

        # Create a new folder in Clowder for the output images
        clowder_output_folder = pyclowder.datasets.create_folder(connector, host, secret_key, dataset_id, output_folder_name)
        
        # Collect all output files
        filepaths = [os.path.join(output_folder, file) for file in os.listdir(output_folder)]

        # Upload the results
        pyclowder.files.upload_multiple_files(connector, host, secret_key, dataset_id, filepaths, folder_id=clowder_output_folder)

        # Clean up resources
        shutil.rmtree(dataset_folder)
        shutil.rmtree(output_folder)
        
        # Shutdown Ray
        ray.shutdown()

if __name__ == "__main__":
    extractor = PDGInferenceExtractor()
    extractor.start()
    

            
        