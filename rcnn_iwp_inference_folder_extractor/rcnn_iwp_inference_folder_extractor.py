#!/usr/bin/env python

import logging
import subprocess
import json
import argparse
from typing import Dict
import shutil

from pyclowder.extractors import Extractor
import pyclowder.files

from pyclowder.utils import CheckMessage
from clowder_utils import create_symlink_folder, create_symlink_file

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcnn_iwp_inference import load_model, get_metadata, run_inference

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
        
        file_path = resource["local_paths"][0]
        file_id = resource['id']
        file_name = parameters['filename']
        dataset_id = resource['parent']['id']

        # Load user-defined params from the GUI.
        MODEL_FILE_ID = ""
        CONFIDENCE_THRESHOLD=0.6
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

        dataset_folder, file_id_dict = create_symlink_folder(host, secret_key,dataset_folder_metadata["datasetId"],dataset_folder_metadata["selectionID"], dataset_folder_metadata["selectionName"], logger=logger)
        folder_name = dataset_folder_metadata["selectionName"]
        print(file_id_dict)

        # Download the model file
        model_file = pyclowder.files.download(connector, host, secret_key, fileid=MODEL_FILE_ID)

        # Load the model and metadata
        model = load_model("configs/mask_rcnn_vitdet.py", "configs/datasets/iwp.py", model_file)
        metadata = get_metadata("iwp_test", "data/iwp/iwp_test.json")

        # Create a new folder for the output images
        output_folder_name = f"{folder_name}_masked"
        output_folder = os.path.join(os.getcwd(),output_folder_name)
        os.makedirs(output_folder, exist_ok=True)
        # Run inference on the images in the dataset folder
        for image_file in os.listdir(dataset_folder):
            print(f"Running inference on {os.path.join(dataset_folder, image_file)}")
            coco_mask_metadata = run_inference(model, os.path.join(dataset_folder, image_file), metadata, threshold=CONFIDENCE_THRESHOLD)
            metadata = {"COCO Bounding Boxes": coco_mask_metadata}
            # post metadata to Clowder
            metadata = self.get_metadata(metadata, 'file', file_id_dict[image_file], host)

            # Update the metadata of file
            pyclowder.files.upload_metadata(connector, host, secret_key, file_id_dict[image_file], metadata)

            # Change the name of the output image to the name of the input image with _masked appended
            # output_file = f"{output_folder}/{image_file.split('.')[0]}_masked.jpg"
            output_file = os.path.join(output_folder,f"{image_file.split('.')[0]}_masked.jpg")
            shutil.copy("output.jpg", output_file)
            os.remove("output.jpg")


        # Create a new in Clowder folder for the output images
        clowder_output_folder = pyclowder.datasets.create_folder(connector, host, secret_key, dataset_id, output_folder_name)
        filepaths = []
        for file in os.listdir(output_folder):
            print()
            # Append the full path to the filepaths list, current working directory is the dataset folder
            filepaths.append(os.path.join(output_folder,file))

        # Upload the results
        pyclowder.files.upload_multiple_files(connector, host, secret_key, dataset_id, filepaths, folder_id=clowder_output_folder)

        # Delete the symlink folder
        shutil.rmtree(dataset_folder)

        # Delete the output folder
        shutil.rmtree(output_folder)


        

if __name__ == "__main__":
    extractor = PDGInferenceExtractor()
    extractor.start()
    

            
        