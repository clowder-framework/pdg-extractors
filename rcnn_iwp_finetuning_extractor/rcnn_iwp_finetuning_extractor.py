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

from rcnn_iwp_training import invoke_main

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PDGFinetuningExtractor(Extractor):
    """Extractor that runs finetuning with a trained model on a given image to detect IWPs."""
    # TODO: This extractor is hard coded to use a specific model and dataset metadata. 
    # We need to make it more flexible
    def __init__(self):
        Extractor.__init__(self)
        self.setup()

        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)
    
    def check_message(self, connector, host, secret_key, resource, parameters):
        # Prevent downloading files by default
        return CheckMessage.bypass
    

    def process_message(self, connector, host, secret_key, resource, parameters):
        # Process the file and upload the results
        logger = logging.getLogger(__name__)
        dataset_id = resource['parent']['id']
        print(f"Dataset ID: {dataset_id}")

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

        # Load the model config file
        if "MODEL_CONFIG_FILE" in params:
            model_config_file_data = json.loads(params["MODEL_CONFIG_FILE"])  
        else:
            raise ValueError("MODEL_CONFIG_FILE is not provided in the parameters.")

        data ={}
        metadata = {}

        if "TRAIN_FOLDER" in params:
            data["train"] = json.loads(params["TRAIN_FOLDER"])
        else:
            raise ValueError("TRAIN_FOLDER is not provided in the parameters.")

        if "TRAIN_DATA_METADATA" in params:
            metadata["train"] = json.loads(params["TRAIN_DATA_METADATA"])
        else:
            raise ValueError("TRAIN_DATA_METADATA is not provided in the parameters.")

        if "VAL_FOLDER" in params:
            data["val"] = json.loads(params["VAL_FOLDER"])
        else:
            raise ValueError("VAL_FOLDER is not provided in the parameters.")

        if "VAL_DATA_METADATA" in params:
            metadata["val"] = json.loads(params["VAL_DATA_METADATA"])
        else:
            raise ValueError("VAL_DATA_METADATA is not provided in the parameters.")

        if "TEST_FOLDER" in params:
            data["test"] = json.loads(params["TEST_FOLDER"])
        else:
            raise ValueError("TEST_FOLDER is not provided in the parameters.")

        if "TEST_DATA_METADATA" in params:
            metadata["test"] = json.loads(params["TEST_DATA_METADATA"])
        else:
            raise ValueError("TEST_DATA_METADATA is not provided in the parameters.")

        for folder, folder_data in data.items():
            data[folder]["path"] = create_symlink_folder(host,secret_key,folder_data['datasetId'],folder_data["selectionID"],folder,"data/iwp")
        
        for file, file_data in metadata.items():
            metadata[file]["path"] = create_symlink_file(connector,host,secret_key,file_data["selectionID"],"data/iwp")

        config_file = create_symlink_file(connector,host,secret_key,model_config_file_data["selectionID"],"data")

        # Set json path environment variable
        os.environ['MODEL_CONFIG_FILE_PATH'] = config_file
        
        try:
            invoke_main()
            print("Finetuning complete.")
        except Exception as e:
            # Clean up if exists
            if os.path.exists("results/"):
                shutil.rmtree("results/")
            if os.path.exists("data/"):
                shutil.rmtree("data/")
            
            print(f"Error: {e}")
            raise e

        output_dir = "results/"
        #TODO: Hardcoded for now but need to be changed as the code evolves.
        model_name = "clowder_mask_rcnn_vitdet"
        dataset_name = "iwp"
        
        # Upload the results
        output_file = f"{output_dir}/{model_name}/{dataset_name}/model_final.pth"
        new_output_file = f"{output_dir}/{model_name}/{dataset_name}/model_weights.pth"
        os.rename(output_file, new_output_file)
        pyclowder.files.upload_to_dataset(connector, host, secret_key, dataset_id, new_output_file)

        # Clean up
        if os.path.exists("results/"):
            shutil.rmtree("results/")
        if os.path.exists("data/"):
            shutil.rmtree("data/")
        
    



        
if __name__ == "__main__":
    extractor = PDGFinetuningExtractor()
    extractor.start()
    

            
        