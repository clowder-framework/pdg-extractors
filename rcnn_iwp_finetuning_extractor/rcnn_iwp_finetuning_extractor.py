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

from clowder_utils import create_symlink_folder
import sys
import os

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
        if "MODEL_CONFIG_FILE_ID" in params:
            model_config_file_data = json.loads(params["MODEL_CONFIG_FILE_ID"])  
        else:
            raise ValueError("MODEL_CONFIG_FILE_ID is not provided in the parameters.")

        data ={}

        if "TRAIN_FOLDER" in params:
            data["train"] = json.loads(params["TRAIN_FOLDER"])
        else:
            raise ValueError("TRAIN_FOLDER is not provided in the parameters.")

        if "VAL_FOLDER" in params:
            data["val"] = json.loads(params["VAL_FOLDER"])
        else:
            raise ValueError("VAL_FOLDER is not provided in the parameters.")

        if "TEST_FOLDER" in params:
            data["test"] = json.loads(params["TEST_FOLDER"])
        else:
            raise ValueError("TEST_FOLDER is not provided in the parameters.")

        for folder, folder_data in data.items():
            data[folder]["path"] = create_symlink_folder(host,secret_key,folder_data['datasetId'],folder_data["selectionID"],folder,"data")
        
        print(data)

        create_symlink_file(host,secret_key,model_config_file_data['datasetId'],model_config_file_data["selectionID"],model_config_file_data["name"],"data")


            
        
        
        print(f"Model config file: {model_config_file_id}")
        print(f"Train folder: {train_folder}")
        print(f"Val folder: {val_folder}")
        print(f"Test folder: {test_folder}")



        
if __name__ == "__main__":
    extractor = PDGFinetuningExtractor()
    extractor.start()
    

            
        