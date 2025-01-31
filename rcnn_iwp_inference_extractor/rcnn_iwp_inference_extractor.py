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
    
    def check_message(self, connector, host, secret_key, resource, parameters):
        # Prevent downloading files by default
        return CheckMessage.bypass

    def process_message(self, connector, host, secret_key, resource, parameters):
        # Process the file and upload the results

        logger = logging.getLogger(__name__)
        
        file_path = resource["local_paths"][0]
        file_id = resource['id']
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

        if "MODEL_FILE_ID" in params:
            MODEL_FILE_ID = params["MODEL_FILE_ID"]
        else:
            raise ValueError("MODEL_FILE_ID is not provided in the parameters.")
        
        if "CONFIDENCE_THRESHOLD" in params:
            CONFIDENCE_THRESHOLD = params["CONFIDENCE_THRESHOLD"]


        # Download the model file
        model_file = pyclowder.files.download(connector, host, secret_key, fileid=MODEL_FILE_ID)
        image_file = file_path


        print(f"Model file: {model_file}")
        print(f"Image file: {image_file}")

        # Load the model and metadata
        model = load_model("configs/mask_rcnn_vitdet.py", "configs/datasets/iwp.py", model_file)
        metadata = get_metadata("iwp_test", "data/iwp/iwp_test.json")

        # Run inference on the image
        run_inference(model, image_file, metadata, threshold=CONFIDENCE_THRESHOLD)

        # Upload the results
        output_file = "output.jpg"
        pyclowder.files.upload_to_dataset(connector, host, secret_key, dataset_id, output_file)

if __name__ == "__main__":
    extractor = PDGInferenceExtractor()
    extractor.start()
    

            
        