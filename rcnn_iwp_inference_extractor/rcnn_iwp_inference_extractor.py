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


        # Download the model file
        model_file = pyclowder.files.download(connector, host, secret_key, fileid=MODEL_FILE_ID)
        image_file = file_path


        print(f"Model file: {model_file}")
        print(f"Image file: {image_file}")

        # Load the model and metadata
        model = load_model("configs/mask_rcnn_vitdet.py", "configs/datasets/iwp.py", model_file)
        metadata = get_metadata("iwp_test", "data/iwp/iwp_test.json")

        # Run inference on the image
        coco_mask_metadata = run_inference(model, image_file, metadata, threshold=CONFIDENCE_THRESHOLD)

        metadata = {"COCO Bounding Boxes": coco_mask_metadata}
        # post metadata to Clowder
        metadata = self.get_metadata(metadata, 'file', file_id, host)

        # Update the metadata of file
        pyclowder.files.upload_metadata(connector, host, secret_key, file_id, metadata)

        # Change the name of the output image to the name of the input image with _masked appended
        output_file = f"{file_name}_masked.jpg"
        shutil.copy("output.jpg", output_file)
        os.remove("output.jpg")

        # Upload the results
        pyclowder.files.upload_to_dataset(connector, host, secret_key, dataset_id, output_file)

        # Delete the output file
        os.remove(output_file)
        

if __name__ == "__main__":
    extractor = PDGInferenceExtractor()
    extractor.start()
    

            
        