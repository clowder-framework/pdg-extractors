import logging
import time
import json
import os
from typing import Dict
import random
import string
import shutil

import pyclowder.files
from pyclowder.extractors import Extractor
import ray
from ray.job_submission import JobSubmissionClient, JobStatus

class IWPKubeRayInferenceExtractor(Extractor):
    """
    Extractor that runs inference with a trained model on a given folder of images to detect IWPs.
    This extractor uses Ray to run inference on a folder of images in parallel by submitting tasks to an actor pool.
    """
    def __init__(self, job_submission_client):
        Extractor.__init__(self)
        self.setup()
        self.job_submission_client = job_submission_client
        logging.getLogger('pyclowder').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)
        
    def process_message(self, connector, host, secret_key, resource, parameters):
        # Process the file and upload the results
        logger = logging.getLogger(__name__)
        dataset_id = resource['id']

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
        
        folder_name = dataset_folder_metadata["selectionName"]
        folder_id = dataset_folder_metadata["selectionID"]


        # Create a random 5 character string
        submission_id = ''.join(random.choices(string.ascii_letters + string.digits, k=5))

        job_id = self.job_submission_client.submit_job(
            # Entrypoint shell command to execute
            entrypoint=f"python inference_kuberay_script.py {host} {secret_key} {dataset_id} {folder_id} {MODEL_FILE_ID} {CONFIDENCE_THRESHOLD} {submission_id}",
            # Path to the local directory that contains the script.py file
            runtime_env={"working_dir": "./",
                         "env_vars": {"CLOWDER_VERSION": "2",
                                      "MINIO_MOUNTED_PATH":os.getenv("MINIO_MOUNTED_PATH")}}
        )
        
        logger.info(f"Job submitted with ID: {job_id}")

        def wait_until_status(job_id, status_to_wait_for, timeout_seconds=10000):
            start = time.time()
            while time.time() - start <= timeout_seconds:
                status = self.job_submission_client.get_job_status(job_id)
                print(f"status: {status}")
                if status in status_to_wait_for:
                    break
                time.sleep(1)

        wait_until_status(job_id, {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED})
        logs = self.job_submission_client.get_job_logs(job_id)
        print(logs)

        # Finish
        logging.warning("Inference completed! Uploading results to Clowder...")
        
        # Create a new folder for the output images
        output_folder_name = f"{folder_name}_masked"
        clowder_output_folder = pyclowder.datasets.create_folder(connector, host, secret_key, dataset_id, output_folder_name)

        # Read the results from the MINIO_MOUNTED_PATH
        results_json_path = os.path.join(os.getenv("MINIO_MOUNTED_PATH"), f"iwp_detection_results_{submission_id}.json")
        results_json = json.load(open(results_json_path))

        file_paths = []
        # Loop through the results json
        # Upload the metatadata to the file
        # Upload the image to the new folder
        for file_id in results_json:
            file_path = os.path.join(os.getenv("MINIO_MOUNTED_PATH"), submission_id, results_json[file_id]["output_file_name"])
            metadata = results_json[file_id]["coco_mask_metadata"]

            # Upload the metadata to the file
            metadata = self.get_metadata({"COCO Bounding Boxes": metadata}, 'file', file_id, host)
            pyclowder.files.upload_metadata(connector, host, secret_key, file_id, metadata)
            file_paths.append(file_path)

        # Upload the results to the new folder
        pyclowder.files.upload_multiple_files(connector, host, secret_key, dataset_id, file_paths, folder_id=clowder_output_folder)

        # Delete the submission folder
        shutil.rmtree(os.path.join(os.getenv("MINIO_MOUNTED_PATH"), submission_id))
        
        # Clean up the results json
        os.remove(results_json_path)
        
        
if __name__ == "__main__":
    
    # Local Testing
    # ray.init() 

    # The address of the Ray cluster stored in the environment variable
    #  KubeRay Cluster URL: http://clowder-raycluster-kuberay-head-svc.ibm-hpc.svc.cluster.local:8265
    # Local Testing URL: http://127.0.0.1:8265

    ray_client = JobSubmissionClient(os.getenv("RAY_CLUSTER_URL"))

    extractor = IWPKubeRayInferenceExtractor(ray_client)
    extractor.start()