import logging
import time
import json
import os


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
        
        job_id = self.job_submission_client.submit_job(
            # Entrypoint shell command to execute
            entrypoint=f"python inference_kuberay_script.py {host} {parameters['datasetId']} {secret_key}",
            # Path to the local directory that contains the script.py file
            runtime_env={"working_dir": "./",
                         "env_vars": {"CLOWDER_VERSION": "2",
                                      "MINIO_MOUNTED_PATH": os.getenv("MINIO_MOUNTED_PATH")}}
        )
        
        logger.info(f"Job submitted with ID: {job_id}")

        def wait_until_status(job_id, status_to_wait_for, timeout_seconds=10000):
            start = time.time()
            while time.time() - start <= timeout_seconds:
                status = client.get_job_status(job_id)
                print(f"status: {status}")
                if status in status_to_wait_for:
                    break
                time.sleep(1)

        wait_until_status(job_id, {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED})
        logs = client.get_job_logs(job_id)
        print(logs)

        # Finish
        logging.warning("Successfully extracted!")
        
        
        
if __name__ == "__main__":
    
    # Production with Kuberay
    # The address of the Ray cluster needs to be updated as needed
    # ray_client = JobSubmissionClient("http://clowder-raycluster-kuberay-head-svc.ibm-hpc.svc.cluster.local:8265")
    
    # Local Testing
    ray.init()
    ray_client = JobSubmissionClient("http://127.0.0.1:8265")

    extractor = IWPKubeRayInferenceExtractor(ray_client)
    extractor.start()