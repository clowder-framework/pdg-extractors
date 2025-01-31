import pyclowder.files as files
from pyclowder.connectors import Connector
import requests
import os
import cv2

# api/v2/datasets/6787069a52b726ff197a708b/folders_and_files?folder_id=678707cc52b726ff197a7223
def get_folder_files(host,key,dataset_id,folder_id):
    # TODO: Hardcoded limit of 3000 files, need to change to dynamic
    url = f"{host}/api/v2/datasets/{dataset_id}/folders_and_files?folder_id={folder_id}&limit=3000"
    headers = {"X-API-KEY": key}
    res = requests.get(url,headers=headers)
    files =[]
    for file in res.json()['data']:
        file_id = file['id']
        name = file['name']
        files.append({'id':file_id,'name':name})
    return files


def create_symlink_folder(host,key,dataset_id,folder_id,folder_name, folder_path ):
    folder_path = f"{folder_path}/{folder_name}"
    os.makedirs(folder_path, exist_ok=True)
    if os.environ.get("$MINIO_MOUNTED_PATH") == "":
        print("Minio mounted path not found")
        return
    minio_path = os.environ.get("MINIO_MOUNTED_PATH")
    files = get_folder_files(host,key,dataset_id,folder_id)
    for file in files:
        file_id = file['id']
        file_name = file['name']
        file_path = f"{minio_path}/{file_id}"
        os.symlink(file_path, f"{folder_path}/{file_name}")
    print(f"Created symlink folder {folder_path}")




if __name__ == "__main__":
    CLOWDER_URL = ""
    API_KEY = ""
    FILE_ID = ""
    connector = Connector(extractor_name="",extractor_info={}, clowder_url=CLOWDER_URL)

    create_symlink_folder(CLOWDER_URL,API_KEY,"6787069a52b726ff197a708b","678707cc52b726ff197a7223","test","data")
    img = cv2.imread("data/test/FID_493_Polygon_2.jpg")
    print(img.shape)