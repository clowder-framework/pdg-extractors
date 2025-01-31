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

def get_file_name(connector,host,key,file_id):
    res = files.download_summary(connector=connector,host=host, key=key, fileid=file_id)
    res = {
        "name":res['name'],
        "id":file_id,
    }
    return res

def create_symlink_folder(host,key,dataset_id,folder_id,folder_name, folder_path ):
    if os.environ.get("$MINIO_MOUNTED_PATH") == "":
        print("Minio mounted path not found")
        return
    folder_path = f"{folder_path}/{folder_name}"
    os.makedirs(folder_path, exist_ok=True)
    minio_path = os.environ.get("MINIO_MOUNTED_PATH")
    files = get_folder_files(host,key,dataset_id,folder_id)
    for file in files:
        file_id = file['id']
        file_name = file['name']
        source_file_path = f"{minio_path}/{file_id}"
        os.symlink(source_file_path, f"{folder_path}/{file_name}")
    return folder_path


def create_symlink_file(host,key,file_id,file_path):
    if os.environ.get("$MINIO_MOUNTED_PATH") == "":
        print("Minio mounted path not found")
        return
    file_name = get_file_name(connector,host,key,file_id)['name']
    minio_path = os.environ.get("MINIO_MOUNTED_PATH")
    source_file_path = f"{minio_path}/{file_id}"
    os.symlink(source_file_path, f"{file_path}/{file_name}")
    return file_path

