##
import os
import cv2
import numpy as np
os.system('sudo pip install gdown')
import gdown
import zipfile
from zipfile import ZipFile
import tarfile
import requests

DIR = os.getcwd()

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    file_id = 'TAKE ID FROM SHAREABLE LINK'
    destination = 'DESTINATION FILE ON YOUR DISK'
    download_file_from_google_drive(file_id, destination)


download_file_from_google_drive('11j_HC40a2t2q9ZtsLp4OPwF2xzhJCG1i', os.path.join(DIR, 'faces.tar.gz'))
download_file_from_google_drive('1tpW7ZVNosXsIAWu8-f5EpwtF3ls3pb79', os.path.join(DIR, 'extra_data.zip'))

filename = "faces.tar.gz"
tf = tarfile.open(filename)
tf.extractall('./')

with ZipFile('extra_data.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()
