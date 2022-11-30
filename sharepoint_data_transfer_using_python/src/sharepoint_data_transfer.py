import os
import pandas as pd
from office365.sharepoint.files.file import File
import io
import errno

from src.utils import connect_to_sharepoint
from src.config import SITE_URL, CLIENT_ID, CLIENT_SECRET, SHAREPOINT_FOLDER_NAME, LOCAL_FILE_PATH, WRITE_FILE_NAME, SHAREPOINT_FOLDER_RELATIVE_PATH, READ_FILE_NAME

import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(name)s-%(levelname)s-%(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]


CONTEXT = connect_to_sharepoint(SITE_URL, CLIENT_ID, CLIENT_SECRET)


def download_from_sharepoint(remote_path:str, context=CONTEXT, local_data_dir:str='./'):
    """Access sharepoint with credentials and get file
    Args:
        remote_path (str): Sharepoint url of file to be downloaded. Example: "/sites/team/Shared Documents/data/refreshed_data.csv"
        ctx (office365.sharepoint.client_context.ClientContext): Authentication for sharepoint connection. Defaults to connect_to_sharepoint(SITE_URL, CLIENT_ID, CLIENT_SECRET).
        local_data_dir (str): Path to local folder where file will be downloaded. Defaults to './'
     """

    download_path = os.path.join(local_data_dir, os.path.basename(remote_path))
    
    logger.info(f"DOWNLOADING FILE FROM SHAREPOINT...")
    with open(download_path, "wb") as local_file:
        myfile = (context.web.get_file_by_server_relative_path(remote_path)
                 .download(local_file)
                 .execute_query()
                 )
    logger.info(f"FILE HAS BEEN DOWNLOADED INTO: {download_path}")



def upload_file_to_sharepoint(context=CONTEXT, dir_name:str=SHAREPOINT_FOLDER_NAME, local_file_path:str=LOCAL_FILE_PATH):
    """Access sharepoint with credentials and upload file
    Args:
        context (office365.sharepoint.client_context.ClientContext): Authentication for sharepoint connection. Defaults to connect_to_sharepoint(SITE_URL, CLIENT_ID, CLIENT_SECRET).
        dir_name (str): Sharepoint folder name where the file is uploaded. Defaults to SHAREPOINT_FOLDER_NAME.
        local_file_name (str): Local file path of file to be uploaded to sharepoint. Defaults to LOCAL_FILE_PATH.
    """

    target_folder = context.web.get_folder_by_server_relative_url(
        f"Shared Documents/{dir_name}"
    )

    logger.info("UPLOADING FILE TO SHAREPOINT...")
    with open(local_file_path, "rb") as content_file:
        file_content = content_file.read()
        target_folder.upload_file(os.path.basename(local_file_path), file_content).execute_query()
    logger.info(f"FILE HAS BEEN UPLOADED TO SHAREPOINT INTO: {dir_name}")



def upload_dataframe_to_sharepoint(df:pd.DataFrame, context=CONTEXT, dir_name:str=SHAREPOINT_FOLDER_NAME, file_name:str=WRITE_FILE_NAME):
    """Access sharepoint with credential and upload dataframe.
    Args:
        df (pd.DataFrame): Pandas dataframe to be written to sharepoint.
        context (office365.sharepoint.client_context.ClientContext): Authentication for sharepoint connection. Defaults to connect_to_sharepoint(SITE_URL, CLIENT_ID, CLIENT_SECRET).
        dir_name (str): Sharepoint folder name where the data is uploaded. Defaults to SHAREPOINT_FOLDER_NAME.
        file_name (str): Sharepoint filename with the uploaded data. Defaults to WRITE_FILE_NAME.
    """
    
    target_folder = context.web.get_folder_by_server_relative_url(
        f"Shared Documents/{dir_name}"
    )

    logger.info("UPLOADING DATA TO SHAREPOINT...")
    
    buffer = io.BytesIO() #create a buffer object
    df.to_csv(buffer, index=False) #write the dataframe to the buffer
    buffer.seek(0) #go to the start of the stream
    file_content = buffer.read() 
    target_folder.upload_file(file_name, file_content).execute_query()
    
    logger.info(f"DATAFRAME HAS BEEN UPLOADED TO SHAREPOINT INTO: {'/'.join([dir_name, file_name])}")



def read_sharepoint_file(context=CONTEXT, folder_relative_path:str=SHAREPOINT_FOLDER_RELATIVE_PATH, file_name:str=READ_FILE_NAME) -> pd.DataFrame:
    """Access sharepoint with credentials and read file into a pandas DataFrame.
    Args:
        context (office365.sharepoint.client_context.ClientContext): Authentication for sharepoint connection. Defaults to connect_to_sharepoint(SITE_URL, CLIENT_ID, CLIENT_SECRET).
        folder_relative_path (str): Sharepoint folder relative path. Defaults to SHAREPOINT_FOLDER_RELATIVE_PATH.
        filename (str): Sharepoint file to be read. Defaults to READ_FILE_NAME.
    Returns:
        pd.DataFrame
    """
    
    response = File.open_binary(context, '/'.join([folder_relative_path, file_name]))
    df = pd.read_csv(io.BytesIO(response.content))
    if df.shape[0] == 0:
        raise FileNotFoundError(f'Either {file_name} does not exist or path to the file is wrong.')
    else:
        return df
    
