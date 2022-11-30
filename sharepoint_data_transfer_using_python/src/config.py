import os
from dotenv import load_dotenv


# For uploading/downloading files to and from sharepoint
SITE_URL = "https://<CompanyUrl>.sharepoint.com/sites/team"
SHAREPOINT_FOLDER_NAME = "data"  #sharepoint folder where file is uploaded
LOCAL_FILE_PATH = "./data/refreshed_data.csv" #local file path of file uploaded to sharepoint
WRITE_FILE_NAME = "refreshed_data.csv" #file created on sharepoint to write dataframe into 
SHAREPOINT_FOLDER_RELATIVE_PATH = "/sites/team/Shared Documents/data" #sharepoint folder relative path from where file is read
READ_FILE_NAME = "historical_data.csv" #sharepoint file read into a dataframe

# Assume you have a .env file at project level with client_id and secret for sharepoint connection
load_dotenv()
CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET = os.environ['CLIENT_SECRET']
