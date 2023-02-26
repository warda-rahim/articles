import os
import sys
import json
import re
import logging
from typing import Tuple, List, Dict

import pandas as pd
from flask import Flask, request, abort
from google.cloud import storage

from predict import load_model, model_predict

logger = logging.getLogger("App")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

app = Flask(__name__)

PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
AIP_STORAGE_URI = os.environ["AIP_STORAGE_URI"]  # Vertex AI sets this env with path to the model artifact
logger.info(f"MODEL PATH: {AIP_STORAGE_URI}")

MODEL_PATH = "model/model.pickle"

# Creation of the Flask app
app = Flask(__name__)

def decode_gcs_url(url: str) -> Tuple[str, str, str]:
    """
        Split a google cloud storage path such as: gs://bucket_name/dir1/filename into
        bucket and path after the bucket: bucket_name, dir1/filename
        :param url: storage url
        :return: bucket_name, blob
        """
    bucket = re.findall(r'gs://([^/]+)', url)[0]
    blob = url.split('/', 3)[-1]
    return bucket, blob

def download_artifacts(artifacts_uri:str, local_path:str):
    logger.info(f"Downloading {artifacts_uri} to {local_path}")
    storage_client = storage.Client()
    src_bucket, src_blob = decode_gcs_url(artifacts_uri)
    source_bucket = storage_client.bucket(src_bucket)
    source_blob = source_bucket.blob(src_blob)
    source_blob.download_to_filename(local_path)
    logger.info(f"Downloaded.")

def load_artifacts(artifacts_uri:str=AIP_STORAGE_URI):
    model_uri = os.path.join(artifacts_uri, "model")
    logger.info(f"Loading artifacts from {model_uri}")
    download_artifacts(model_uri, MODEL_PATH)

# Flask route for Liveness checks
@app.route(HEALTH_ROUTE, methods=['GET'])
def health_check():
    return "I am alive, 200"

# Flask route for predictions
@app.route(PREDICT_ROUTE, methods=['POST'])
def prediction():
    logger.info("SERVING ENDPOINT: Received predict request.")
    
    load_artifacts()
    model = load_model(MODEL_PATH)
    logger.info(f"MODEL LOADED")
    payload = json.loads(request.data)

    instances = payload["instances"]

    try:
        df_str = "\n".join(instances)
        instances = pd.read_json(df_str, lines=True)
    except Exception as e:
        logger.error(f"Failed to process payload:\n {e}")
        abort(500, "Failed to score request.")

    logger.info("Running MODEL_PREDICT for request.")
    model_output, target_name = model_predict(model, instances)
    logger.info("MODEL_PREDICT completed.")

    response = {"predictions": model_output.tolist()}
    logger.info("SERVING ENDPOINT: Finished processing.")
    
    return response


if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0', port=8080)	

