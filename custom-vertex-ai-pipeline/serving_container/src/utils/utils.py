import matplotlib.pyplot as plt
import io
from google.cloud import storage
import base64

IMAGE_PATH = 'images/test.png'


def save_fig_to_bucket(bucket_name:str, image_path:str=IMAGE_PATH):
    """Saves a png image file to GCP Storage Bucket"""
    fig_to_upload = plt.gcf()

    # save figure image to a bytes buffer
    buf = io.BytesIO()
    fig_to_upload.savefig(buf, format='png')

    # init GCS client and upload buffer contents
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(image_path)  
    blob.upload_from_file(buf, content_type='image/png', rewind=True)


def get_image_data():
    """Gets binary data url for a matplotlib figure"""
    image = plt.gcf()
    buf = io.BytesIO()
    # save image to memory
    image.savefig(buf, format='png', bbox_inches="tight")
    binary_image = buf.getvalue()

    image_base64_utf8_str = base64.b64encode(binary_image).decode('utf-8')
    image_type = "png"
    dataurl = f'data:image/{image_type};base64,{image_base64_utf8_str}'

    return dataurl
