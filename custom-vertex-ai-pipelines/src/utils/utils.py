import matplotlib.pyplot as plt
import io
from google.cloud import storage
import base64


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
