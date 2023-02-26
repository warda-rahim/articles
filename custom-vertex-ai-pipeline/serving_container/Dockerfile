FROM mirror.gcr.io/library/python:3.8

RUN mkdir -p /app/model
WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py
COPY predict.py /app/predict.py
COPY src /app/src

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn --bind :$AIP_HTTP_PORT --log-level info --workers 1 --timeout 90 app:app
