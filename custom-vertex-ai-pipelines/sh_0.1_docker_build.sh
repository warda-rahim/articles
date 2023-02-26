#!/bin/bash     
PROJECT_ID=<PROJECT_NAME>
REGION="europe-west2"
REPOSITORY="houseprice"
IMAGE='training'
IMAGE_TAG='training:latest'

docker build -t $IMAGE .
docker tag $IMAGE $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG
