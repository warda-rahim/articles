#!/bin/bash     
PROJECT_ID=<PROJECT_NAME>
REGION="europe-west2"
REPOSITORY="houseprice"
IMAGE_TAG='serving_image:latest'

#Create repository in the artifact registry
gcloud beta artifacts repositories create $REPOSITORY \
  --repository-format=docker \
  --location=$REGION
 
# Configure Docker
gcloud auth configure-docker $REGION-docker.pkg.dev
 
 # Push
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG
