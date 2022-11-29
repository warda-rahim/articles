
#!/bin/bash     
PROJECT_ID="gb-lab-us-binders-15"
REGION="europe-west2"
REPOSITORY="test"
IMAGE='training'
IMAGE_TAG='training:latest'

#gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG
docker build -t $IMAGE .
docker tag $IMAGE $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG
