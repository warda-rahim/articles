
#!/bin/bash     
PROJECT_ID="gb-lab-us-binders-15"
REGION="europe-west2"
REPOSITORY="test"
IMAGE_TAG='training:latest'
KMS_KEY='projects/$PROJECT_ID/locations/$REGION/keyRings/project-services/cryptoKeys/gb-key-artifact-registry'

#Create repository in the artifact registry
gcloud beta artifacts repositories create $REPOSITORY \
  --repository-format=docker \
  --location=$REGION
  --kms-key=$KMS_KEY
 
# Configure Docker
gcloud auth configure-docker $REGION-docker.pkg.dev
 
 # Push
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG
