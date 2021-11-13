FROM python:3.9-slim-bullseye

COPY ./requirements-docker.txt /usr/requirements.txt

WORKDIR /usr

RUN pip3 install -r requirements.txt

COPY ./src /usr/src
COPY ./models /usr/models

# Service must listen to $PORT environment variable.
# This default value facilitates local development.
ENV PORT 5000

ENV API_KEY 123

COPY ./.gcp /usr/.gcp
ENV GOOGLE_APPLICATION_CREDENTIALS ../.gcp/mlops-deploy-331313-68d916ea4645.json

# Run the web service on container startup. 
# For gunicorn webserver, workers should match the amount of CPU cores
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 --chdir /usr/src main:app