FROM python:3.9-slim-bullseye

COPY ./requirements-docker.txt /usr/requirements.txt

WORKDIR /usr

RUN pip3 install -r requirements.txt

COPY ./src /usr/src
COPY ./models /usr/models

# Service must listen to $PORT environment variable.
# This default value facilitates local development.
ENV PORT 5000
ENV BASIC_AUTH_USERNAME Admin
ENV BASIC_AUTH_PASSWORD 123

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 --chdir /usr/src main:app