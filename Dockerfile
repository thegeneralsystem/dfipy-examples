# https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html

FROM python:3.11-slim

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    gcc g++ libpq-dev gdal-bin libgdal-dev python3-dev npm

# install the notebook package
RUN pip install --no-cache --upgrade pip && \
    pip install --no-cache notebook jupyterlab

# create user with a home directory
ARG NB_USER=binder
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}
USER ${USER}

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

# Install Python libraries and setup JupyterLab extensions
RUN pip install --no-cache-dir --user -r requirements.txt
