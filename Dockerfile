# Use Ubuntu 20.04 as base image (from 2020)
FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    git \
    pkg-config \
    libhdf5-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    zlib1g-dev \
    libjpeg-dev \
    cython \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Upgrade pip and install build tools with compatible versions
RUN pip3 install --no-cache-dir --upgrade pip==21.3.1 setuptools==57.5.0 wheel

# Install Cython first (2019-2020 era version)
RUN pip3 install --no-cache-dir cython==0.29.24

# Install core packages in dependency order
RUN pip3 install --no-cache-dir numpy==1.20.3
RUN pip3 install --no-cache-dir h5py==2.10.0
RUN pip3 install --no-cache-dir scipy==1.5.4
RUN pip3 install --no-cache-dir matplotlib==3.3.4
RUN pip3 install --no-cache-dir pandas==1.3.5
RUN pip3 install --no-cache-dir xarray==0.19.0
RUN pip3 install --no-cache-dir typing-extensions==4.6.0
RUN pip3 install --no-cache-dir theano==1.0.5
RUN pip3 install --no-cache-dir pymc3==3.9.3
RUN pip3 install --no-cache-dir arviz==0.11.0
RUN pip3 install --no-cache-dir pymc3-ext==0.1.0
RUN pip3 install --no-cache-dir starry==1.0.0

# Install additional packages needed for the scripts
RUN pip3 install --no-cache-dir corner==2.2.1
RUN pip3 install --no-cache-dir exoplanet-core==0.1.2
RUN pip3 install --no-cache-dir emcee==3.1.4
RUN pip3 install --no-cache-dir tqdm==4.62.3
RUN pip3 install --no-cache-dir netCDF4==1.5.7
RUN pip3 install --no-cache-dir dill==0.3.4
RUN pip3 install --no-cache-dir astropy==4.2.1
RUN pip3 install --no-cache-dir exoplanet==0.4.4

RUN pip3 install --no-cache-dir numpy==1.20.3
RUN pip3 install --no-cache-dir scipy==1.6.0
RUN pip3 install --no-cache-dir pandas==1.3.5
RUN pip3 install --no-cache-dir xarray==0.19.0
RUN pip3 install --no-cache-dir matplotlib==3.3.4
RUN pip3 install --no-cache-dir exceptiongroup==1.0.0

# Set working directory
WORKDIR /app

# Command to run when container starts
CMD ["/bin/bash"] 
