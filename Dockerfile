###############################################################################
# Base image: PyTorch with CUDA 11.7 and cuDNN 8
# Provides GPU-enabled environment suitable for training and inference.
###############################################################################
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Non-interactive mode for package installation
ENV DEBIAN_FRONTEND=noninteractive

# Working directory inside the container
WORKDIR /workspace


###############################################################################
# System dependencies required for scientific Python stack and OpenCV
###############################################################################
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*


###############################################################################
# Python dependencies required by STU-Net, nnU-Net v2, preprocessing and eval
###############################################################################
RUN pip install --upgrade pip && \
    \
    # Core scientific libraries
    pip install "numpy<2" scipy scikit-image nibabel opencv-python tqdm pyyaml matplotlib pandas && \
    \
    # nnU-Net v2 training and inference stack
    pip install SimpleITK batchgenerators==0.25 connected-components-3d && \
    pip install nnunetv2==2.2 && \
    \
    # STU-Net model utilities
    pip install torchinfo timm==0.6.12 gdown


###############################################################################
# Project source code
###############################################################################
COPY . .


###############################################################################
# nnU-Net v2 environment variables
# Required directory structure for dataset, preprocessing and results.
###############################################################################
ENV nnUNet_raw="/workspace/nnUNet_raw"
ENV nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
ENV nnUNet_results="/workspace/nnUNet_results"


###############################################################################
# Default container entrypoint
###############################################################################
CMD ["bash"]