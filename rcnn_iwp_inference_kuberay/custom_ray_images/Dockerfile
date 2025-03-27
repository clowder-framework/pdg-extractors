FROM rayproject/ray:2.43.0-py310

USER root

RUN apt-get update && apt-get install -y \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Install torch first, separately, to avoid the metadata issue
RUN pip install --no-cache-dir torch==2.2.1 torchvision==0.17.1

# Install cv2 headless
RUN pip install opencv-python-headless==4.11.0.86

RUN pip install -r requirements.txt

# Clone pyclowder, install specific branch, and return to working directory
RUN git clone https://github.com/clowder-framework/pyclowder.git && \
    cd pyclowder && \
    git checkout 118-create-apis-to-upload-folders-and-uploading-files-to-folder && \
    python setup.py install && \
    cd .. && \
    rm -rf pyclowder
