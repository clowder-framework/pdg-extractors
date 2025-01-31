FROM minio-mounted-clowder-extractor 

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

WORKDIR /extractor

COPY requirements.txt ./

# Install torch first, separately, to avoid the metadata issue
RUN pip install --no-cache-dir torch==2.2.1 torchvision==0.17.1


RUN pip install -r requirements.txt

COPY . .


# Clone pyclowder, install specific branch, and return to working directory
RUN git clone https://github.com/clowder-framework/pyclowder.git && \
    cd pyclowder && \
    git checkout 116-modify-download-apis-for-minio-mounted-fs && \
    python setup.py install && \
    cd .. && \
    rm -rf pyclowder

# CMD ["python", "-u","rcnn_iwp_inference_extractor.py"]
CMD ["python", "-u","rcnn_iwp_finetuning_extractor.py"]
