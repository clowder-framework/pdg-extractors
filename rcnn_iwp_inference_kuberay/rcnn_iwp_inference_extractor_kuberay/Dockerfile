FROM python:3.10

WORKDIR /extractor

RUN pip install ray[default]==2.44.0

# Clone pyclowder, install specific branch, and return to working directory
RUN git clone https://github.com/clowder-framework/pyclowder.git && \
    cd pyclowder && \
    git checkout 118-create-apis-to-upload-folders-and-uploading-files-to-folder && \
    pip install --no-cache-dir -r requirements.txt && \
    python setup.py install && \
    cd .. && \
    rm -rf pyclowder

COPY . .

CMD ["python", "-u","rcnn_iwp_inference_kuberay_extractor.py"]

