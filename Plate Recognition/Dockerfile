FROM pytorch/torchserve:latest-cpu

WORKDIR /home/model-server/
#FROM pytorch/torchserve:latest-gpu
# install dependencies
RUN python3 -m pip install --upgrade pip
RUN pip3 install transformers
RUN pip3 install opencv-python
RUN pip3 install yolov5
RUN pip3 install torch==1.8.1 torchvision==0.9.1
RUN pip3 install torchserve==0.6.0
RUN pip3 install torch-model-archiver==0.6.0

USER root
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# copy model artifacts, custom handler and other dependencies
#COPY MyHandler.py /home/model-server/tmp
COPY my_handler.py /home/model-server/
COPY best.pt /home/model-server/
COPY requirements.txt /home/model-server/
#RUN pip3 install -r requirements.txt

# COPY ./model/$APP_NAME/ /home/model-server/
# create torchserve configuration file
#USER root
#RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
#RUN printf "\ninference_address=http://0.0.0.0:7080" >> /home/model-server/config.properties
#RUN printf "\nmanagement_address=http://0.0.0.0:7081" >> /home/model-server/config.properties

USER model-server
# expose health and prediction listener ports from the image
EXPOSE 8080
EXPOSE 8081


# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver --model-name license_plate --version=1.0 --serialized-file best.pt --handler my_handler.py\
 --export-path model-store -f

# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "license_plate=license_plate.mar", \
     "--model-store", \
     "/home/model-server/model-store"]