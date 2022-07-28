FROM tensorflow/serving:latest

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN python3 -m pip install --upgrade pip
RUN pip install tensorflow


USER root
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY tt_license_plates_model_best_new_data_tf_saved_model /home/

