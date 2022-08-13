from OCR.ocr_handler import OCR 
import requests
from PIL import Image
import cv2 
import tensorflow as tf
import sys
from time import time
import numpy as np
"""
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
"""
# used to handle both plate recog and ocr models whilst using video input
class detect_predict_license_plate:
    def __init__(self, lP_endpt,ocr_endpt,threshold,gstreamer,outfile) -> None:
        self.plate_recog_endpoint = lP_endpt
        self.plate_threshold = threshold
        self.frame = 'car.jpg'
        # ocr handler -> placed here so that it would load once
        self.ocr_endpt = ocr_endpt
        self.OCR_HANDLER = OCR(self.ocr_endpt)
        # video stream
        self.gstreamer = gstreamer
        self.outfile = outfile
        return

    # checks the confidence of plate detection -> used to trigger OCR recog
    def is_plate(self, conf):
        return True if conf >= self.plate_threshold else False

    # crop the image based on the coordinates returned by license plate model. This allows the ocr to properly predict
    def crop_image(self, response):
        # get image
        img = self.read_image_array()
        
        # get coordinate
        x1, x2, y1, y2 = self.box_coordinates(response)

        #crop cv2 image
        img_crop = img[y1:y2,x1:x2]
     
        #save cropped image
        cv2.imwrite('cropped_car.jpg',img_crop)
        return img_crop
    
    # takes the response and provides coordinates of plate -> needed to crop image whihc is then passed to ocr model
    def box_coordinates(self, response):
        x1 = int(response[0])
        x2 = int(response[2])
        y1 = int(response[1])
        y2 = int(response[3])
        return x1, x2, y1, y2

    def read_image_array(self):
        return cv2.imread(self.frame)
    
    # used to convert image to byte for predictions
    def read_image_byte(self):
        return open(self.frame, 'rb').read()
     
    # detect the license plate from image
    def find_license_plate(self):
        # cconvert image to bytes for processing
        img = self.read_image_byte()

        # post request ot the server      
        response = requests.post(self.plate_recog_endpoint, data=img)

        try:      
            return True, response.json()['predictions'][0]
        except:
            return False,response.json()['predictions']
        
    
    def get_video(self):
        """
        Function creates a streaming object to read the video from the webcam frame by frame.
        :param self:  class object
        :return:  OpenCV object to stream video frame by frame.
        """
        cap = cv2.VideoCapture(self.gstreamer)
        assert cap is not None
        return cap

    def get_frame(self,video):
        # read the 29th image in the sequence - ensures that the sytsmte is predicting on the most upto date image
        for i in range(28):
            ret, frame = video.read()
            # save video frame as well
            cv2.imwrite("car.jpg", frame)
        if not ret:
            return false, []
        return True, frame

    # generates the ocr reading based on the response of the plate recog model
    def ocr_generate(self,response):

        # get the conf of the readings
        conf = response[4]

        # check to see if plate is detected
        if self.is_plate(conf):
            # crop image before sending to ocr
            self.crop_image(response)

            # preprocessing for ocr reader
            img = self.OCR_HANDLER.encode_single_sample(img = 'cropped_car.jpg')
            img = img["image"]
            
            # send to server to predict
#            print("Predicting..")
            result = self.OCR_HANDLER.post(img)
            return result
        else:
#            print("No plate detected")
            return []
    
    def set_saved_video(self, output_video, size):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        video = cv2.VideoWriter(output_video, fourcc, 29, size)
        return video
        
    
    def __call__(self):
        # create videpo object
        video = self.get_video()

        # write frames to video file
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.outfile,four_cc,20,(960,1080))
        savevideo = cv2.VideoWriter(self.outfile, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (960,540))

        # check to see if the video object is opened and ready to use
        if video.isOpened():
            while True:
                # get the frame to perform analysis on 
                is_frame, frame = self.get_frame(video)
                savevideo.write(frame)
                
                # check if the frame was read sucessfylly
                if is_frame:
                    # check the time to for plate model\
                    t_plate1 = time()
                    # check frame for license plate
                    is_plate, response = self.find_license_plate()
                    t_plate2 = time()
                    print("\nTime taken for plate model: ", 1/np.round(t_plate2-t_plate1,3))
                      # display response once plate isdtected
                   
                      
                    # check time  for OCR model
                    if is_plate:
                        print("Plate Prediction Acc: ", response[4]*100,"%")
                        # # generate ocr reading (if needed)
                        ocr_reading = self.ocr_generate(response)
                        if ocr_reading:
                          print(ocr_reading)
                    t_ocr = time()

                    t_ocr = 0 if np.round(t_ocr-t_plate2)==0 else 1/np.round(t_ocr-t_plate2,3)
                    print("Time taken for OCR model: ", t_ocr)
                    print("Total Time: ", t_ocr + 1/round(t_plate2-t_plate1,3))
                     
                else:
                    print("The frame was not properly read")
        return

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


if __name__ == "__main__":
    # define inputs to detect and predict class
    lP_endpt = 'https://wide-llamas-sing-170-246-161-114.loca.lt/predictions/license_plate'
    threshold = 0.5
    ocr_endpt = 'https://small-dolls-hear-170-246-161-114.loca.lt/v1/models/ocr_model/versions/2:predict'
    # get output file to store video
    outfile = sys.argv[1]
    # generate the stream for predictions
    gstreamer = gstreamer_pipeline()
    # instantiate class
    car = detect_predict_license_plate(lP_endpt,ocr_endpt,threshold,gstreamer,outfile)
    # print(car.find_license_plate())
    car()
Footer

