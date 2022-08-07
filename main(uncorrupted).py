from OCR.ocr_handler import OCR 
import requests
from PIL import Image
import cv2 
import tensorflow as tf
import sys

# used to handle both plate recog and ocr models whilst using video input
class detect_predict_license_plate:
    def __init__(self, lP_endpt,ocr_endpt,threshold,gstreamer,outfile) -> None:
        self.plate_recog_endpoint = lP_endpt
        self.plate_threshold = threshold
        self.frame = 'car.jpg'
        # ocr handler -> placed here so that it would load once
        self.OCR_HANDLER = OCR()
        self.ocr_endpt = ocr_endpt
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
    def find_license_plate(self,data):
        response = requests.post('http://localhost:8080/predictions/license_plate', data=data)
        return response.json()['predictions'][0]
    
    def get_video(self):
        cap = cv2.VideoCapture(self.gstreamer)
        assset cap is not None
        return cap

    def get_frame(self,video):
        # read the 29th image in the sequence - ensures that the sytsmte is predicting on the most upto date image
        for i in range(28):
            ret, frame = video.read
        if not ret:
            return false, []
        return True, frame

    # generates the ocr reading based on the response of the plate recog model
    def ocr_generate(self,response,img):

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
            print("Predicting..")
            result = self.OCR_HANDLER.post(img)
            return result
        else:
            print("No plate detected")
            return []

    def __call__(self):
        # create videpo object
        video = self.get_video()

        # write frames to video file
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.outfile,four_cc,20,(960,1080))

        # check to see if the video object is opened and ready to use
        if video.isOpened():
            while True:
                # get the frame to perform analysis on 
                is_frame, frame = self.get_frame()
                
                # check if the frame was read sucessfylly
                if is_frame:
                    # read image
                    img = self.read_image_byte()
                
                    # check frame for license plate
                    response = self.find_license_plate(img)
                    print(response)

                    # # generate ocr reading (if needed)
                    ocr_reading = self.ocr_generate(response,img)
                    print(ocr_reading)

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
    lP_endpt = 'https://plain-ideas-wait-170-246-161-114.loca.lt:8080/predictions/license_plate'
    plate_threshold = 0.5
    ocr_endpt = 'https://sixty-hairs-rule-170-246-161-114.loca.lt:8501/v1/models/ocr_model/versions/2:predict'
    # get output file to store video
    outfile = sys.argv[1]
    # generate the stream for predictions
    gstreamer = gstreamer_pipeline()
    # instantiate class
    car = detect_predict_license_plate(lP_endpt,ocr_endpt,threshold,gstreamer,outfile)
    # print(car.find_license_plate())
    car()