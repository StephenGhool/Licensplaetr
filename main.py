from OCR.ocr_handler import OCR 
import requests
from PIL import Image
import cv2 
import tensorflow as tf

# used to handle both plate recog and ocr models whilst using video input
class detect_predict_license_plate:
    def __init__(self, inputs) -> None:
        self.plate_recog_endpoint = inputs['plate_endpoint']
        self.plate_threshold = inputs['plate_threshold']
        self.frame = 'car.jpg'
        # ocr handler -> placed here so that it would load once
        self.OCR_HANDLER = OCR()
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
            print(result)
        else:
            print("No plate detected")
        return

    def __call__(self):

        # read image
        img = self.read_image_byte()
    
        # check frame for license plate
        response = self.find_license_plate(img)
        print(response)

        # # generate ocr reading (if needed)
        ocr_reading = self.ocr_generate(response,img)

        self.crop_image(response)
        return

if __name__ == "__main__":
    # define inputs to detect and predict class
    inputs ={
        'plate_endpoint':'http://localhost:8080/predictions/license_plate',
        'plate_threshold':0.5
    }

    # instantiate class
    car = detect_predict_license_plate(inputs)
    # print(car.find_license_plate())
    car()