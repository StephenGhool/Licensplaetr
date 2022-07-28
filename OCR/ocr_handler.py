import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import requests 
import urllib

print("Tensorflow version: ", tf.__version__)
tf.random.set_seed(1234)

class OCR():
    def __init__(self):
        self.img_width = 200
        self.img_height = 50
        self.characters = ['X', '8', 'E', 'B', '3', 'K', 'W', 'A', 'C', 'G', '7', 'L', '6', 'Z', 'M', '5', '4', 'N',
                           'H', '9', 'T', 'U', 'O', 'J', '1', 'R', 'Y', 'S', '2', '0', 'F', 'D', 'P']
        self.max_length = 7
        self.endpoint = "http://localhost:8501/v1/models/ocr_model/versions/2:predict"

    # Mapping characters to integers\n
    def char_to_num(self):
        return layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self.characters), num_oov_indices=0, mask_token=None)

    # Mapping integers back to original characters\n
    def num_to_char(self):
        return layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self.characters), mask_token=None, invert=True)

    def encode_single_sample(self,img):
        # 1. Read image\n
        # img = tf.io.read_file(self.img_path)
        img = open(img, 'rb').read()
        # 2. Decode and convert to grayscale\n
        img = tf.io.decode_image(img, channels=1)
        # 3. Convert to float32 in [0 1] range\n
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size\n
        img = tf.image.resize(img, [self.img_height, self.img_width])
        # 5. Transpose the image because we want the time\n
        # dimension to correspond to the width of the image.\n
        img = tf.transpose(img, perm=[1, 0, 2])
        return {"image": img}

    # A utility function to decode the output of the network\n
    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks you can use beam search\n
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
                  :, :self.max_length
                  ]
        print(results)
        # Iterate over the results and get back the text\n
        output_text = []
        # Mapping integers back to original characters\n
        num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self.characters), mask_token=None, invert=True)
        for res in results:
            res = tf.strings.reduce_join(num_to_char(res + 1)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    # used to send and retrieve prediction from server
    def post(self, img):
        # reformant input to send
        image_tensor = self.img_reformat(img)
       
        # Prepare the data that is going to be sent in the POST request
        json_data = {
            "signature_name": "serving_default",
            "inputs": {"image": image_tensor}
        }

        # Send the request to the Prediction API
        response = requests.post(self.endpoint, json=json_data)
    
        # extract prediction from response to get license plate
        pred = response.json()['outputs']

        # convert to numoy array
        pred = np.array(pred)

        # convert numpy to int
        pred = pred.astype(np.float32)
       
       # decode the prediction
        res = self.decode_batch_predictions(pred)
    
        return  res

    # reformatting img to post over http
    def img_reformat(self,img):
        img = tf.squeeze(img)
        image_tensor = tf.expand_dims(img, 0)
        image_tensor = tf.expand_dims(image_tensor, 3)
        image_tensor = image_tensor.numpy().tolist()
        return image_tensor


# if __name__ == "__main__":
#     #loading image
#     print("Loading image..")
#     path = 'HAF1129.jpg'

#     # intialize the OCR
#     print("Loading class..")
#     ocr_model = OCR(path)

#     for i in range(4):
#         # encode input image
#         print("Formatting image..")
#         img = ocr_model.encode_single_sample()
#         img = img["image"]
    
#         # send to server to predict
#         print("Predicting..")
#         result = ocr_model.post(img)
#         print(result)

# Note removed img_path from class init since it is not needed in production