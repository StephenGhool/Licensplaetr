from tensorflow.python.framework.ops import enable_eager_execution

enable_eager_execution()

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import requests 
import urllib

print("Tensorflow version: ", tf.__version__)
tf.random.set_seed(1234)


class Model():
    def __init__(self):
        # self.model = keras.models.load_model("tt_license_plates_model_best_new_data_tf_saved_model")
        # self.prediction_model = keras.models.Model(
        #     self.model.get_layer(name="image").input, self.model.get_layer(name="dense2").output)
       return

# load the model
model = Model()

# %%
class OCR(Model):
    def __init__(self, img_path, model):
        # self.prediction_model = model.prediction_model
        self.img_width = 200
        self.img_height = 50
        self.characters = ['X', '8', 'E', 'B', '3', 'K', 'W', 'A', 'C', 'G', '7', 'L', '6', 'Z', 'M', '5', '4', 'N',
                           'H', '9', 'T', 'U', 'O', 'J', '1', 'R', 'Y', 'S', '2', '0', 'F', 'D', 'P']
        self.max_length = 7
        self.img_path = img_path
        self.endpoint = "http://localhost:8501/v1/models/ocr_model/versions/2:predict"

    # Mapping characters to integers\n
    def char_to_num(self):
        return layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self.characters), num_oov_indices=0, mask_token=None)

    # Mapping integers back to original characters\n
    def num_to_char(self):
        return layers.experimental.preprocessing.StringLookup(
            vocabulary=list(self.characters), mask_token=None, invert=True)

    def encode_single_sample(self):
        # 1. Read image\n
        # img = tf.io.read_file(self.img_path)
        img = open(self.img_path, 'rb').read()
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

    # A function to perform predictions given the image
    # def predict(self, img):
    #     # pred = self.prediction_model.predict(img[None, :, :])
    #     pred = self.prediction_model.predict(img)
    #     pred_texts = self.decode_batch_predictions(pred)
    #     return pred_texts

    def post(self, img):
        # ----------------------------------1------------------------------------------
        # data = json.dumps({"signature_name": "serving_default", "instances": img.tolist()})
        # print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))
        # headers = {"content-type": "application/json"}
        # json_response = requests.post(self.endpoint, data=data, headers=headers)
        # predictions = json.loads(json_response.text)
        # print(predictions)

        # ----------------------------------2------------------------------------------
        img = tf.squeeze(img)
        image_tensor = tf.expand_dims(img, 0)
        image_tensor = tf.expand_dims(image_tensor, 3)
        image_tensor = image_tensor.numpy().tolist()
        lista = image_tensor
        arow = len(lista)
        acol = len(lista[0])
        print("Rows : " + str(arow))
        print("Columns : " + str(acol))

        # Prepare the data that is going to be sent in the POST request
        json_data = {
            "signature_name": "serving_default",
            "inputs": {"image": image_tensor}
        }

        # Send the request to the Prediction API
        response = requests.post(self.endpoint, json=json_data)
        print(response.json())

        # extract prediction from response to get license plate
        prediction = tf.argmax(response.json()['outputs'])
        pred = response.json()['outputs']

        # convert to numoy array
        pred = np.array(pred)

        # get some characteristic of the output
        print(type(pred))
        # print(pred)
        print(pred.shape)
        print(pred[0][0][0])
        print(type(pred[0][0][0]))

        # convert numpy to int
        pred = pred.astype(np.float32)
        print(type(pred[0][0][0]))
        res = self.decode_batch_predictions(pred)
        print(res)
        return


path = 'HAF1129.jpg'

# intialize the OCR
ocr_model = OCR(path,model)

# encode input image
img = ocr_model.encode_single_sample()
img = img["image"]

img2 = np.array(img).tolist()
lista = img2
arow = len(lista)
acol = len(lista[0])
print("Rows : " + str(arow))
print("Columns : " + str(acol))

img = tf.expand_dims(img, 0)
print(img.shape)
# pred_texts = ocr_model.predict(img)

# send to server to predict
ocr_model.post(img)

# img = tf.squeeze(img)
# print(img.shape)
# image_tensor = tf.expand_dims(img, 0)
# image_tensor = tf.expand_dims(image_tensor, 3)
# print(image_tensor.shape)

# image_tensor = image_tensor.numpy().tolist()
# lista = image_tensor
# arow = len(lista)
# acol = len(lista[0])
# print("Rows : " + str(arow))
# print("Columns : " + str(acol))
# pred_texts = ocr_model.predict(image_tensor)

# # %%
# # save model - removing unwanted layers
# model = keras.models.load_model("tt_license_plates_model_best_new_data_tf_saved_model")
# prediction_model = keras.models.Model(
#     model.get_layer(name="image").input, model.get_layer(name="dense2").output)
# tf.saved_model.save(prediction_model, r'C:\Users\steph\OneDrive - The University of the West Indies, '
#                                       r'St. Augustine\TTLABS\License Plate Recognition\Deployment\OCR')
# print("Done")