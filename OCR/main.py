# used to fixed "cannot allocate memory in static TLS block" error with cv2
#export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

# MODEL DEFINITION      __________________________________________________________________________________________________\n
#      Layer (type)                    Output Shape         Param #     Connected to                     \n
#      ==================================================================================================\n
#      image (InputLayer)              [(None 200 50 1)] 0                                            \n
#      __________________________________________________________________________________________________\n
#      Conv1 (Conv2D)                  (None 200 50 64)  640         image[0][0]                      \n
#      __________________________________________________________________________________________________\n
#      pool1 (MaxPooling2D)            (None 100 25 64)  0           Conv1[0][0]                      \n
#      __________________________________________________________________________________________________\n
#      Conv2 (Conv2D)                  (None 100 25 128) 73856       pool1[0][0]                      \n
#      __________________________________________________________________________________________________\n
#      pool2 (MaxPooling2D)            (None 50 12 128)  0           Conv2[0][0]                      \n
#      __________________________________________________________________________________________________\n
#      Conv3 (Conv2D)                  (None 50 12 256)  295168      pool2[0][0]                      \n
#      __________________________________________________________________________________________________\n
#      pool3 (MaxPooling2D)            (None 25 6 256)   0           Conv3[0][0]                      \n
#      __________________________________________________________________________________________________\n
#      reshape (Reshape)               (None 50 768)      0           pool3[0][0]                      \n
#      __________________________________________________________________________________________________\n
#      dense1 (Dense)                  (None 50 128)      98432       reshape[0][0]                    \n
#      __________________________________________________________________________________________________\n
#      dropout (Dropout)               (None 50 128)      0           dense1[0][0]                     \n
#      __________________________________________________________________________________________________\n
#      bidirectional (Bidirectional)   (None 50 512)      788480      dropout[0][0]                    \n
#      __________________________________________________________________________________________________\n
#      bidirectional_1 (Bidirectional) (None 50 256)      656384      bidirectional[0][0]              \n
#      __________________________________________________________________________________________________\n
#      bidirectional_2 (Bidirectional) (None 50 128)      164352      bidirectional_1[0][0]            \n
#      __________________________________________________________________________________________________\n
#      label (InputLayer)              [(None None)]       0                                            \n
#      __________________________________________________________________________________________________\n
#      dense2 (Dense)                  (None 50 34)       4386        bidirectional_2[0][0]            \n
#      __________________________________________________________________________________________________\n
#      ctc_loss (Custom>CTCLayer)      (None 50 34)       0           label[0][0]                      \n
#                                                                       dense2[0][0]                     \n
#      ==================================================================================================\n
#      Total params: 2081698\n
#      Trainable params: 2081698\n
#      Non-trainable params: 0\n
#      __________________________________________________________________________________________________\n
#     ]
#    }
#   ]
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution

enable_eager_execution ()
import os
import numpy as np
#import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("Tensorflow version: ",tf.__version__)
tf.random.set_seed(1234)

from os import listdir
from os.path import isfile, join
import os

# used to fixed "cannot allocate memory in static TLS block" error with cv2
#export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
import cv2

model = keras.models.load_model("tt_license_plates_model_best_new_data_tf_saved_model")
#model.summary()

characters = ['X', '8', 'E', 'B', '3', 'K', 'W', 'A', 'C', 'G', '7', 'L', '6', 'Z', 'M', '5', '4', 'N', 'H', '9','T', 'U', 'O', 'J', '1', 'R', 'Y', 'S', '2', '0', 'F', 'D', 'P']

#            [ 2    1    1    1   49    1    2    1    2   44   47    2   45    2    1   45   47    1    0    1   2    2    2    2    47   1    2    2    46   1    2    1    0  36]
#            ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4','5', '6', '7','8','9','0']

# required image dimensions\n
img_width = 200
img_height = 50

# maximum num of characters in license plate\n
max_length = 7

# Mapping characters to integers\n
char_to_num = layers.experimental.preprocessing.StringLookup(
  vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

# Mapping integers back to original characters\n
num_to_char = layers.experimental.preprocessing.StringLookup(
  vocabulary=list(characters), mask_token=None, invert=True
)

# used to show images\n
#def plot_image(img):
#  plt.imshow(img[:, :, 0].T, cmap="gray")
#  plt.show()

def encode_single_sample(img_path):
  # 1. Read image\n
  img = tf.io.read_file(img_path)
  # 2. Decode and convert to grayscale\n
  img = tf.io.decode_png(img, channels=1)
  # 3. Convert to float32 in [0 1] range\n
  img = tf.image.convert_image_dtype(img, tf.float32)
  # 4. Resize to the desired size\n
  img = tf.image.resize(img, [img_height, img_width])
  # 5. Transpose the image because we want the time\n
  # dimension to correspond to the width of the image.\n
  img = tf.transpose(img, perm=[1, 0, 2])
  # 6. Map the characters in label to numbers\n
  # label = char_to_num(tf.strings.unicode_split(label input_encoding=\UTF-8\))\n
  # 7. Return a dict as our model is expecting two inputs\n
  return {"image": img}

# A utility function to decode the output of the network\n
def decode_batch_predictions(pred):
  input_len = np.ones(pred.shape[0]) * pred.shape[1]
  # Use greedy search. For complex tasks you can use beam search\n
  results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :max_length
            ]
  print(results)
  # Iterate over the results and get back the text\n
  output_text = []
  for res in results:
      res = tf.strings.reduce_join(num_to_char(res+1)).numpy().decode("utf-8")
      output_text.append(res)
  return output_text
   
   

   
img_path = 'HAF1129.jpg'
img_path = 'HAF1129.jpg'

img = encode_single_sample(img_path)

print(characters)

img = img["image"]
img = np.array(img)
#plot_image(img)

# Get the prediction model by extracting layers till the output layer\n
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output)

prediction_model.summary()
img = tf.squeeze(img)
print(img.shape)
image_tensor = tf.expand_dims(img, 0)
image_tensor = tf.expand_dims(image_tensor, 3)
print(image_tensor.shape)
image_tensor = image_tensor.numpy().tolist()

pred = prediction_model.predict(image_tensor)
print(pred.shape)
print(type(pred))
print(pred)
print(tf.argmax(pred))
print(type(pred[0][0][0]))
print(pred[0][0][0])

pred_texts = decode_batch_predictions(pred)
#print(pred[0,:1,:])
print(pred_texts)



#%%
img
#%%
