import torch
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
characters = ['X', '8', 'E', 'B', '3', 'K', 'W', 'A', 'C', 'G', '7', 'L', '6', 'Z', 'M', '5', '4', 'N', 'H', '9','T', 'U', 'O', 'J', '1', 'R', 'Y', 'S', '2', '0', 'F', 'D', 'P']

# required image dimensions\n
img_width = 200
img_height = 50

def encode_single_sample(img_path):
    # 1. Read image\n
    img = tf.io.read_file(img_path)
    print(type(img))
    # 2. Decode and convert to grayscale\n
    img = tf.io.decode_png(img, channels=1)
    print(type(img))
    # 3. Convert to float32 in [0 1] range\n
    img = tf.image.convert_image_dtype(img, tf.float32)
    print(type(img))
    # 4. Resize to the desired size\n
    img = tf.image.resize(img, [img_height, img_width])
    print(type(img))
    # 5. Transpose the image because we want the time\n
    # dimension to correspond to the width of the image.\n
    img = tf.transpose(img, perm=[1, 0, 2])
    print(type(img))
    # 6. Map the characters in label to numbers\n
    # label = char_to_num(tf.strings.unicode_split(label input_encoding=\UTF-8\))\n
    # 7. Return a dict as our model is expecting two inputs\n
    return {"image": img}


# Pytorch
# PATH = 'best_script.pt'
# image = 'HAF1129.jpg'
# model = torch.jit.load(PATH)
# img = encode_single_sample(image)
# img = img["image"]
# img = tf.convert_to_tensor(img)
# print(type(img))
# print(img)
# # Get the prediction model by extracting layers till the output layer\n
# print("Predicting...")
# model(img)
# print("DONE!!")


# Tensorflow
model = keras.models.load_model("tt_license_plates_model_best_new_data_tf_saved_model")
#model.summary()

#%%
path = 'HAF1129.jpg'
img = tf.keras.preprocessing.image.load_img(path)
img = tf.convert_to_tensor(img)
print(type(img))
#%%
