# import os
# import cv2
# import random
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# import tensorflow as tf
# from keras import backend as K
# from keras.models import Model
# from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
# from keras.optimizers import Adam
# from keras.models import load_model

# def preprocess(img):
#     (h, w) = img.shape
    
#     final_img = np.ones([64, 256])*255 # blank white image
    
#     # crop
#     if w > 256:
#         img = img[:, :256]
        
#     if h > 64:
#         img = img[:64, :]
    
    
#     final_img[:h, :w] = img
#     return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

# alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
# max_str_len = 24 # max length of input labels
# num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
# num_of_timestamps = 64 # max length of predicted labels


# def label_to_num(label):
#     label_num = []
#     for ch in label:
#         label_num.append(alphabets.find(ch))
        
#     return np.array(label_num)

# def num_to_label(num):
#     ret = ""
#     for ch in num:
#         if ch == -1:  # CTC Blank
#             break
#         else:
#             ret+=alphabets[ch]
#     return ret

# model = load_model('/mnt/c/Users/evane/source/repos/handwriting-recognition-training/handwriting-recognition.h5', compile=False)

# test = pd.read_csv('/mnt/c/Users/evane/.cache/kagglehub/datasets/landlord/handwriting-recognition/versions/1/written_name_test_v2.csv')

# plt.figure(figsize=(15, 10))
# for i in range(6):
#     ax = plt.subplot(2, 3, i+1)
#     img_dir = '/mnt/c/Users/evane/.cache/kagglehub/datasets/landlord/handwriting-recognition/versions/1/test_v2/test/'+test.loc[i, 'FILENAME']
#     image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
#     plt.imshow(image, cmap='gray')
    
#     image = preprocess(image)
#     image = image/255.
#     pred = model.predict(image.reshape(1, 256, 64, 1))
#     decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
#                                        greedy=True)[0][0])
#     plt.title(num_to_label(decoded[0]), fontsize=12)
#     plt.axis('off')
    
# plt.subplots_adjust(wspace=0.2, hspace=-0.8)
# plt.show()

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K

model = tf.keras.models.load_model("handwriting-recognition.h5")

# Define alphabets and utilities
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "

def num_to_label(num):
    return ''.join(alphabets[ch] for ch in num if ch != -1)

def preprocess(img):
    h, w = img.shape
    final_img = np.ones([64, 256]) * 255
    if w > 256:
        img = img[:, :256]
    if h > 64:
        img = img[:64, :]
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

# Load test set
# test = pd.read_csv('C:/Users/Julius/.cache/kagglehub/datasets/landlord/handwriting-recognition/versions/1/written_name_test_v2.csv')
test = pd.read_csv('/mnt/c/Users/evane/.cache/kagglehub/datasets/landlord/handwriting-recognition/versions/1/written_name_test_v2.csv')


for i in range(6):
    data = test.loc[[i]]
    # img_path = f"C:/Users/Julius/.cache/kagglehub/datasets/landlord/handwriting-recognition/versions/1/test_v2/test/{data['FILENAME'].values[0]}"
    img_path = f"/mnt/c/Users/evane/.cache/kagglehub/datasets/landlord/handwriting-recognition/versions/1/test_v2/test/{data['FILENAME'].values[0]}"
    
    
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Image not found: {img_path}")
        continue

    image = preprocess(image) / 255.0
    image = np.expand_dims(image, axis=(0, -1))  # shape (1, 256, 64, 1)

    pred = model.predict(image)
    decoded = tf.keras.backend.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0].numpy()

    actual_label = data['IDENTITY'].values[0]
    predicted_label = num_to_label(decoded[0])
    print(f"Image {i+1}: True = {actual_label}, Predicted = {predicted_label}")

    # Plot
    plt.subplot(2, 3, i+1)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f"True: {actual_label}\nPred: {predicted_label}", fontsize=10)
    plt.axis("off")

plt.tight_layout()
plt.show()

plt.subplots_adjust(wspace=0.2, hspace=-0.8)