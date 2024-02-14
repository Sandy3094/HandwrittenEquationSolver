import cv2
import os
import numpy as np
from keras.models import model_from_json

# img = cv2.imread("../data/smallerDataset/dataset/3/0Qswwo6J.png")
img = cv2.imread("3_2.png")


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshold_image = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
threshold_image = cv2.resize(threshold_image, (32, 32))

json_file = open('model_final.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

test_X = np.expand_dims(threshold_image, axis=0)
test_X = np.expand_dims(test_X, axis=-1)


print(test_X.shape)
print(np.argmax(loaded_model.predict(test_X)))

