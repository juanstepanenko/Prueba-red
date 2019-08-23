import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Load files we already generated

length, height = 224, 224 # Same as in train.py
model = './model/model.h5'
weigths = './model/weigths.h5'
cnn = load_model(model)
cnn.load_weights(weigths)

def predict(file): 
  image = load_img(file, target_size=(length, height))
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0) # On axis 0 or first dimention we add a new dimention
  predictionArray = cnn.predict(x) # Format: [[1,0]]. On the first dimention it have a number 1 when the prediction is the correct one
  result = predictionArray[0] 
  answer = np.argmax(result) # Get index of the maximum result value
  if answer == 0:
    print("Prediction: Problem 1")
  elif answer == 1:
    print("Prediction: Problem 2")

  return answer

# How to predict:
  predict("descarga (1)") # File from validation