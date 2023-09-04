# Import dependencies
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


model = tf.keras.models.load_model('app/ml/similarity_learning.h5',custom_objects={'KerasLayer':hub.KerasLayer})


class_labels = ['apple',
 'bean sprouts',
 'chicken meat',
 'cucumber',
 'ginger',
 'jeruk',
 'keju',
 'lamb meat',
 'meat',
 'nasi',
 'orange',
 'pisang',
 'potate',
 'salmon meat',
 'sausage',
 'sawi',
 'susu',
 'wortel']


def predictImage(img):
    img = img.resize((150, 150))
    img = img.convert('RGB') 
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label, np.max(predictions[0])