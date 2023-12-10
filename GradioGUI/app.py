import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# set GPU
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

Mushrooms = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']

# Model path
model_path = 'GradioGUI/mushroom-classification/INPUT_model_path/mushroom-cnn'
model = load_model(model_path)




# Function to make predictions using the loaded model
def predict(image):
    original_image = image
    original_image = original_image.resize((400, 400))
    images_to_predict = np.expand_dims(np.array(original_image), axis=0)
    predictions = model.predict(images_to_predict) #[0 1 0]
    classifications = predictions.argmax(axis=1) # [1]
    
    #print certaintity of all classes
    print(predictions)

    return f'{Mushrooms[classifications.tolist()[0]]}'

# Gradio Interface
iface = gr.Interface(fn=predict, 
                     inputs=gr.Image(type='pil', label='Take a Picture'),
                     outputs='text',
                     
                     live=True)

# Launch the Gradio app
iface.launch(inline=False, share=True)