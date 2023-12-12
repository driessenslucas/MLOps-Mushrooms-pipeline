from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Set GPU
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Mushrooms = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']

model_path = os.path.join("mushroom-cnn")
model = load_model(model_path)

# Function to make predictions using the loaded model
def predict(image):
    original_image = image
    original_image = original_image.resize((400, 400))
    images_to_predict = np.expand_dims(np.array(original_image), axis=0)
    predictions = model.predict(images_to_predict)
    classifications = predictions.argmax(axis=1)
    
    # Print certainty of all classes
    print(predictions)

    return f'{Mushrooms[classifications.tolist()[0]]}'

# Gradio Interface for Mushroom Prediction
iface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type='pil', label='Take a Picture'),
    outputs='text',
    live=True
)


async def gradio():
    # Replace the chatbot interface with the mushroom prediction interface
    with gr.Blocks() as demo:
        Mushrooms = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']

        model_path = os.path.join("mushroom-cnn")
        model = load_model(model_path)

        # Function to make predictions using the loaded model
        def predict(image):
            original_image = image
            original_image = original_image.resize((400, 400))
            images_to_predict = np.expand_dims(np.array(original_image), axis=0)
            predictions = model.predict(images_to_predict)
            classifications = predictions.argmax(axis=1)
            
            # Print certainty of all classes
            print(predictions)

            return f'{Mushrooms[classifications.tolist()[0]]}'

        # Gradio Interface for Mushroom Prediction
        iface = gr.Interface(
            fn=predict, 
            inputs=gr.Image(type='pil', label='Take a Picture'),
            outputs='text',
            live=True
        )

        mushroom_prediction = iface  # Assuming your Gradio interface variable is named 'iface'

    # Run Gradio Interface in the background
    global app
    demo.queue()
    demo.startup_events()
    app = gr.mount_gradio_app(app,demo, '/gradio')
    
    
app.add_event_handler("startup", gradio)


@app.get("/gradio_exists")
async def gradio_exists():
    return {"message": "The /gradio endpoint exists."}

# Existing FastAPI endpoints
@app.post('/upload/image')
async def uploadImage(img: UploadFile = File(...)):
    original_image = Image.open(img.file)
    # Gradio will handle resizing
    images_to_predict = np.expand_dims(np.array(original_image), axis=0)
    predictions = model.predict(images_to_predict)
    classification = Mushrooms[predictions.argmax(axis=1).tolist()[0]]
    return classification

@app.get("/healthcheck")
def healthcheck():
    return {
        "status": "Healthy",
        "version": "0.1.1"
    }
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
