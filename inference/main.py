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


origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8000",
    "http://localhost:8001",
    "http://localhost:8500",
    "http://localhost:8600",
    "http://localhost:8700",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Mushrooms = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']

#normally this would be model_path = os.path.join("mushroom-classification", "INPUT_model_path", "mushroom-cnn")
#and then model = load_model(model_path)
#but my regestiation of the model failed the second time around, so I added the model manually into the model registry in azure ml, thus you need this for it to work
model = load_model('./mushroom-classification/mushroom-cnn')



async def gradio():
    # implement gradio
    with gr.Blocks() as demo:

        # Function to make predictions using the loaded model
        def predict(image):
            original_image = image
            original_image = original_image.resize((400, 400))
            images_to_predict = np.expand_dims(np.array(original_image), axis=0)
            predictions = model.predict(images_to_predict)
            
            #get the highest probability
            classifications = predictions.argmax(axis=1)
            
            # Print probability of all classes
            print(predictions)

            #get the name of the mushroom after the prediction
            return f'{Mushrooms[classifications.tolist()[0]]}'

        # Gradio Interface for Mushroom Prediction
        iface = gr.Interface(
            fn=predict, 
            inputs=gr.Image(type='pil', label='Take a Picture'),
            outputs='text',
            live=True
        )


    # Run Gradio Interface in the background
    global app
    demo.queue()
    demo.startup_events()
    app = gr.mount_gradio_app(app, demo, '/gradio')
    
#this allows the gradio endpoint to be created on startup
app.add_event_handler("startup", gradio)

#let the user know that the gradio endpoint exists
@app.get("/gradio_exists")
async def gradio_exists():
    return {"message": "The /gradio endpoint exists."}

# Existing FastAPI endpoints
@app.post('/upload/image')
async def uploadImage(img: UploadFile = File(...)):
    original_image = Image.open(img.file)
    original_image = original_image.resize((400, 400))
    images_to_predict = np.expand_dims(np.array(original_image), axis=0)
    predictions = model.predict(images_to_predict) #[0 1 0]
    classifications = predictions.argmax(axis=1) # [1]

    #print certaintity of all classes
    print(predictions)
    return Mushrooms[classifications.tolist()[0]] # "Dog"

# Healthcheck endpoint
@app.get("/healthcheck")
def healthcheck():
    return {
        "status": "Healthy",
        "version": "0.1.1"
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)