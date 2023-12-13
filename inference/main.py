from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import pymongo
from pymongo import MongoClient

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

#normally this would be model_path = os.path.join('mushroom-classification', "INPUT_model_path", "mushroom-cnn")
#and then model = load_model(model_path)
#but my regestiation of the model failed, so I added it manually into the model section in azureml, thus you need this for it to work
model = load_model('./mushroom-classification/mushroom-cnn')

##old gradio implementation

# # Function to make predictions using the loaded model
# def predict(image):
#     original_image = image
#     original_image = original_image.resize((400, 400))
#     images_to_predict = np.expand_dims(np.array(original_image), axis=0)
#     predictions = model.predict(images_to_predict)
#     classifications = predictions.argmax(axis=1)
    
#     # Print certainty of all classes
#     print(predictions)

#     return f'{Mushrooms[classifications.tolist()[0]]}'

# # Gradio Interface for Mushroom Prediction
# iface = gr.Interface(
#     fn=predict, 
#     inputs=gr.Image(type='pil', label='Take a Picture'),
#     outputs='text',
#     live=True
# )


async def gradio():
    # implement gradio
    with gr.Blocks() as demo:
        # Mushrooms = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']

        # # model_path = os.path.join("mushroom-cnn")
        # # print(model_path)
        
        # #normally this would be model_path = os.path.join('mushroom-classification', "INPUT_model_path", "mushroom-cnn")
        # #and then model = load_model(model_path)
        # #but my regestiation of the model failed, so I added it manually into the model section in azureml, thus you need this for it to work
        # model = load_model('./mushroom-classification/mushroom-cnn')

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
    
# MongoDB setup (Replace with your MongoDB connection details)
#load env variables
import os

#get the env variables
mongo_user = os.environ.get('MONGODB_USERNAME')
mongo_password = os.environ.get('MONGODB_PASSWORD')
mongo_host = os.environ.get('MONGODB_HOSTNAME')
mongo_port = os.environ.get('MONGODB_PORT')
mongo_database = os.environ.get('MONGODB_DATABASE')


#mongo_uri = f"mongodb://{mongodb_username}:{mongodb_password}@{mongodb_hostname}:27017/{mongodb_database}"
mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:27017/{mongo_database}"


client = MongoClient(mongo_uri)
# Accessing the database
db = client[mongodb_database]
collection = db["mushrooms"]

@app.post("/upload/mushroom")
async def upload_mushroom(name: str, image: UploadFile = File(...)):
    image_content = await image.read()
    encoded_image = base64.b64encode(image_content).decode('utf-8')
    
    # Store in MongoDB
    collection.insert_one({"name": name, "image": encoded_image})
    return {"message": "Mushroom data stored successfully"}

@app.get("/mushroom/history")
def get_mushroom_history():
    history = collection.find({}, {"_id": 0, "name": 1, "image": 1})
    return list(history)
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
