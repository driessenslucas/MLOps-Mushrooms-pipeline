import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os

#set gpu  
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


#inference/mushroom-classification/INPUT_model_path/mushroom-cnn
# It would've been better to use an environment variable to fix this line actually...
model_path = os.path.join('mushroom-classification', "INPUT_model_path", "mushroom-cnn")

model = load_model(model_path) # Model_name here!

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

@app.get("/healthcheck")
def healthcheck():
    return {
        "status": "Healthy",
        "version": "0.1.1"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
