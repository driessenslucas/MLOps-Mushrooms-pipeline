import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Activation, Dropout

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from typing import List
import os

from keras.applications import VGG19


def getTargets(filepaths: List[str]) -> List[str]:
    labels = [fp.split('/')[-1].split('_')[0] for fp in filepaths] # Get only the animal name

    return labels


def encodeLabels(y_train: List, y_test: List):
    label_encoder = LabelEncoder()
    y_train_labels = label_encoder.fit_transform(y_train)
    y_test_labels = label_encoder.transform(y_test)

    y_train_1h = to_categorical(y_train_labels)
    y_test_1h = to_categorical(y_test_labels)

    LABELS = label_encoder.classes_
    print(f"{LABELS} -- {label_encoder.transform(LABELS)}")

    return LABELS, y_train_1h, y_test_1h

def getFeatures(filepaths: List[str]) -> np.array:
    images = []
    for imagePath in filepaths:
        image = Image.open(imagePath).convert("RGB")
        image = np.array(image)
        images.append(image)
    return np.array(images)


def buildModel(inputShape: tuple, classes: int) -> Sequential:
    model = Sequential()
    height, width, depth = inputShape
    inputShape = (height, width, depth)
    chanDim = -1
    # Create the VGG19 base model (excluding the top fully-connected layers)
    vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=inputShape)

    # Freeze the layers of the VGG19 base model
    for layer in vgg19_base.layers:
        layer.trainable = False

    model.add(vgg19_base)

    # Add your custom layers on top of the VGG19 base
    model.add(Flatten())
    model.add(Dense(512, name='fc_1'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(classes, name='output'))
    model.add(Activation("softmax"))

    return model