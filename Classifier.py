import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os
import cv2 as cv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# LOAD DATA
DIR = "./WasteImagesDataset"
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(DIR, validation_split=0.1, subset="training", seed=42, batch_size=16)
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(DIR, validation_split=0.1, subset="validation", seed=42, batch_size=16)

classes = train_dataset.class_names
numClasses = len(train_dataset.class_names)
print(classes)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# MODELS

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.RandomFlip('horizontal_and_vertical'),
  tf.keras.layers.RandomRotation(0.2),
])

# RESNET152 MODEL (using pre-trained model, no training)
baseModel = tf.keras.applications.ResNet152(input_shape=(256, 256, 3), weights='imagenet', include_top=False, classes=numClasses)
for layers in baseModel.layers:
    layers.trainable = False  # Freeze the layers

last_output = baseModel.layers[-1].output
x = tf.keras.layers.Dropout(0.5)(last_output)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(numClasses, activation='softmax')(x)

model = tf.keras.Model(inputs=baseModel.input, outputs=x)

# ONE IMAGE PREDICTION

url = "https://images.unsplash.com/photo-1577705998148-6da4f3963bc8?ixid=MnwxMjA3fDB8MHxzZWFyY2h8Nnx8Y2FyZGJvYXJkfGVufDB8fDB8fA%3D%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60"
# Download image without the `progress` argument
image = tf.keras.utils.get_file("Image1.jpg", origin=url, cache_dir='./')

# Preprocess the image
img = tf.keras.preprocessing.image.load_img(image, target_size=(256, 256))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

# Predict the class
predictions = model.predict(img_array)

# Display the image and prediction
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show()

# Output the prediction
predicted_class = classes[np.argmax(predictions)]
print(f"Prediction: {predicted_class}")

# MODEL EVALUATION

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center", color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(f'Predicted label\naccuracy={accuracy:.4f}; misclass={misclass:.4f}')
    plt.show()

plt.figure(figsize=(10, 10))
true = []
predictions = []

# Evaluate the model on test dataset (for confusion matrix)
for images, labels in test_dataset.take(10):
    pred = model.predict(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        true.append(labels[i])
        predictions.append(np.argmax(pred[i]))
        plt.title(classes[labels[i]])
        plt.axis("off")

# Confusion matrix plot
plot_confusion_matrix(tf.math.confusion_matrix(true, predictions), classes)
