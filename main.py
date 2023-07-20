# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:48:25 2023

@author: aitza
"""






import os
from IPython.display import Image, display
from tensorflow.keras.utils import load_img
from PIL import ImageOps
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers



img_size = (160, 160)
num_classes = 6
batch_size = 32


train_image_path = r'D:\Extra\LoveDA\Train\Rural\images_png'
train_mask_path = r'D:\Extra\LoveDA\Train\Rural\masks_png'
val_image_path = r'D:\Extra\LoveDA\Val\Rural\images_png'
val_mask_path = r'D:\Extra\LoveDA\Val\Rural\masks_png'
test_image_path = r'D:\Extra\LoveDA\Test\Rural\images_png'


train_img_paths = sorted(
    [
        os.path.join(train_image_path, fname)
        for fname in os.listdir(train_image_path)
        if fname.endswith(".png")
    ]
)
train_mask_paths = sorted(
    [
        os.path.join(train_mask_path, fname)
        for fname in os.listdir(train_mask_path)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

val_img_paths = sorted(
    [
        os.path.join(val_image_path, fname)
        for fname in os.listdir(val_image_path)
        if fname.endswith(".png")
    ]
)
val_mask_paths = sorted(
    [
        os.path.join(val_mask_path, fname)
        for fname in os.listdir(val_mask_path)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

test_mask_paths = sorted(
    [
        os.path.join(test_image_path, fname)
        for fname in os.listdir(test_image_path)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of images in training data:", len(train_img_paths))

for input_path, target_path in zip(train_img_paths[:10], train_mask_paths[:10]):
    print(input_path, "|", target_path)
    

display(Image(filename=train_img_paths[9]))

img = ImageOps.autocontrast(load_img(train_mask_paths[9]))
display(img)

img_array = np.array(img)
unique_pixels = set(tuple(pixel) for row in img_array for pixel in row)

print("Unique Pixels:", unique_pixels)

class OxfordPets(keras.utils.Sequence):
    

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            
            y[j] -= 1
        return x, y
    
    



def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    previous_block_activation = x  

    
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  
        previous_block_activation = x  

    

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    outputs = layers.Conv2D(num_classes, 6, activation="softmax", padding="same")(x)


    model = keras.Model(inputs, outputs)
    return model



keras.backend.clear_session()


model = get_model(img_size, num_classes)
model.summary()


train_gen = OxfordPets(batch_size, img_size, train_img_paths, train_mask_paths)

val_gen = OxfordPets(batch_size, img_size , val_img_paths, val_mask_paths)


model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint("Rural_segmentation.h5", save_best_only=True)
]


model.fit(train_gen, epochs=15, validation_data=val_gen, callbacks=callbacks)


