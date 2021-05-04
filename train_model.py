import os
import glob
import numpy as np
import cv2
import cvlib as cv
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input


batch_size = 32
img_height = 224
img_width  = 224
base_lr = 1e-5
epochs  = 20

path = './data/dataset'


# 데이터 셋
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    seed=93,
    subset='training'
)

valid_generator = test_datagen.flow_from_directory(
    path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    seed=93,
    subset='validation'
)

# input, 특징추출기
img_shape = (img_height, img_width, 3)
base_model = MobileNetV2(input_shape=img_shape,
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False

# 분류기(output)
x = base_model.output
x = layers.AveragePooling2D(pool_size=(5, 5), strides=1, padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(2, activation='softmax')(x)

# 모델 생성
model = keras.models.Model(inputs=base_model.input, outputs=x)

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_lr, decay=base_lr/epochs),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 저장 경로
if not os.path.isdir('./models'):
    os.mkdir('/data')

# 콜백 함수
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
    filepath='./models/face_mask_detector.h5',
    monitor='val_loss',
    save_best_only=True)
]

# 모델 학습
print("=====모델 학습=====")
history = model.fit(
          train_generator,
          validation_data=valid_generator,
          epochs=epochs,
          callbacks=callbacks_list
)