import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
import matplotlib.pyplot as plt

# Directories
train_dir = 'Dataset/image_classifier/train'
test_dir = 'Dataset/image_classifier/test'

# Data augmentation and preparation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Set aside 20% of the training data for validation
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Adjusted to MobileNetV2 recommended size
    batch_size=16,  # Adjusted batch size for quicker processing
    class_mode='binary',
    subset='training'  # Use the remaining 80% for training
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Adjusted to MobileNetV2 recommended size
    batch_size=16,  # Adjusted batch size for quicker processing
    class_mode='binary',
    subset='validation'  # Use the 20% for validation
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),  # Adjusted to MobileNetV2 recommended size
    batch_size=16,  # Adjusted batch size for quicker processing
    class_mode='binary'
)

# Transfer learning with MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=15,  # Increased epochs for more training
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Load the best model
model = tf.keras.models.load_model('C:\\Users\\arnav\\Documents\\GitHub\\Coding\\Python\\Image Classifier\\best_model.keras')

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)
