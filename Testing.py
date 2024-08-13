import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image_dataset_from_directory # type: ignore

model = load_model('dog_cat_classifier.keras')

test_folder_path = 'C:\\Coding\\Projects\\Image Classifier\\test'

test_ds = image_dataset_from_directory(
    test_folder_path,
    image_size=(256, 256),
    batch_size=32
)

test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)
