import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('Python/Image Classifier/best_model.keras')

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust size if needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image array

    # Make prediction
    prediction = model.predict(img_array)
    
    # Return the prediction
    return 'Dog' if prediction[0] > 0.5 else 'Cat'

def count_images_in_folder(folder_path):
    counts = {'Cat': 0, 'Dog': 0}
    
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it is a file (ignore directories)
        if os.path.isfile(file_path):
            try:
                # Predict the image
                prediction = predict_image(file_path)
                
                # Update counts
                if prediction in counts:
                    counts[prediction] += 1
            except Exception as e:
                # Handle any errors (e.g., non-image files)
                print(f"Error processing {filename}: {e}")
    
    return counts

# Test the function with a folder
folder_path = 'Dataset/image_classifier/test/cats'  # Change to your folder path
counts = count_images_in_folder(folder_path)

# Print the counts
print(f"Number of Cats: {counts['Cat']}")
print(f"Number of Dogs: {counts['Dog']}")
