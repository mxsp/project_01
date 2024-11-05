import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('/home/kali/Downloads/project_01-main/best_model.keras')

# Define the confidence threshold
confidence_threshold = 0.6

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(768, 768))  # Match input size
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    print(predictions)  # Optional: See the raw output of the model
    confidence = np.max(predictions)  # Get the highest confidence score
    class_index = np.argmax(predictions)  # Get the index of the predicted class

    # Check if confidence meets the threshold
    if confidence >= confidence_threshold:
        class_labels = ["aca", "n", "scc"]  # Get class labels corresponding to the indices
        predicted_class = class_labels[class_index]  # Get the predicted class label
        return "Predicted class: " + predicted_class + " with confidence: " + str(confidence)
    else:
        return "Prediction confidence is below the threshold. Confidence: " + str(confidence)

# Example usage
img_path = '/home/kali/Downloads/project_01-main/lungtumorSCCMcGeough05.jpg'  # Replace with your image path
result = predict_image(img_path)
print(result)
