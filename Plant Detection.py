# import cv2
# import numpy as np
#
# # Load YOLO
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
#
# # Load classes
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]
#
# # Load image
# image = cv2.imread("/home/mayank/PycharmProjects/melomaniac2oo3/Plant.jpeg")
# height, width, _ = image.shape
#
# # Create blob from image
# blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
# net.setInput(blob)
#
# # Get output layer names
# output_layer_names = net.getUnconnectedOutLayersNames()
#
# # Forward pass and get predictions
# outs = net.forward(output_layer_names)
#
# # Process predictions
# for out in outs:
#     for detection in out:
#         scores = detection[5:]
#         class_id = np.argmax(scores)
#         confidence = scores[class_id]
#
#         if confidence > 0.5:  # Set your confidence threshold
#             center_x = int(detection[0] * width)
#             center_y = int(detection[1] * height)
#             w = int(detection[2] * width)
#             h = int(detection[3] * height)
#
#             # Calculate bounding box coordinates
#             x = int(center_x - w/2)
#             y = int(center_y - h/2)
#
#             # Draw bounding box
#             cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(image, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#             # Extract and display ROI (Region of Interest)
#             roi = image[y:y+h, x:x+w]
#             cv2.imshow("ROI", roi)
#             cv2.waitKey(0)
#
# # Display the result
# cv2.imshow("Plant Disease Detection", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
#
# # Load pre-trained model
# model = load_model("/home/mayank/PycharmProjects/melomaniac2oo3/model.h5")
#
# # Load classes
# classes = ["healthy", "diseased"]
#
# # Load image
# image = cv2.imread("/home/mayank/PycharmProjects/melomaniac2oo3/Plant.jpeg")
# image = cv2.resize(image, (224, 224))  # Resize to match the model's input size
#
# # Preprocess the image
# image = image / 255.0  # Normalize pixel values to be between 0 and 1
# image = np.expand_dims(image, axis=0)  # Add batch dimension
#
# # Perform inference
# predictions = model.predict(image)
#
# # Get the predicted class and confidence
# predicted_class = classes[np.argmax(predictions)]
# confidence = predictions[0][np.argmax(predictions)]
#
# # Display the result
# if predicted_class == "diseased" and confidence > 0.5:
#     print("Plant is diseased with confidence:", confidence)
#     # Draw bounding box or take further action as needed
#     # You may need additional code here to draw a bounding box or take other actions based on your requirements.
# else:
#     print("Plant is healthy.")
#
# # Note: You need to replace "path/to/your/plant_disease_model.h5" and "path/to/your/image.jpg" with the actual paths.

# import cv2
# import numpy as np
# from tensorflow.keras.applications.inception_v3 import MobileNetV2, preprocess_input, decode_predictions
#
# # Load pre-trained InceptionV3 model
# model = InceptionV3(weights='imagenet')
#
# # Load image
# image_path = "/home/mayank/PycharmProjects/melomaniac2oo3/Plant.jpeg"
# image = cv2.imread(image_path)
# image_for_display = image.copy()  # Create a copy for display purposes
#
# # Resize image for InceptionV3 input size
# input_image = cv2.resize(image, (299, 299))
# input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
#
# # Preprocess the image for InceptionV3 model
# input_image = preprocess_input(input_image)
#
# # Perform inference
# predictions = model.predict(input_image)
#
# # Decode and print predictions
# decoded_predictions = decode_predictions(predictions, top=3)[0]
# for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
#     print(f"{i + 1}: {label} ({score:.2f})")
#
# # Get bounding box coordinates
# height, width, _ = image.shape
# x, y, w, h = 0, 0, width, height  # Full image bounding box
#
# # Draw bounding box around the detected disease portion
# cv2.rectangle(image_for_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# # Display the result
# cv2.putText(image_for_display, f"Disease: {decoded_predictions[0][1]}", (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#
# cv2.imshow("Plant Disease Detection", image_for_display)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# from keras.models import load_model
# import cv2
# import numpy as np
#
# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)
#
# # Load the model
# model = load_model("keras_model.h5", compile=False)
#
# # Load the labels
# class_names = open("labels.txt", "r").readlines()
#
# # Load the image for detection
# image_path = "/home/mayank/PycharmProjects/melomaniac2oo3/Plant.jpeg"  # Replace with the path to your image file
# image = cv2.imread(image_path)
# image_for_display = image.copy()  # Create a copy for display purposes
# image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
#
# # Display the input image
# cv2.imshow("Input Image", image_for_display)
# cv2.waitKey(0)
#
# # Reshape the image to match the model's input shape
# image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
#
# # Normalize the image array
# image = (image / 127.5) - 1
#
# # Predict the model
# prediction = model.predict(image)
# index = np.argmax(prediction)
# class_name = class_names[index]
# confidence_score = prediction[0][index]
#
# # Print prediction and confidence score
# print("Class:", class_name[2:])
# print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
#
# # Draw a rectangle around the detected region
# cv2.rectangle(image_for_display, (0, 0), (image_for_display.shape[1], image_for_display.shape[0]), (0, 255, 0), 2)
# cv2.putText(image_for_display, 'Disease Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#
# # Display the image with the bounding box
# cv2.imshow("Disease Detection Result", image_for_display)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#

# import cv2
# import numpy as np
# from tensorflow.keras.applications.densenet import CropNet, preprocess_input, decode_predictions
#
# # Load pre-trained MobileNetV2 model
# model = CropNet(weights='imagenet')
#
# # Load image
# image_path = "/home/mayank/PycharmProjects/melomaniac2oo3/Plant.jpeg"  # Replace with the path to your image file
# image = cv2.imread(image_path)
# image_for_display = image.copy()  # Create a copy for display purposes
#
# # Resize image for MobileNetV2 input size
# input_image = cv2.resize(image, (224, 224))
# input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
#
# # Preprocess the image for MobileNetV2 model
# input_image = preprocess_input(input_image)
#
# # Perform inference
# predictions = model.predict(input_image)
#
# # Decode and print predictions
# decoded_predictions = decode_predictions(predictions, top=3)[0]
# for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
#     print(f"{i + 1}: {label} ({score:.2f})")
#
# # Display the result
# cv2.putText(image_for_display, f"Disease: {decoded_predictions[0][1]}", (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#
# cv2.imshow("Plant Disease Detection", image_for_display)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import tensorflow_hub as hub

# Define the custom objects, including KerasLayer from TensorFlow Hub
custom_objects = {'KerasLayer': hub.KerasLayer}

# Load the trained model with the custom objects
model = load_model('cropnet_1.h5', custom_objects=custom_objects)

# Load an image for prediction
img_path = '/home/mayank/PycharmProjects/melomaniac2oo3/diseases4.jpeg'  # Replace with the path to your image
img = cv2.imread(img_path)
img_display = img.copy()  # Create a copy for display purposes
img = cv2.resize(img, (224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make predictions
predictions = model.predict(img_array)

# Get the class with the highest probability
predicted_class = np.argmax(predictions[0])

# Define class labels based on your dataset
class_labels = ['Cassava Bacterial Blight', 'Cassava Brown Streak Disease', 'Cassava Green Mottle', 'Cassava Mosaic Disease', 'Healthy']

# Print the predicted class label
print("Disease:", class_labels[predicted_class])

# Display the image with the predicted class label (larger display size)
img = cv2.resize(img, (800, 800))  # Adjust the size for display
cv2.putText(img, f"Plant Disease: {class_labels[predicted_class]}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Add a rectangular box around the detected disease region
# Note: Adjust these coordinates based on the actual coordinates of the detected region
box_coordinates = (100, 100, 500, 500)  # Format: (x1, y1, x2, y2)
cv2.rectangle(img, (box_coordinates[0], box_coordinates[1]), (box_coordinates[2], box_coordinates[3]), (0, 255, 0), 2)

cv2.imshow("Predicted Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
