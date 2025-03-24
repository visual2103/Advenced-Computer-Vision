import dlib
import numpy as np
import os
from PIL import Image

## File paths
dataset_path = "./dataset/"
descriptor_file = "./recognizers/face_descriptors.npz"
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
face_recognizer_path = "dlib_face_recognition_resnet_model_v1.dat"

# load dlib models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_recognizer = dlib.face_recognition_model_v1(face_recognizer_path)

# variables for face descriptors and IDs
descriptors = []
ids = []

print("Processing images from the dataset...")

for user_folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, user_folder)
    if not os.path.isdir(folder_path):
        continue

    user_id = int(user_folder.split("_")[1])  # extract ID from folder name
    print(f"Processing images for ID {user_id}...")

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        try:
            # convert the image to grayscale and detect the face
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            faces = detector(image_np)

            if len(faces) != 1:
                print(f"Skipping {image_name} (expected one face, found: {len(faces)})")
                continue

            face = faces[0]
            shape = shape_predictor(image_np, face)
            face_descriptor = np.array(face_recognizer.compute_face_descriptor(image_np, shape))

            descriptors.append(face_descriptor)
            ids.append(user_id)
            print(f"Processed: {image_name}, ID: {user_id}")
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

# save descriptors and IDs
print("Saving descriptors and IDs...")
descriptors = np.array(descriptors)
ids = np.array(ids)
np.savez(descriptor_file, descriptors=descriptors, ids=ids)
print(f"Descriptors saved to {descriptor_file}.")
