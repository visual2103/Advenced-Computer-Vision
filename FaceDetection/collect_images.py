# collect_images.py
import cv2
import os
import json

# create a folder for images
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# get user ID and name
user_id = input("Enter the person's ID (e.g., 1, 2, 3): ")
user_name = input("Enter the person's name: ")

# folder path for the user's ID
user_folder = f"dataset/user_{user_id}"
if not os.path.exists(user_folder):
    os.makedirs(user_folder)

# save ID-Name mapping to a JSON file -> standard 
mapping_file = "id_name_mapping.json"

# Load existing mappings if the file exists
try:
    with open(mapping_file, "r") as file:
        id_name_mapping = json.load(file)
except (FileNotFoundError, json.JSONDecodeError):
    id_name_mapping = {}

# add the new ID-Name pair
id_name_mapping[user_id] = user_name

# Save the updated mapping
with open(mapping_file, "w") as file:
    json.dump(id_name_mapping, file)

print(f"ID {user_id} and name '{user_name}' saved to {mapping_file}.")

# capture images
video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
count = 0

print("Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Cannot access the camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y + h, x:x + w]
        # save the images in the specific folder for the user ID
        cv2.imwrite(f"{user_folder}/{user_name}_{count}.jpg", face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Capturing Images", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:  # Max 100 images
        break

video_capture.release()
cv2.destroyAllWindows()
print(f"{count} images saved for {user_name} in folder '{user_folder}'.")

