import cv2
import dlib
import numpy as np
import json
import os 

descriptor_file = "./recognizers/face_descriptors.npz"
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
face_recognizer_path = "dlib_face_recognition_resnet_model_v1.dat"

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_recognizer = dlib.face_recognition_model_v1(face_recognizer_path)

data = np.load(descriptor_file, allow_pickle=True)
saved_descriptors = data["descriptors"]
saved_ids = data["ids"]

mapping_file = "id_name_mapping.json"
if os.path.exists(mapping_file):
    with open(mapping_file, "r") as file:
        id_to_name = json.load(file)
    id_to_name = {int(k): v for k, v in id_to_name.items()}
else:
    id_to_name = {}

print("ID-Name mapping loaded:", id_to_name)

MAX_DISTANCE = 1.0

def calculate_similarity(distance, max_distance):
    similarity = max(0, (1 - distance / max_distance)) * 100
    return round(similarity, 2)

# Pornim camera
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot access the camera.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_frame)

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        shape = shape_predictor(rgb_frame, face)
        face_descriptor = np.array(face_recognizer.compute_face_descriptor(rgb_frame, shape))

        distances = np.linalg.norm(saved_descriptors - face_descriptor, axis=1)
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]

        if min_distance <= MAX_DISTANCE:
            id = saved_ids[min_distance_index]
            name = id_to_name.get(id, f"ID: {id}")
            similarity = calculate_similarity(min_distance, MAX_DISTANCE)
            label = f"{name} ({similarity}%)"
        else:
            label = "unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()