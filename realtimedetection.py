import cv2
from keras.models import model_from_json
import numpy as np
from keras.models import Sequential  # Ensure Sequential is imported

# Load the pre-trained model
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()

# Load the model using custom_objects
model = model_from_json(model_json, custom_objects={"Sequential": Sequential})
model.load_weights("facialemotionmodel.h5")

# Initialize Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Feature extraction function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start webcam
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    if not ret:
        print("Error accessing webcam.")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (p, q, r, s) in faces:
        face = gray[q:q + s, p:p + r]
        cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)

        try:
            face = cv2.resize(face, (48, 48))
            img = extract_features(face)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(im, prediction_label, (p, q - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error processing face: {e}")

    cv2.imshow("Output", im)

    if cv2.waitKey(27) & 0xFF == 27:  # Press 'Esc' to exit
        break

webcam.release()
cv2.destroyAllWindows()
