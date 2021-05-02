import face_recognition
import cv2
import numpy as np

cam = cv2.VideoCapture(0)

cv2.namedWindow("train")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        print(img_name)
        img_counter += 1

cam.release()

cv2.destroyAllWindows()


user_one_face = face_recognition.load_image_file(img_name)
user_one_face_encoding = face_recognition.face_encodings(user_one_face)[0]
known_face_encodings = [user_one_face_encoding]
known_face_names = ["user1"]
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        print(img_name)
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
face1 = face_recognition.load_image_file(img_name)
face_locations = face_recognition.face_locations(face1)
face_encodings = face_recognition.face_encodings(face1, face_locations)



for face_encoding in face_encodings:
    # See if the face is a match for the known face (that we saved in the precedent step)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    # name that we will give if the employee is not in the system
    name = "Unknown"
    # check the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    # Take the best one
    best_match_index = np.argmin(face_distances)
    # if we have a match:
    if matches[best_match_index]:
        # Give the detected face the name of the employee that match
        name = known_face_names[best_match_index]
    print(name)
