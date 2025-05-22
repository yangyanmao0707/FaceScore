import cv2
import dlib
import numpy as np
import os

path = r'C:\SIDE_PROJECT\shape_predictor_68_face_landmarks.dat'
print("File exists:", os.path.exists(path))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

def calculate_beauty_score(face_landmarks):
    eye_dist = np.linalg.norm(np.array(face_landmarks[36]) - np.array(face_landmarks[45]))
    mouth_dist = np.linalg.norm(np.array(face_landmarks[62]) - np.array(face_landmarks[66]))
    nose_dist = np.linalg.norm(np.array(face_landmarks[30]) - np.array(face_landmarks[33]))
    
    score = (eye_dist / 1.7) + (mouth_dist / 3.5) + (nose_dist / 1.7)
    score = min(max(0, score), 100)
    return score

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("無法讀取圖片，請確認路徑是否正確")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        print("未檢測到人臉")
        return []

    scores = []
    for face in faces:
        landmarks = predictor(gray, face)
        face_landmarks = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
        score = calculate_beauty_score(face_landmarks)
        scores.append(score)
        
        cv2.putText(image, f' Score: {score:.2f}', (face.left(), face.top()-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        for (x, y) in face_landmarks:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow('Face Score', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return scores

image_path = r"C:\SIDE_PROJECT\image.jpg"
scores = process_image(image_path)
if scores:
    for i, s in enumerate(scores):
        print(f"Face {i+1} Score: {s:.2f}")
else:
    print("沒有計算到分數。")
