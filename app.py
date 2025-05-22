from flask import Flask, request, render_template
import cv2
import dlib
import numpy as np
import os
import tempfile
import base64

app = Flask(__name__)

# 模型路徑（請確保路徑正確）
predictor_path = r'C:\SIDE_PROJECT\FaceScore\shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def calculate_beauty_score(face_landmarks):
    """計算顏值分數的簡單函式"""
    eye_dist = np.linalg.norm(np.array(face_landmarks[36]) - np.array(face_landmarks[45]))
    mouth_dist = np.linalg.norm(np.array(face_landmarks[62]) - np.array(face_landmarks[66]))
    nose_dist = np.linalg.norm(np.array(face_landmarks[30]) - np.array(face_landmarks[33]))
    score = (eye_dist / 1.7) + (mouth_dist / 3.5) + (nose_dist / 1.7)
    return round(min(max(score, 0), 100), 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def score():
    if 'file' not in request.files:
        return render_template('index.html', error="請上傳圖片")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="沒有選擇圖片")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        file.save(tmp.name)
        image_path = tmp.name

    try:
        image = cv2.imread(image_path)
        if image is None:
            return render_template('index.html', error="圖片讀取失敗，請確認圖片格式")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            return render_template('index.html', error="沒有偵測到人臉")

        face = faces[0]
        landmarks = predictor(gray, face)
        face_landmarks = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

        score = calculate_beauty_score(face_landmarks)

        # 讀取圖片並轉 base64 編碼，用於網頁顯示
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        return render_template('index.html', score=score, img_data=img_base64)

    except Exception as e:
        return render_template('index.html', error=f"處理過程發生錯誤：{str(e)}")

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == '__main__':
    app.run(debug=True)
