from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)

registered_keypoints = None
registered_descriptors = None
pre_img = None

def preprocess_image(roi):
    if roi is None:
        return None
    
    gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    edge_x = cv2.filter2D(gray_image, -1, prewitt_x)

    prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    edge_y = cv2.filter2D(gray_image, -1, prewitt_y)

    edges = cv2.addWeighted(edge_x, 0.5, edge_y, 0.5, 0)

    _, binary_edges = cv2.threshold(edges, 3, 255, cv2.THRESH_BINARY)
    
    return binary_edges

def extract_features(pre_img):
    if pre_img is None:
        return None, None, None
    
    akaze = cv2.AKAZE_create(
        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,  # 特徴記述子の種類  
        threshold=0.005,  # より厳しい特徴点検出（精度向上）
        nOctaves=6,      # オクターブ数を増やして、より多様なスケールを捉える
    )

    keypoints, descriptors = akaze.detectAndCompute(pre_img, None)

    keypoint_image = cv2.drawKeypoints(pre_img, keypoints, None, color=(0, 255, 0))
    _, buffer = cv2.imencode('.png', keypoint_image)
    image_data = base64.b64encode(buffer).decode('utf-8')

    if descriptors is None:
        print("⚠ 特徴点が抽出されませんでした。")
        return None, None, image_data
    
    return keypoints, descriptors, image_data

def match_palm(new_keypoints, new_descriptors, keypoints, descriptors, new_pre_img, pre_img):
    if new_keypoints is None or new_descriptors is None:
        return False, None  # 特徴点が抽出されなかった場合

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(new_descriptors, descriptors, k=2)

    good_matches = ratio_test(matches)

    score = len(good_matches)
    print(f"一致スコア: {score}")

    if score < 10:  
        print("❌ 認証失敗。")
        return False, None

    if len(good_matches) >= 4:
        new_pts = np.float32([new_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(new_pts, pts, cv2.RANSAC, 3.0)
        matchesMask = mask.ravel().tolist()

        result_img = cv2.drawMatches(new_pre_img, new_keypoints, pre_img, keypoints, good_matches, None, matchesMask=matchesMask)
        _, buffer = cv2.imencode('.png', result_img)
        match_image_data = base64.b64encode(buffer).decode('utf-8')

        print("✅ 認証成功！")
        return True, match_image_data
    return False, None

def ratio_test(matches, ratio=0.7):
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    global registered_keypoints, registered_descriptors, pre_img
    data = request.get_json()
    image_data = base64.b64decode(data['image'].split(',')[1])
    image = Image.open(BytesIO(image_data))
    open_cv_image = np.array(image)[:, :, ::-1]

    pre_img = preprocess_image(open_cv_image)
    
    registered_keypoints, registered_descriptors, registered_image_data = extract_features(pre_img)

    if registered_keypoints is None:
        return jsonify({"message": "特徴点が抽出されませんでした"}), 400
    

    return jsonify({
        "message": "画像が登録されました",
        "registeredImage": f"data:image/png;base64,{registered_image_data}"
    })

@app.route('/match', methods=['POST'])
def match():
    global registered_keypoints, registered_descriptors, pre_img

    if registered_keypoints is None or registered_descriptors is None:
        return jsonify({"message": "⚠ 登録された特徴点がありません。まずは登録してください。"}), 400
    
    data = request.get_json()
    image_data = base64.b64decode(data['image'].split(',')[1])
    image = Image.open(BytesIO(image_data))
    open_cv_image = np.array(image)[:, :, ::-1]

    new_pre_img = preprocess_image(open_cv_image)

    new_keypoints, new_descriptors, match_image_data = extract_features(new_pre_img)

    if new_keypoints is None:
        return jsonify({"message": "特徴点が抽出されませんでした"}), 400

    result, result_image_data = match_palm(new_keypoints, new_descriptors, registered_keypoints, registered_descriptors, new_pre_img, pre_img)
    
    if not result:
        return jsonify({"message": "認証失敗しました。もう一度試してください。"}), 400

    return jsonify({
        "message": "認証成功！",
        "matchImage": f"data:image/png;base64,{match_image_data}",
        "resultImage": f"data:image/png;base64,{result_image_data}"
        })

if __name__ == "__main__":
    app.run(debug=True)