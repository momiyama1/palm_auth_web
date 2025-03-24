from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import logging

app = Flask(__name__)

registered_keypoints = None
registered_descriptors = None
pre_img = None

# 画像処理のパラメータを定数として定義
# これにより調整が容易になり、コードの可読性も向上します
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
GAUSSIAN_KERNEL_SIZE = (7, 7)
THRESHOLD_VALUE = 6
MIN_MATCH_POINTS = 5
MATCH_RATIO = 0.7
AKAZE_THRESHOLD = 0.001


# ログの設定を追加して、デバッグと監視を容易に
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_image(roi):
    """
    手のひら画像の前処理を行う関数
    - 定数を使用してパラメータを管理
    """
    # 入力画像がNoneの場合は処理を中止
    if roi is None:
        return None
   
    # グレースケール変換: 色情報を排除し、明暗情報だけを残す
    gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 正規化処理: 画像の明暗の範囲を0から255にスケーリングして、コントラストを強調
    normalize_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # CLAHEによるコントラストの強調
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    clahe_image = clahe.apply(normalize_image)

    # ガウシアンブラー: 画像を平滑化してノイズを減らす
    Gaussian_image = cv2.GaussianBlur(clahe_image, GAUSSIAN_KERNEL_SIZE, 0)

    # Prewittフィルタを用いたエッジ検出:
    # X方向のエッジ検出
    prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    edge_x = cv2.filter2D(Gaussian_image, -1, prewitt_x)

    # Y方向のエッジ検出
    prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    edge_y = cv2.filter2D(Gaussian_image, -1, prewitt_y)

    # X方向とY方向のエッジを合成:
    edges = cv2.addWeighted(edge_x, 0.5, edge_y, 0.5, 0)
    
    # 二値化処理:
    _, binary_edges = cv2.threshold(edges, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    
    return binary_edges

def extract_features(pre_img):
    """
    前処理済み画像から特徴点と特徴量を抽出する関数
    AKAZEアルゴリズムを使用して特徴点を検出
    """
    if pre_img is None:
        return None, None, None
    
    # AKAZE特徴量検出器の設定
    akaze = cv2.AKAZE_create(
        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,  # 特徴記述子の種類  
        threshold=AKAZE_THRESHOLD,  # より厳しい特徴点検出（精度向上）
        nOctaves=6,      # オクターブ数を増やして、より多様なスケールを捉える
    )

    # 特徴点と特徴量の抽出
    keypoints, descriptors = akaze.detectAndCompute(pre_img, None)

    # # ORBを使用
    # orb = cv2.ORB_create(nfeatures=500)  # ORBを使用
    # keypoints, descriptors = orb.detectAndCompute(pre_img, None)

    # 特徴点を可視化
    keypoint_image = cv2.drawKeypoints(pre_img, keypoints, None, color=(0, 255, 0))
    _, buffer = cv2.imencode('.png', keypoint_image)
    match_image_data = base64.b64encode(buffer).decode('utf-8')

    if descriptors is None:
        print("⚠ 特徴点が抽出されませんでした。")
        return None, None, match_image_data
    
    return keypoints, descriptors, match_image_data

def match_palm(new_keypoints, new_descriptors, keypoints, descriptors, new_pre_img, pre_img):
    """
    手のひら認証を実行する関数
    - ログ機能を追加してデバッグを容易に
    """
    if new_keypoints is None or new_descriptors is None:
        logging.warning("特徴点が見つかりません")
        return False, None

    # Brute Force Matcherの設定
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors, new_descriptors, k=2)

    # 比率テストで良いマッチングを抽出
    good_matches = ratio_test(matches)

    if len(good_matches) >= 4:
        # ホモグラフィ計算のための点の準備
        new_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts = np.float32([new_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # RANSACでホモグラフィを計算
        H, mask = cv2.findHomography(pts, new_pts, cv2.RANSAC, 3.0)
        matchesMask = mask.ravel().tolist()

        # マッチした点の数をカウント
        matched_points = sum(matchesMask)
        logging.info(f"マッチした特徴点の数: {matched_points}")

        if matched_points <= MIN_MATCH_POINTS:
            logging.warning("認証失敗: マッチング点数不足")
            return False, None

        # マッチング結果の可視化
        result_img = cv2.drawMatches(pre_img, keypoints, new_pre_img, new_keypoints,good_matches, None, matchesMask=matchesMask)
        _, buffer = cv2.imencode('.png', result_img)
        result_image_data = base64.b64encode(buffer).decode('utf-8')

        logging.info("認証成功")
        return True, result_image_data
    return False, None

def ratio_test(matches, ratio=MATCH_RATIO):
    """
    比率テストを実行して、良いマッチングを抽出する関数
    ratio: 距離の比率の閾値
    """
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

@app.route('/')
def index():
    """メインページを表示"""
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    """
    手のひら画像を登録するエンドポイント
    - エラーハンドリングを追加して安定性を向上
    - 入力データの検証を追加
    """
    try:
        global registered_keypoints, registered_descriptors, pre_img
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"message": "画像データが見つかりません"}), 400

        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(BytesIO(image_data))
        rgb_image = np.array(image)
        open_cv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # 画像の前処理
        pre_img = preprocess_image(open_cv_image)
        
        # 特徴点と特徴量の抽出
        registered_keypoints, registered_descriptors, registered_image_data = extract_features(pre_img)

        if registered_keypoints is None:
            return jsonify({"message": "特徴点が抽出されませんでした"}), 400
        

        return jsonify({
            "message": "画像が登録されました",
            "registeredImage": f"data:image/png;base64,{registered_image_data}"
        })

    except Exception as e:
        return jsonify({"message": f"エラーが発生しました: {str(e)}"}), 500

@app.route('/match', methods=['POST'])
def match():
    """手のひら認証を実行するエンドポイント"""
    global registered_keypoints, registered_descriptors, pre_img

    if registered_keypoints is None or registered_descriptors is None:
        return jsonify({"message": "⚠ 登録された特徴点がありません。まずは登録してください。"}), 400
    
    data = request.get_json()
    image_data = base64.b64decode(data['image'].split(',')[1])
    image = Image.open(BytesIO(image_data))
    rgb_image = np.array(image)
    open_cv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    # 新しい画像の前処理
    new_pre_img = preprocess_image(open_cv_image)

    # 特徴点と特徴量の抽出
    new_keypoints, new_descriptors, match_image_data = extract_features(new_pre_img)

    if new_keypoints is None:
        return jsonify({"message": "特徴点が抽出されませんでした"}), 400

    # 手のひら照合の実行
    result, result_image_data = match_palm(new_keypoints, new_descriptors, registered_keypoints, registered_descriptors, new_pre_img, pre_img)
    
    if not result:
        return jsonify({
            "message": "認証失敗しました。もう一度試してください。",
            "matchImage": f"data:image/png;base64,{match_image_data}" 
        }), 400

    return jsonify({
        "message": "認証成功！",
        "matchImage": f"data:image/png;base64,{match_image_data}",
        "resultImage": f"data:image/png;base64,{result_image_data}"
        })  

if __name__ == "__main__":
    app.run(debug=True)