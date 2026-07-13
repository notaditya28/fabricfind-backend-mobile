"""
FabricFind Mobile Backend — Fixed routes to match mobile app API calls
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import os
import re
import uuid
from PIL import Image, ImageEnhance
import io

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fabricfind_model.onnx")

CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_API_KEY    = os.environ.get("CLOUDINARY_API_KEY", "")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET", "")
FIREBASE_PROJECT_ID   = os.environ.get("FIREBASE_PROJECT_ID", "fabricfind-a8f7f")

session = None

def load_model():
    global session
    if session is not None:
        return
    import onnxruntime as ort
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print(f"✓ Model loaded. Output: {session.get_outputs()[0].shape}")

def upload_to_cloudinary(img: Image.Image, uid: str, entry_id: str) -> dict:
    import cloudinary
    import cloudinary.uploader
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
    )
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=85)
    buf.seek(0)
    result = cloudinary.uploader.upload(
        buf,
        folder=f"fabricfind/{uid}",
        public_id=entry_id,
        resource_type="image",
    )
    return {"url": result["secure_url"], "public_id": result["public_id"]}

def delete_from_cloudinary(public_id: str):
    import cloudinary
    import cloudinary.uploader
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
    )
    cloudinary.uploader.destroy(public_id)

def get_firestore():
    import firebase_admin
    from firebase_admin import credentials, firestore
    if not firebase_admin._apps:
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccount.json")
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def verify_token(id_token: str) -> str:
    from firebase_admin import auth
    get_firestore()
    decoded = auth.verify_id_token(id_token)
    return decoded["uid"]

def find_border_lines(arr, threshold_frac=0.5, mode="dark"):
    sh, sw = arr.shape
    mask = (arr < 60) if mode == "dark" else (arr > 200)
    row_count = mask.sum(axis=1)
    col_count = mask.sum(axis=0)
    border_rows = np.where(row_count > sw * threshold_frac)[0]
    border_cols = np.where(col_count > sh * threshold_frac)[0]
    return border_rows, border_cols

def try_extract_box(search_region, border_rows, border_cols, sh, sw):
    if len(border_rows) < 2 or len(border_cols) < 2:
        return None
    top, bottom = border_rows.min(), border_rows.max()
    left, right = border_cols.min(), border_cols.max()
    if (bottom - top) < sh * 0.08 or (right - left) < sw * 0.15:
        return None
    margin_y = max(2, int((bottom - top) * 0.03))
    margin_x = max(2, int((right - left) * 0.03))
    inner = search_region.crop((left + margin_x, top + margin_y, right - margin_x, bottom - margin_y))
    iw, ih = inner.size
    if iw == 0 or ih == 0:
        return None
    if not (0.4 < iw / ih < 2.5):
        return None
    return inner

def find_swatch_by_border(img: Image.Image):
    w, h = img.size
    search_region = img.crop((int(w * 0.45), 0, w, h))
    arr = np.array(search_region.convert("L"))
    sh, sw = arr.shape
    for threshold in (0.5, 0.3):
        for mode in ("dark", "light"):
            rows, cols = find_border_lines(arr, threshold_frac=threshold, mode=mode)
            result = try_extract_box(search_region, rows, cols, sh, sw)
            if result is not None:
                return result
    return None

def crop_catalogue_swatch(img: Image.Image) -> Image.Image:
    detected = find_swatch_by_border(img)
    if detected is not None:
        return detected
    w, h = img.size
    return img.crop((int(w*0.58), int(h*0.15), int(w*0.97), int(h*0.42)))

def get_search_patches(img: Image.Image) -> list:
    w, h = img.size
    pw, ph = int(w * 0.5), int(h * 0.5)
    patches = []
    for row in range(3):
        for col in range(3):
            x = int(col * (w - pw) / 2)
            y = int(row * (h - ph) / 2)
            patches.append(img.crop((x, y, x+pw, y+ph)))
    patches.append(img.rotate(90, expand=True))
    return patches

def enhance(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    return img

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    arr  = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, axis=0)

def extract_vector(img: Image.Image) -> list:
    load_model()
    arr = preprocess(img)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: arr})
    return outputs[0].flatten().tolist()

def cosine_similarity(a, b):
    return float(np.dot(np.array(a), np.array(b)))

def decode_image(s: str) -> Image.Image:
    b64 = s.split(",")[1] if s.startswith("data:") else s
    return Image.open(io.BytesIO(base64.b64decode(b64)))

def extract_design_number(img: Image.Image) -> str:
    try:
        import pytesseract
        w, h = img.size
        region = img.crop((int(w * 0.45), 0, w, int(h * 0.25)))
        region = region.resize((region.width * 2, region.height * 2), Image.LANCZOS)
        region = region.convert("L")
        region = ImageEnhance.Contrast(region).enhance(2.0)
        text = pytesseract.image_to_string(region, config="--psm 11")
        matches = re.findall(r'\b\d{3,6}\b', text)
        return max(matches, key=len) if matches else ""
    except Exception:
        return ""

def get_uid_from_request() -> str:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise ValueError("Missing or invalid Authorization header")
    id_token = auth_header[7:]
    return verify_token(id_token)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    load_model()
    return jsonify({"status": "FabricFind Mobile Backend running"})

@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json()
        img = decode_image(data["image"])
        img = enhance(img)
        patches = get_search_patches(img)
        patch_vectors = [extract_vector(p) for p in patches]
        results = []
        for item in data["library"]:
            if "vector" not in item or "id" not in item:
                continue
            best = max(cosine_similarity(pv, item["vector"]) for pv in patch_vectors)
            results.append({"id": item["id"], "score": round(best, 4)})
        results.sort(key=lambda x: x["score"], reverse=True)
        return jsonify({"results": results})
    except Exception as e:
        print(f"Error in /search: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/add", methods=["POST"])
def add_pattern():
    try:
        uid = get_uid_from_request()
        data = request.get_json()
        img = decode_image(data["image"])
        enhanced = enhance(img)
        swatch = crop_catalogue_swatch(enhanced)
        vector = extract_vector(swatch)
        design_no = data.get("designNo", "").strip()
        if not design_no:
            design_no = extract_design_number(img)
        entry_id = str(uuid.uuid4())
        cloud = upload_to_cloudinary(img, uid, entry_id)
        db = get_firestore()
        import time
        entry = {
            "id": entry_id,
            "name": data.get("name", "Untitled"),
            "designNo": design_no,
            "vector": vector,
            "photoUrl": cloud["url"],
            "cloudinaryPublicId": cloud["public_id"],
            "createdAt": int(time.time() * 1000),
        }
        db.collection("users").document(uid).collection("patterns").document(entry_id).set(entry)
        return jsonify({"id": entry_id, "name": entry["name"],
                       "designNo": entry["designNo"], "photoUrl": entry["photoUrl"]})
    except Exception as e:
        print(f"Error in /add: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/library", methods=["GET"])
def get_library():
    try:
        uid = get_uid_from_request()
        db = get_firestore()
        docs = db.collection("users").document(uid).collection("patterns")\
                 .order_by("createdAt", direction="DESCENDING").stream()
        patterns = []
        for doc in docs:
            d = doc.to_dict()
            patterns.append({
                "id": d["id"], "name": d["name"],
                "designNo": d.get("designNo", ""),
                "photoUrl": d["photoUrl"], "createdAt": d["createdAt"],
            })
        return jsonify({"library": patterns})
    except Exception as e:
        print(f"Error in /library: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/library/full", methods=["GET"])
def get_library_full():
    try:
        uid = get_uid_from_request()
        db = get_firestore()
        docs = db.collection("users").document(uid).collection("patterns").stream()
        patterns = [{"id": d.to_dict()["id"], "vector": d.to_dict()["vector"]} for d in docs]
        return jsonify({"library": patterns})
    except Exception as e:
        print(f"Error in /library/full: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/delete/<entry_id>", methods=["DELETE"])
def delete_pattern(entry_id):
    try:
        uid = get_uid_from_request()
        db = get_firestore()
        doc_ref = db.collection("users").document(uid).collection("patterns").document(entry_id)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            if "cloudinaryPublicId" in data:
                delete_from_cloudinary(data["cloudinaryPublicId"])
            doc_ref.delete()
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error in /delete: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/update-design/<entry_id>", methods=["PATCH"])
def update_design_no(entry_id):
    try:
        uid = get_uid_from_request()
        data = request.get_json()
        db = get_firestore()
        db.collection("users").document(uid).collection("patterns")\
          .document(entry_id).update({"designNo": data.get("designNo", "")})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting FabricFind Mobile Backend")
    load_model()
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
