from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import face_recognition
import os
import shutil
import uuid
from PIL import Image
import numpy as np
import logging
import time
import threading

# ========== Logging ==========
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ========== FastAPI app ==========
app = FastAPI()

# Static + Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Folders
KNOWN_FACES_DIR = "known_faces"
TEMP_DIR = "static/temp"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


# ========== Helper: Save UploadFile safely ==========
def save_upload_file(upload_file: UploadFile, folder: str) -> str:
    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(folder, filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    upload_file.file.close()
    return path


# ========== Helper: Load image safely ==========
def load_image_as_rgb(path: str) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img_array = np.array(img).astype(np.uint8)
        if img_array.ndim != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Unsupported image shape {img_array.shape}")
        return img_array


# ========== Helper: Crop and save faces ==========
def save_face_crop(image: np.ndarray, location, folder: str) -> str:
    top, right, bottom, left = location
    face_image = image[top:bottom, left:right]
    pil_img = Image.fromarray(face_image)
    filename = f"{uuid.uuid4()}_face.jpg"
    path = os.path.join(folder, filename)
    pil_img.save(path)
    return f"/static/temp/{filename}"


# ========== Home ==========
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ========== Compare Faces ==========
@app.post("/compare/")
async def compare_face(
    request: Request,
    mode: str = Form(...),                 # "db" or "upload"
    file1: UploadFile = File(...),         # first image (can be group)
    file2: UploadFile = File(None)         # second image (only for "upload")
):
    results = []
    try:
        logger.debug(f"Mode: {mode}")

        # Validate first file
        if not file1.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error1": "❌ First image must be JPG/PNG"
            })

        # Save + load first image
        image1_path = save_upload_file(file1, TEMP_DIR)
        img1 = load_image_as_rgb(image1_path)

        # Detect all faces in first image
        locations1 = face_recognition.face_locations(img1)
        encodings1 = face_recognition.face_encodings(img1, locations1)

        if not encodings1:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error1": "❌ No face detected in first image. Please upload another.",
                "img1_url": f"/static/temp/{os.path.basename(image1_path)}"
            })

        TOLERANCE = 0.5  # tweak as needed

        # === Database Mode ===
        if mode == "db":
            for idx, (enc1, loc) in enumerate(zip(encodings1, locations1)):
                face_crop_url = save_face_crop(img1, loc, TEMP_DIR)
                best_match = None
                best_score = 1.0
                matched_img_url = None

                for file in os.listdir(KNOWN_FACES_DIR):
                    known_path = os.path.join(KNOWN_FACES_DIR, file)
                    try:
                        known_img = load_image_as_rgb(known_path)
                        known_encs = face_recognition.face_encodings(known_img)
                        if not known_encs:
                            continue
                        dist = face_recognition.face_distance([known_encs[0]], enc1)[0]
                        if dist < TOLERANCE and dist < best_score:
                            best_score = dist
                            best_match = file.replace(".jpg", "")
                            db_copy_filename = f"{uuid.uuid4()}.jpg"
                            db_copy_path = os.path.join(TEMP_DIR, db_copy_filename)
                            shutil.copyfile(known_path, db_copy_path)
                            matched_img_url = f"/static/temp/{db_copy_filename}"
                    except Exception as e:
                        logger.error(f"Error processing {file}: {str(e)}")
                        continue

                if best_match:
                    score_value = round(1 - best_score, 2)
                    results.append({
                        "face": f"Face {idx+1}",
                        "crop_url": face_crop_url,
                        "result": f"✅ Match: {best_match}",
                        "score": score_value,
                        "matched_url": matched_img_url
                    })
                else:
                    results.append({
                        "face": f"Face {idx+1}",
                        "crop_url": face_crop_url,
                        "result": "❌ No match in database.",
                        "score": None,
                        "matched_url": None
                    })

        # === Upload Mode ===
        elif mode == "upload":
            if not file2:
                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "error2": "❌ Second image required",
                    "img1_url": f"/static/temp/{os.path.basename(image1_path)}"
                })

            if not file2.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "error2": "❌ Second image must be JPG/PNG",
                    "img1_url": f"/static/temp/{os.path.basename(image1_path)}"
                })

            image2_path = save_upload_file(file2, TEMP_DIR)
            img2 = load_image_as_rgb(image2_path)
            encodings2 = face_recognition.face_encodings(img2)

            if not encodings2:
                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "error2": "❌ No face detected in second image.",
                    "img1_url": f"/static/temp/{os.path.basename(image1_path)}",
                    "img2_url": f"/static/temp/{os.path.basename(image2_path)}"
                })

            enc2 = encodings2[0]  # only first face from second image
            for idx, (enc1, loc) in enumerate(zip(encodings1, locations1)):
                face_crop_url = save_face_crop(img1, loc, TEMP_DIR)
                dist = face_recognition.face_distance([enc2], enc1)[0]
                score_value = round(1 - dist, 2)

                if dist < TOLERANCE:
                    results.append({
                        "face": f"Face {idx+1}",
                        "crop_url": face_crop_url,
                        "result": f"✅ Match found (score: {score_value})",
                        "score": score_value,
                        "matched_url": f"/static/temp/{os.path.basename(image2_path)}"
                    })
                else:
                    results.append({
                        "face": f"Face {idx+1}",
                        "crop_url": face_crop_url,
                        "result": f"❌ No match (score: {score_value})",
                        "score": score_value,
                        "matched_url": f"/static/temp/{os.path.basename(image2_path)}"
                    })

        else:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Mode must be 'db' or 'upload'",
            })

        # Render template with results
        return templates.TemplateResponse("index.html", {
            "request": request,
            "results": results,
            "img1_url": f"/static/temp/{os.path.basename(image1_path)}"
        })

    except Exception as e:
        logger.error(f"compare_face error: {str(e)}", exc_info=True)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Server error: {str(e)}"
        })


# ========== Auto Cleanup ==========
def cleanup_temp_folder(interval: int = 3600, max_age: int = 86400):
    while True:
        now = time.time()
        for filename in os.listdir(TEMP_DIR):
            path = os.path.join(TEMP_DIR, filename)
            try:
                if os.path.isfile(path):
                    file_age = now - os.path.getmtime(path)
                    if file_age > max_age:
                        os.remove(path)
                        logger.info(f"🧹 Deleted old temp file: {path}")
            except Exception as e:
                logger.error(f"Cleanup error on {path}: {str(e)}")
        time.sleep(interval)


@app.on_event("startup")
def start_cleanup_task():
    thread = threading.Thread(target=cleanup_temp_folder, daemon=True)
    thread.start()
    logger.info("🧹 Auto-cleanup thread started (files older than 24h will be removed).")
