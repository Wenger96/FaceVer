from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
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
        img = img.resize((500, 500), Image.Resampling.LANCZOS)
        img_array = np.array(img).astype(np.uint8)
        if img_array.ndim != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Unsupported image shape {img_array.shape}")
        return img_array


# ========== Home ==========
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ========== Compare Faces ==========
@app.post("/compare/")
async def compare_face(
    request: Request,
    mode: str = Form(...),                 # "db" or "upload"
    file1: UploadFile = File(...),         # first image
    file2: UploadFile = File(None)         # second image (only for "upload")
):
    image1_path = image2_path = db_copy_path = None
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
        enc1 = face_recognition.face_encodings(img1)

        if not enc1:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error1": "❌ No face detected in first image. Please upload another.",
                "img1_url": f"/static/temp/{os.path.basename(image1_path)}"
            })
        enc1 = enc1[0]

        # Result vars
        result_text = ""
        img2_url = None
        score_value = None
        TOLERANCE = 0.5   # slightly relaxed for glasses/angles

        # === Database Mode ===
        if mode == "db":
            best_match = None
            best_score = 1.0

            for file in os.listdir(KNOWN_FACES_DIR):
                known_path = os.path.join(KNOWN_FACES_DIR, file)
                try:
                    known_img = load_image_as_rgb(known_path)
                    known_encs = face_recognition.face_encodings(known_img)
                    if not known_encs:
                        logger.warning(f"No face in {file}")
                        continue

                    dist = face_recognition.face_distance([known_encs[0]], enc1)[0]
                    if dist < TOLERANCE and dist < best_score:
                        best_score = dist
                        best_match = file.replace(".jpg", "")
                        db_copy_filename = f"{uuid.uuid4()}.jpg"
                        db_copy_path = os.path.join(TEMP_DIR, db_copy_filename)
                        shutil.copyfile(known_path, db_copy_path)
                        img2_url = f"/static/temp/{db_copy_filename}"
                except Exception as e:
                    logger.error(f"Error processing {file}: {str(e)}")
                    continue

            if best_match:
                score_value = round(1 - best_score, 2)
                result_text = f"✅ Match: {best_match} (score: {score_value})"
            else:
                result_text = "❌ No match in database."

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
            enc2 = face_recognition.face_encodings(img2)

            if not enc2:
                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "error2": "❌ No face detected in second image. Please upload another.",
                    "img1_url": f"/static/temp/{os.path.basename(image1_path)}",
                    "img2_url": f"/static/temp/{os.path.basename(image2_path)}"
                })

            enc2 = enc2[0]
            dist = face_recognition.face_distance([enc2], enc1)[0]
            score_value = round(1 - dist, 2)

            if dist < TOLERANCE:
                result_text = f"✅ Images match (score: {score_value})"
            else:
                result_text = f"❌ Images do not match (score: {score_value})"

            img2_url = f"/static/temp/{os.path.basename(image2_path)}"

        else:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Mode must be 'db' or 'upload'",
            })

        # Render template with results
        return templates.TemplateResponse("index.html", {
            "request": request,
            "compare_result": result_text,
            "img1_url": f"/static/temp/{os.path.basename(image1_path)}",
            "img2_url": img2_url,
            "score": score_value
        })

    except Exception as e:
        logger.error(f"compare_face error: {str(e)}", exc_info=True)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Server error: {str(e)}"
        })
