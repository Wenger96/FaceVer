from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import face_recognition
import os
import shutil
import uuid

# 1️⃣ Create FastAPI app instance
app = FastAPI()

# 2️⃣ Mount static folder for CSS, JS, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

# 3️⃣ Setup templates folder
templates = Jinja2Templates(directory="templates")

# 4️⃣ Folders
KNOWN_FACES_DIR = "known_faces"
TEMP_DIR = "static/temp"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 🚀 Compare faces with tolerance + score + side-by-side image display
@app.post("/compare/")
async def compare_face(
    request: Request,
    mode: str = Form(...),
    file1: UploadFile = File(...),
    file2: UploadFile = File(None)
):
    # Save first image temporarily
    image1_filename = f"{uuid.uuid4()}.jpg"
    image1_path = os.path.join(TEMP_DIR, image1_filename)
    with open(image1_path, "wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)

    img1 = face_recognition.load_image_file(image1_path)
    enc1 = face_recognition.face_encodings(img1)
    if not enc1:
        raise HTTPException(status_code=400, detail="No face found in first image")
    enc1 = enc1[0]

    result_text = ""
    TOLERANCE = 0.4
    image2_filename = None

    if mode == "db":
        best_match_name = None
        best_score = 1.0  

        for person_file in os.listdir(KNOWN_FACES_DIR):
            known_img_path = os.path.join(KNOWN_FACES_DIR, person_file)
            known_img = face_recognition.load_image_file(known_img_path)
            known_encodings = face_recognition.face_encodings(known_img)
            if not known_encodings:
                continue

            distance = face_recognition.face_distance([known_encodings[0]], enc1)[0]
            if distance < TOLERANCE and distance < best_score:
                best_score = distance
                best_match_name = person_file.replace('.jpg', '')
                image2_filename = person_file  # show the matching DB image

        if best_match_name:
            result_text = f"✅ Match found: {best_match_name} (score: {1 - best_score:.2f})"
        else:
            result_text = "❌ No match found in database."

    elif mode == "upload":
        if not file2:
            raise HTTPException(status_code=400, detail="Second image required for mode='upload'")

        # Save second image temporarily
        image2_filename = f"{uuid.uuid4()}.jpg"
        image2_path = os.path.join(TEMP_DIR, image2_filename)
        with open(image2_path, "wb") as buffer:
            shutil.copyfileobj(file2.file, buffer)

        img2 = face_recognition.load_image_file(image2_path)
        enc2 = face_recognition.face_encodings(img2)
        if not enc2:
            raise HTTPException(status_code=400, detail="No face found in second image")
        enc2 = enc2[0]

        distance = face_recognition.face_distance([enc2], enc1)[0]
        match = distance < TOLERANCE
        result_text = (
            f"✅ Images match! (score: {1 - distance:.2f})"
            if match
            else f"❌ Images do not match (score: {1 - distance:.2f})"
        )

    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'db' or 'upload'")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "compare_result": result_text,
        "image1": f"temp/{image1_filename}",      # ✅ pass to HTML
        "image2": f"temp/{image2_filename}" if image2_filename else None
    })
