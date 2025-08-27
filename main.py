from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import face_recognition
import os
import shutil

# 1️⃣ Create FastAPI app instance
app = FastAPI()

# 2️⃣ Mount static folder for CSS, JS, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

# 3️⃣ Setup templates folder
templates = Jinja2Templates(directory="templates")

# 4️⃣ Folder to store known faces
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)


# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Register a new person
@app.post("/register/")
async def register_person(
    request: Request,
    name: str = Form(...),
    file: UploadFile = File(...)
):
    save_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    message = f"✅ {name} registered successfully!"
    return templates.TemplateResponse("index.html", {"request": request, "register_result": message})


# Compare faces with low tolerance and score
@app.post("/compare/")
async def compare_face(
    request: Request,
    mode: str = Form(...),
    file1: UploadFile = File(...),
    file2: UploadFile = File(None)
):
    # Load first image and encode
    img1 = face_recognition.load_image_file(file1.file)
    enc1 = face_recognition.face_encodings(img1)
    if not enc1:
        raise HTTPException(status_code=400, detail="No face found in first image")
    enc1 = enc1[0]

    result_text = ""
    TOLERANCE = 0.4  # Low tolerance for stricter comparison

    if mode == "db":
        best_match_name = None
        best_score = 1.0  # Lower distance = better match

        for person_file in os.listdir(KNOWN_FACES_DIR):
            known_img = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, person_file))
            known_encodings = face_recognition.face_encodings(known_img)
            if not known_encodings:
                continue

            distance = face_recognition.face_distance([known_encodings[0]], enc1)[0]
            if distance < TOLERANCE and distance < best_score:
                best_score = distance
                best_match_name = person_file.replace('.jpg', '')

        if best_match_name:
            result_text = f"✅ Match found: {best_match_name} (score: {1 - best_score:.2f})"
        else:
            result_text = "❌ No match found in database."

    elif mode == "upload":
        if not file2:
            raise HTTPException(status_code=400, detail="Second image required for mode='upload'")
        img2 = face_recognition.load_image_file(file2.file)
        enc2 = face_recognition.face_encodings(img2)
        if not enc2:
            raise HTTPException(status_code=400, detail="No face found in second image")
        enc2 = enc2[0]

        distance = face_recognition.face_distance([enc2], enc1)[0]
        match = distance < TOLERANCE
        result_text = f"✅ Images match! (score: {1 - distance:.2f})" if match else f"❌ Images do not match (score: {1 - distance:.2f})"

    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'db' or 'upload'")

    return templates.TemplateResponse("index.html", {"request": request, "compare_result": result_text})
