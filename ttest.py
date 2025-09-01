from PIL import Image # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
import face_recognition # pyright: ignore[reportMissingImports]

img_path = "\..\known_faces\gyakie 1.jpg"  # replace with your file
img = Image.open(img_path)
print("Original mode:", img.mode)
img = img.convert("RGB")               # force RGB
img_array = np.array(img).astype(np.uint8)
print("Shape:", img_array.shape, "dtype:", img_array.dtype)

encodings = face_recognition.face_encodings(img_array)
print("Encodings found:", len(encodings))

