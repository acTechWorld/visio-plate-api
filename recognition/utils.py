import cv2
import pytesseract
import os
from uuid import uuid4
from ultralytics import YOLO
from django.conf import settings

# Tesseract path for macOS
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

CUSTOM_MODEL_PATH = "license_plate_detector.pt"  # <-- update this
yolo_model = YOLO(CUSTOM_MODEL_PATH)

# Ensure plate folder exists
PLATE_SAVE_DIR = os.path.join(settings.MEDIA_ROOT, "plates")
os.makedirs(PLATE_SAVE_DIR, exist_ok=True)


def detect_plate_yolo(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return [{"plate_number": "Image unreadable", "bbox": None, "crop_url": None}]

    results = yolo_model(image)
    detected_plates = []

    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            plate_crop = image[y1:y2, x1:x2]

            # Save crop
            crop_filename = f"plate_{uuid4().hex}.jpg"
            crop_path = os.path.join(PLATE_SAVE_DIR, crop_filename)
            cv2.imwrite(crop_path, plate_crop)

            # OCR
            plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(
                plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            ocr_config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            plate_text = pytesseract.image_to_string(
                thresh, config=ocr_config).strip()

            # Construct crop URL
            crop_url = f"{settings.MEDIA_URL}plates/{crop_filename}"

            detected_plates.append({
                "plate_number": plate_text if plate_text else "Not recognized",
                "bbox": [x1, y1, x2, y2],
                "crop_url": crop_url
            })

    if not detected_plates:
        return [{"plate_number": "License plate not detected", "bbox": None, "crop_url": None}]

    return detected_plates
