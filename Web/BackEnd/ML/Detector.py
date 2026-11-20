"""
This file is the main brain of our computer vision system. It looks at video frames,
finds cars and crosswalks, and checks if a car is driving over the crosswalk.
If it catches a car in the wrong place, it reads the license plate and saves a record.
"""

# ── Section 1: Setup and Imports ──
# We import tools to help with AI (ultralytics YOLO), image processing (cv2), and math.
from ultralytics import YOLO  # type: ignore
import os
import sys
import cv2
import math
import threading
import time
import numpy as np

# Find where this file is located so we can find other files easily.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Violation pictures go to BackEnd/captures, the same folder main.py serves at /captures/<filename>.
# (They used to be saved in ML/captures, which the web server never served, so the images 404'd.)
CAPTURE_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "captures"))

# Add the parent folder to the system path so we can import our database and data models.
sys.path.insert(0, os.path.join(BASE_DIR, ".."))
from database import SessionLocal
from model import Violation

# ── Section 2: Global Variables and State ──

# YOLO (You Only Look Once) is an AI model that finds and draws boxes around objects in an image really fast.
# We will use three different YOLO models:
model = None             # Vehicle model: Detects cars and motorcycles.
crosswalk_model = None   # Crosswalk model: Finds where the crosswalk lines are.
plate_model = None       # License plate model: Finds the small rectangle of the license plate.

# OCR (Optical Character Recognition) is a technology that reads text from images.
# EasyOCR is a popular tool for this, and we use it because it can read both Thai and English.
_ocr_reader = None

# To make things run faster, we shrink the video frame by half (0.5) before running the heavy AI.
# We still draw the results on the high-quality original frame.
DETECT_SCALE = 0.5

# A list to keep track of cars that broke the rules.
# Each entry looks like: {"id": 1, "filename": "photo.jpg", "plate": "ABC 123", "timestamp": "..."}
_violations: list[dict] = []

# A Lock acts like a traffic light for our code. 
# Because different background tasks (threads) might try to add or change violations at the same time,
# the lock ensures only one task can edit the list at a time. This prevents errors and crashes.
_violations_lock = threading.Lock()

# Cooldown is a waiting period. 
# If a car triggers a violation, we wait a few seconds before capturing another one.
# This stops the system from taking 100 pictures of the exact same car in one second.
_last_capture_time: float = 0.0
CAPTURE_COOLDOWN = 3.0       # Wait 3 seconds between captures.

# A crosswalk is painted on the road, so it does not move between frames.
# Running the crosswalk model on every frame wastes time, so we only run it every N frames
# and reuse the last result in between.
CROSSWALK_EVERY_N_FRAMES = 15
_crosswalk_cache: list[tuple[int, int, int, int, float]] = []   # (x1, y1, x2, y2, conf)
_frame_idx = 0


# ── Section 3: Model Initialization ──

def init_models():
    """
    Load all the AI models (only the first time) and reset the per-video state.
    Called every time a new detection run starts, so it must be cheap after the first call.
    """
    global model, crosswalk_model, plate_model, _violations, _last_capture_time

    # Load our three YOLO models once. Re-loading them on every upload is slow
    # and throws away GPU warm-up.
    if model is None:
        model = YOLO(os.path.join(BASE_DIR, "..", "yolo26n.pt"))
        crosswalk_model = YOLO(os.path.join(BASE_DIR, "best.pt"))
        plate_model = YOLO(os.path.join(BASE_DIR, "LC.pt"))

        # Start loading the OCR reader in the background.
        # This takes a few seconds, so doing it in the background lets the rest of the app start fast.
        threading.Thread(target=_load_ocr, daemon=True).start()

    # Reset everything that belongs to the previous video
    _violations = []
    _last_capture_time = 0.0
    reset_tracking()


def reset_tracking():
    """
    Forget tracked vehicles and cached crosswalks.
    Call this when a new video starts or the current video loops back to the beginning,
    otherwise the tracker tries to match cars from the end of the video with the start.
    """
    global _frame_idx, _crosswalk_cache
    _frame_idx = 0
    _crosswalk_cache = []
    predictor = getattr(model, "predictor", None) if model is not None else None
    for tracker in getattr(predictor, "trackers", None) or []:
        tracker.reset()


def _load_ocr():
    """
    Loads EasyOCR in the background. We tell it to look for Thai ('th') and English ('en') text.
    """
    global _ocr_reader
    import easyocr
    # 'gpu=True' makes reading faster if a graphics card is available.
    _ocr_reader = easyocr.Reader(['th', 'en'], gpu=True, verbose=False)


# ── Section 4: Image Preprocessing ──

def _preprocess_plate(crop):
    """
    Cleans up the picture of the license plate so the OCR can read the text more easily.
    Small or blurry pictures are the main reason OCR fails.
    """
    h, w = crop.shape[:2]

    # 1. Enlarge: If the picture is too small (less than 200 pixels wide), we stretch it to make it bigger.
    if w < 200:
        scale = 200 / w
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale: Removes all color, leaving just black, white, and gray. 
    # This removes colorful shadows or reflections that confuse the AI.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization): 
    # This fixes bad lighting. It makes dark areas brighter and bright areas darker so the text pops out.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. Sharpening: Uses a math filter (kernel) to trace hard edges, making blurry letters crisp.
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened


# ── Section 5: OCR and Logging ──

def _run_ocr_and_log(crop, filename: str, violation_id: int):
    """
    Takes the picture of the license plate, reads the text, and saves it to the database.
    This runs in a background thread so it doesn't freeze the live video feed while it reads.
    """

    def _worker():
        plate_text = ""
        try:
            # If the OCR is still loading, we can't read the text yet.
            if _ocr_reader is None:
                return

            # Clean up the image first
            processed = _preprocess_plate(crop)

            # Ask EasyOCR to read the text in the cleaned-up image
            results = _ocr_reader.readtext(processed)
            best_prob = 0.0
            
            # Look at all the words it found and pick the one it is most confident about
            for (_, text, prob) in results:
                if prob > best_prob:
                    best_prob = prob
                    plate_text = text
                    
            # If it's less than 20% sure, it's probably garbage text, so throw it away.
            if best_prob < 0.2:
                plate_text = ""

            # Sometimes cleaning the image makes it worse. 
            # If we didn't find anything, try again on the original, colorful image just in case.
            if not plate_text:
                results2 = _ocr_reader.readtext(crop)
                for (_, text, prob) in results2:
                    if prob > best_prob:
                        best_prob = prob
                        plate_text = text
                if best_prob < 0.2:
                    plate_text = ""
        except Exception as e:
            print(f"[Detector] OCR error: {e}")

        # Format the text nicely (uppercase and remove extra spaces)
        final_plate = plate_text.upper().strip() if plate_text else "—"

        # Lock the list before we change it, so no other thread messes it up!
        with _violations_lock:
            # Find the violation we created earlier and fill in the plate text we just read.
            for v in _violations:
                if v["id"] == violation_id:
                    v["plate"] = final_plate
                    break

        # Save this final information to our permanent SQL database
        db = SessionLocal()
        try:
            violation = Violation(
                plate=final_plate,
                filename=filename,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            db.add(violation)
            db.commit()
        except Exception as e:
            # If saving fails, undo (rollback) to avoid breaking the database.
            db.rollback()
            print(f"[Detector] DB save error: {e}")
        finally:
            db.close()

    # Start the worker task in the background.
    threading.Thread(target=_worker, daemon=True).start()


# ── Section 6: License Plate Detection ──

def _detect_plate(car_crop):
    """
    Takes a picture of just a car, and uses the plate YOLO model to find exactly
    where the license plate is inside that car picture.

    Returns (plate_box, conf) with the box relative to car_crop, or (None, 0.0) if no plate
    was found with more than 30% confidence.
    """
    results = plate_model(car_crop, verbose=False)
    best_box = None
    best_conf = 0.0

    # Look through all the boxes it drew and pick the most confident one
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box

    if best_box is not None and best_conf > 0.3:
        px1, py1, px2, py2 = [int(v) for v in best_box.xyxy[0]]
        if px2 > px1 and py2 > py1:
            return (px1, py1, px2, py2), best_conf
    return None, 0.0


def get_violations() -> list[dict]:
    """
    Provides a safe way for other parts of the app to see the current list of violations.
    """
    with _violations_lock:
        return list(_violations)


# ── Section 7: Detection Pipeline ──

def process_frame(frame):
    """
    This is the main step-by-step pipeline that runs on every single video frame.

    Step 1: Resize the frame to make processing faster.
    Step 2: Find all crosswalks (only every few frames, they don't move).
    Step 3: Find and track all cars and motorcycles, and check which ones are on a crosswalk.
    Step 4: Find license plates, but ONLY for vehicles on a crosswalk (the only ones we need).
    Step 5: Draw all the boxes.
    Step 6: Save a picture and run OCR if a vehicle is on a crosswalk (and the cooldown allows it).

    Speed note: the plate model used to run on every vehicle in every frame, which meant
    5 cars = 7 YOLO runs per frame. Now a normal frame costs 1 YOLO run (the tracker).
    """
    global _last_capture_time, _frame_idx, _crosswalk_cache

    if frame is None:
        return None

    # Step 1: Resize for speed.
    # We run the AI on a smaller 'small' frame, but we will draw boxes on the big original 'frame'.
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (int(w * DETECT_SCALE), int(h * DETECT_SCALE)))

    # 'inv' is a multiplier to stretch the small coordinates back to the big frame's size.
    inv = 1.0 / DETECT_SCALE

    # Step 2: Detect crosswalks, but only every CROSSWALK_EVERY_N_FRAMES frames.
    if _frame_idx % CROSSWALK_EVERY_N_FRAMES == 0:
        found = []
        for result in crosswalk_model(small, verbose=False):
            for box in result.boxes:
                conf = float(box.conf[0])
                # Only count it if the AI is more than 50% sure it's a crosswalk
                if conf > 0.5:
                    x1, y1, x2, y2 = [int(v * inv) for v in box.xyxy[0]]
                    found.append((x1, y1, x2, y2, conf))
        _crosswalk_cache = found
    _frame_idx += 1
    crosswalk_boxes = _crosswalk_cache

    # Step 3: Detect & track vehicles.
    # Classes 2 and 3 mean cars and motorcycles in the standard YOLO model.
    # persist=True keeps the tracker's memory between frames so each car keeps its ID.
    vehicles = model.track(small, classes=[2, 3], verbose=False,
                           tracker="bytetrack.yaml", persist=True)

    now = time.perf_counter()
    can_capture = (now - _last_capture_time) > CAPTURE_COOLDOWN

    # Each entry: (x1, y1, x2, y2, conf, label, in_crosswalk)
    detections = []
    for result in vehicles:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            # Get the big-frame coordinates for the vehicle (kept inside the picture)
            x1, y1, x2, y2 = [int(v * inv) for v in box.xyxy[0]]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
            cls_id = int(box.cls[0]) if box.cls is not None else 2
            label = "Car" if cls_id == 2 else "Motorcycle"

            # The midpoint is the center of the car's bottom edge (where the tires touch the ground).
            # Is this midpoint inside any of the crosswalk boxes?
            mid_x, mid_y = (x1 + x2) // 2, y2
            in_crosswalk = any(
                cx1 <= mid_x <= cx2 and cy1 <= mid_y <= cy2
                for (cx1, cy1, cx2, cy2, _) in crosswalk_boxes
            )
            detections.append((x1, y1, x2, y2, conf, label, in_crosswalk))

    # Step 4: Detect license plates only for vehicles on a crosswalk.
    # This runs BEFORE drawing, so the crops we send to OCR have no boxes drawn on them.
    plate_boxes = {}      # detection index -> ((x1, y1, x2, y2), conf) on the big frame
    ocr_crop = None       # picture we will send to OCR if we capture a violation
    for i, (x1, y1, x2, y2, _, _, in_crosswalk) in enumerate(detections):
        if not in_crosswalk:
            continue
        car_crop = frame[y1:y2, x1:x2]
        if car_crop.size == 0:
            continue

        plate_box, plate_conf = _detect_plate(car_crop)
        plate_crop = None
        if plate_box is not None:
            # The plate's coordinates are relative to the car picture,
            # so add the car's position to place it on the big frame.
            px1, py1, px2, py2 = plate_box
            plate_boxes[i] = ((x1 + px1, y1 + py1, x1 + px2, y1 + py2), plate_conf)
            plate_crop = car_crop[py1:py2, px1:px2]

        if can_capture and ocr_crop is None:
            # Use the plate picture if we found one, otherwise the whole car
            ocr_crop = (plate_crop if plate_crop is not None and plate_crop.size > 0
                        else car_crop).copy()

    # Step 5: Draw everything on the big frame.
    for (x1, y1, x2, y2, conf) in crosswalk_boxes:
        # Blue rectangle around the crosswalk
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 2)
        cv2.putText(frame, f"Crosswalk {conf:.2f}", (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 160, 0), 2)

    for i, (x1, y1, x2, y2, conf, label, _) in enumerate(detections):
        # Green rectangle around the vehicle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (99, 49, 222), 2)
        # Red dot on the bottom-center point
        cv2.circle(frame, ((x1 + x2) // 2, y2), 6, (0, 0, 255), cv2.FILLED)

        if i in plate_boxes:
            # Cyan (light blue) box around the license plate
            (px1, py1, px2, py2), pconf = plate_boxes[i]
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 2)
            cv2.putText(frame, f"Plate {pconf:.2f}", (px1, max(py1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)

    # Step 6: Capture violation (if enough time has passed since the last one)
    if ocr_crop is not None:
        _last_capture_time = now

        # Save the picture to the folder the web server serves at /captures
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        filename = f"capture_{int(time.time() * 1000)}.jpg"
        cv2.imwrite(os.path.join(CAPTURE_DIR, filename), frame)

        # Add a temporary entry to our violation list saying "Reading..."
        # We use the lock so we don't mess up the list.
        with _violations_lock:
            vid = len(_violations) + 1
            _violations.append({
                "id":        vid,
                "filename":  filename,
                "plate":     "Reading…",
                "timestamp": ts,
            })

        # Send the plate picture to the background task to read the text
        if ocr_crop.size > 0:
            _run_ocr_and_log(ocr_crop, filename, vid)

        # Flash a red warning text on the video frame
        cv2.putText(frame, "! VIOLATION CAPTURED !", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # ── Overlay violations count on the screen ──
    with _violations_lock:
        count = len(_violations)
        latest_plate = _violations[-1]["plate"] if _violations else ""

    # Show how many cars broke the rule so far
    if count > 0:
        cv2.putText(frame, f"Violations: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
    # Show the text of the last license plate we successfully read
    if latest_plate and latest_plate not in ("Reading…", "—"):
        cv2.putText(frame, f"Plate: {latest_plate}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Return the fully drawn frame so it can be shown on screen
    return frame
