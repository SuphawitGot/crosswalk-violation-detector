"""
Video Detection Router Module

This module provides FastAPI API endpoints for video upload, real-time object/violation detection control,
and live video streaming.

How the Upload -> Start -> Stream Flow Works:
1. Upload (/upload): User uploads a video file. The server saves it to disk and automatically triggers detection.
2. Start (_do_start / /start): ML models load into memory and two background threads (Reader & Detector) are started.
3. Stream (/stream): The web browser connects to receive a continuous live MJPEG video stream of annotated frames.

Core Concepts Explained for Beginners:
- Threading: Allows Python to execute multiple tasks simultaneously in the background. Here, reading video frames
  and processing AI detection run on separate background threads so the web server stays responsive to user requests.
- Queues: Thread-safe storage boxes (buffers) used to pass video frames securely between threads without data corruption.
- MJPEG Streaming: Motion JPEG continuously streams individual JPEG image frames over HTTP (`multipart/x-mixed-replace`)
  so web browsers can render live video just like a camera feed.
- How Reader and Detector Threads Work Together:
  - Reader Thread reads raw video frames from disk at the video's own FPS and puts them into `_raw_q`.
  - Detector Thread grabs frames from `_raw_q`, runs YOLO model detection, converts processed frames to JPEG bytes,
    and publishes them as the "latest frame".
  - `_frame_generator()` waits for each new latest frame and yields it to the browser. Every browser tab gets
    every frame (the old version used a shared queue, so two viewers stole frames from each other).
- Stopping safely: every run gets its own `threading.Event` (a stop signal). Stopping sets the signal and WAITS for
  the threads to finish before a new run starts, so old threads can never keep running next to new ones.
- Playback mode (environment variable DETECT_REALTIME):
  - "1" (default): the video plays at its real speed. If detection is slower than the video, frames that the
    detector had no time for are skipped.
  - "0": every frame is processed. The video plays slower if detection can't keep up, but it is smooth and the
    vehicle tracker sees every frame.
"""

import sys
import os

# Add the relative path to the 'ML' directory so Python can locate and import the local Detector module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ML"))

import asyncio
import cv2
import queue
import threading
import time
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse

import Detector

# ── Router Setup ──
# API router instance grouping all video detection endpoints
router = APIRouter(tags=["detection"])

# ── Settings ──
REALTIME = os.getenv("DETECT_REALTIME", "1") != "0"
JPEG_QUALITY = 75

# ── Shared Global State ──
_video_path: str | None = None
_is_paused = False                              # Indicates whether processing is temporarily paused
_stop_event: threading.Event | None = None      # Stop signal of the current run (None = nothing running)
_workers: list[threading.Thread] = []           # Reader + detector threads of the current run
_run_id = 0                                     # Increases with every run, so old streams know to end
_video_looped = threading.Event()               # Reader -> detector: "the video restarted from frame 0"

# Only one start/stop/upload request may change the threads at a time
_control_lock = asyncio.Lock()

# Single-slot queue between reader and detector. It always holds the newest raw frame (BGR image).
_raw_q: queue.Queue = queue.Queue(maxsize=1)

# "Latest frame" slot shared with every browser stream.
# A Condition lets stream generators sleep until the detector publishes a new frame.
_frame_cond = threading.Condition()
_latest_jpeg: bytes | None = None
_frame_seq = 0                                  # Increases with every published frame

# Define folder paths for file uploads and saved violation captures
UPLOAD_DIR  = os.path.join(os.path.dirname(__file__), "..", "uploads")
CAPTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "captures")

# Ensure target storage directories exist on disk
os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs(CAPTURE_DIR, exist_ok=True)


def _running() -> bool:
    """True while a detection run is active."""
    return _stop_event is not None and not _stop_event.is_set()


# ── Thread 1: Frame Reader ─────────────────────────────────────────────────────
def _reader_thread(video_path: str, stop: threading.Event):
    """
    Reads frames from disk at the video's own FPS and pushes them into _raw_q.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[detect] Could not open video: {video_path}")
        stop.set()
        return

    # Some files report 0 or nonsense FPS; fall back to 30 in that case
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0
    frame_delay = 1.0 / fps

    # Fixed schedule: frame N is due at start + N * frame_delay.
    # (Sleeping "frame_delay minus this frame's work" lets small errors pile up and the video drifts.)
    next_due = time.perf_counter()
    failed_reads = 0

    try:
        while not stop.is_set():
            # If paused, wait and restart the clock afterwards so we don't rush to "catch up"
            if _is_paused:
                time.sleep(0.05)
                next_due = time.perf_counter()
                continue

            success, frame = cap.read()

            # End of video: loop back to frame 0 and tell the detector to reset its tracker
            if not success:
                failed_reads += 1
                if failed_reads > 2:
                    print("[detect] Video can't be read any more, stopping")
                    stop.set()
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _video_looped.set()
                continue
            failed_reads = 0

            if REALTIME:
                # Replace the frame the detector hasn't picked up yet, so the stream stays in time
                try:
                    _raw_q.get_nowait()
                except queue.Empty:
                    pass
                _raw_q.put(frame)
            else:
                # Wait until the detector takes the frame, so no frame is skipped
                while not stop.is_set():
                    try:
                        _raw_q.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        continue

            next_due += frame_delay
            delay = next_due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:
                # Far behind (e.g. waiting on the detector): don't try to catch up in a burst
                next_due = time.perf_counter()
    finally:
        cap.release()


# ── Thread 2: YOLO Detector ────────────────────────────────────────────────────
def _publish(jpeg_bytes: bytes):
    """Make a new frame the 'latest frame' and wake every stream that is waiting for it."""
    global _latest_jpeg, _frame_seq
    with _frame_cond:
        _latest_jpeg = jpeg_bytes
        _frame_seq += 1
        _frame_cond.notify_all()


def _detector_thread(stop: threading.Event):
    """
    Pulls raw frames, runs YOLO + crosswalk logic, encodes to JPEG and publishes it.
    """
    while not stop.is_set():
        if _is_paused:
            time.sleep(0.05)
            continue

        # Short timeout so we notice the stop signal quickly
        try:
            frame = _raw_q.get(timeout=0.1)
        except queue.Empty:
            continue

        # The video restarted: forget the cars tracked at the end of the video
        if _video_looped.is_set():
            _video_looped.clear()
            Detector.reset_tracking()

        try:
            annotated = Detector.process_frame(frame)
        except Exception as e:
            print(f"[detect] Detection error: {e}")
            continue
        if annotated is None:
            continue

        ret, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ret:
            continue

        # Never publish a frame that belongs to a run that was just stopped
        if stop.is_set():
            break
        _publish(jpeg.tobytes())


# ── MJPEG Stream Generator & Thread Controller ────────────────────────────────
def _frame_generator():
    """
    Yield every new annotated frame to one browser connection.
    Ends when detection stops or a new run (new upload) starts.
    """
    my_run = _run_id
    last_seq = -1

    while True:
        with _frame_cond:
            _frame_cond.wait_for(
                lambda: _frame_seq != last_seq or _run_id != my_run or not _running(),
                timeout=0.5,
            )
            finished = _run_id != my_run or not _running()
            jpeg_bytes, seq = _latest_jpeg, _frame_seq

        if finished:
            break
        if jpeg_bytes is None or seq == last_seq:
            continue
        last_seq = seq

        # Format binary chunk as standard multipart HTTP MJPEG frame boundary
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
        )


def _stop_workers(timeout: float = 10.0):
    """
    Signal the current run's threads to stop and WAIT for them to finish.
    Blocking, so endpoints call it through asyncio.to_thread.
    """
    global _stop_event, _workers, _is_paused
    stop, workers = _stop_event, _workers
    _stop_event, _workers = None, []
    _is_paused = False

    if stop is not None:
        stop.set()
    with _frame_cond:
        _frame_cond.notify_all()   # wake streams so they end now

    for t in workers:
        t.join(timeout)
        if t.is_alive():
            print(f"[detect] Warning: {t.name} did not stop within {timeout}s")


def _do_start():
    """
    Internal helper: initialise models and spawn worker threads for a new run.
    Blocking (model loading), so endpoints call it through asyncio.to_thread.
    """
    global _stop_event, _workers, _is_paused, _latest_jpeg, _run_id

    # Loads models the first time; afterwards only resets per-video state
    Detector.init_models()

    # Empty any stale frame left from the previous run
    _video_looped.clear()
    try:
        _raw_q.get_nowait()
    except queue.Empty:
        pass
    with _frame_cond:
        _latest_jpeg = None
        _run_id += 1
        _frame_cond.notify_all()

    stop = threading.Event()
    _is_paused = False
    _workers = [
        threading.Thread(target=_reader_thread, args=(_video_path, stop), name="reader", daemon=True),
        threading.Thread(target=_detector_thread, args=(stop,), name="detector", daemon=True),
    ]
    _stop_event = stop
    for t in _workers:
        t.start()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video and immediately start detection.
    """
    global _video_path

    async with _control_lock:
        # Fully stop any running detection before replacing the video file
        await asyncio.to_thread(_stop_workers)

        ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
        dest = os.path.join(UPLOAD_DIR, "input_video" + ext)

        contents = await file.read()
        with open(dest, "wb") as f:
            f.write(contents)
        _video_path = dest

        await asyncio.to_thread(_do_start)

    return {"message": "Video uploaded and detection started", "path": dest}


@router.post("/start")
async def start_detection():
    """
    Start detection on the currently uploaded video file if not already running.
    """
    async with _control_lock:
        if _video_path is None:
            return {"error": "No video uploaded"}
        if _running():
            return {"message": "Already running"}

        # Clean up threads of a run that ended by itself (e.g. unreadable video)
        await asyncio.to_thread(_stop_workers)
        await asyncio.to_thread(_do_start)
    return {"message": "Detection started"}


@router.post("/stop")
async def stop_detection():
    """
    Stop detection and wait for the background threads to finish.
    """
    async with _control_lock:
        await asyncio.to_thread(_stop_workers)
    return {"message": "Detection stopped"}


@router.post("/pause")
async def pause_detection():
    """
    Pause frame reading and detection processing without shutting down threads.
    """
    global _is_paused
    if not _running():
        return {"error": "Detection is not running"}
    _is_paused = True
    return {"message": "Detection paused"}


@router.post("/resume")
async def resume_detection():
    """
    Resume frame reading and detection processing after being paused.
    """
    global _is_paused
    if not _running():
        return {"error": "Detection is not running"}
    _is_paused = False
    return {"message": "Detection resumed"}


@router.get("/stream")
def stream_video():
    """
    Stream annotated JPEG frames continuously to the browser using HTTP MJPEG format.
    """
    return StreamingResponse(
        _frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@router.get("/violations")
def get_violations():
    """
    Return all auto-captured violation records (id, filename, plate, timestamp).
    """
    return {"violations": Detector.get_violations()}


@router.get("/status")
def get_status():
    """
    Return current detection status flags (running and paused state).
    """
    return {"running": _running(), "paused": _is_paused}
