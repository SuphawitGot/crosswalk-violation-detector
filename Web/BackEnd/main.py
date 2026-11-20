"""
Main Application Entrypoint.

This file sets up and configures the FastAPI web application server.
- FastAPI: A modern Python web framework for building APIs efficiently.
- Middleware / CORS: CORS (Cross-Origin Resource Sharing) middleware allows frontend clients (like React at localhost:3000) to safely talk to backend APIs running on a different port.
- Static File Mounting: Exposes a local folder to web clients so images or files can be accessed via URL (e.g., /captures/image.jpg).
- Routers: Modular components used to split routes/endpoints into separate files (e.g., auth routes, detection routes).
"""

# ── Imports ──
import os
from Rounter import auth, detect
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from database import engine
from model import Base

# ── Application Setup ──
# Create the main FastAPI application instance
app = FastAPI()

# Automatically create all database tables defined in model.py if they don't exist yet
Base.metadata.create_all(bind=engine)

# ── Middleware Configuration (CORS) ──
# Configure CORS to allow frontend web apps to make API requests to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static File Mounting ──
# Serve captured violation images as static files at /captures/<filename>
CAPTURE_DIR = os.path.join(os.path.dirname(__file__), "captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)
app.mount("/captures", StaticFiles(directory=CAPTURE_DIR), name="captures")


# ── Health Check Endpoint ──
@app.get("/")
def root():
    """
    Root endpoint for health checking.
    Returns a JSON message confirming that the backend server is running.
    """
    return {"message": "FastAPI is running"}


# ── Router Registration ──
# Include routes from external modules to keep endpoint code organized and modular
app.include_router(auth.router)
app.include_router(detect.router)