"""
Authentication and User Management Router Module.

This file provides API endpoints for user creation, fetching user lists, and admin checks.

Key Concepts Explained for Beginners:
- APIRouter: Groups related route endpoints into a modular sub-router to keep main API code clean.
- Pydantic BaseModel: Defines and validates the shape and data types of incoming HTTP request JSON bodies.
- Password Hashing: Converts plain-text passwords into secure, unreadable cryptographic hashes using bcrypt.
- Dependency Injection (Depends): Design pattern where FastAPI automatically creates and manages dependencies 
  (such as database sessions) for route handlers during request processing.
"""

# ── Imports & Setup ──

from pydantic import BaseModel
from database import SessionLocal
from fastapi import APIRouter, Depends, HTTPException, status # type: ignore
from sqlalchemy.orm import Session # type: ignore
from typing import Annotated
from model import User
from passlib.context import CryptContext 

# Setup passlib CryptContext using bcrypt algorithm for password hashing and verification
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Create API router under the '/auth' prefix so all routes start with /auth
router = APIRouter(
    prefix='/auth',
    tags=["auth"]
)


# ── Data Models ──

class createUserRequest(BaseModel):
    """
    Request body schema for user registration.
    
    Pydantic's BaseModel validates that all required fields are present in the HTTP request.
    """
    username: str
    email: str
    role: str
    password: str
    licence_plate: str


# ── Database Connection & Dependency Injection ──

def get_db():
    """
    Database session generator function.
    
    Yields a database session for request execution and closes it when finished.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Dependency Injection: Instructs FastAPI to automatically inject an active DB session into route parameters
db_dependency = Annotated[Session, Depends(get_db)]


# ── Helper Functions ──

def authenticate_user(username: str, password: str, db: Session):
    """
    Verifies user identity by checking username existance and password hash match.
    
    Returns the User object on success, or False on failure.
    """
    # Query database for user with matching username
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    
    # Verify plain-text password against stored bcrypt password hash
    if not pwd_context.verify(password, user.hash_password):
        return False
    return user


# ── API Endpoints ──

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_user(create_user_request: createUserRequest, db: db_dependency):
    """
    Endpoint: POST /auth/
    
    Creates a new user record in the database with a hashed password.
    Returns HTTP 201 Created status upon completion.
    """
    # Create User model instance, converting raw password into secure hash
    create_user_model = User(
        email=create_user_request.email,
        username=create_user_request.username,
        hash_password=pwd_context.hash(create_user_request.password),
        licence_plate=create_user_request.licence_plate
    )
    
    # Stage and commit new user record into database
    db.add(create_user_model)
    db.commit()


# NOTE: This duplicate authenticate_user function is identical to the one above.
# In Python, defining a function twice silently overwrites the first definition.
# Kept intact to preserve original file code structure while noting the redundancy.
def authenticate_user(username: str, password: str, db: Session):
    """
    Verifies user identity by checking username existance and password hash match.
    
    Returns the User object on success, or False on failure.
    """
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not pwd_context.verify(password, user.hash_password):
        return False
    return user


@router.get("/all-users")
async def get_all_users(db: db_dependency):
    """
    Endpoint: GET /auth/all-users
    
    Fetches and returns all registered user records from the database.
    """
    users = db.query(User).all()
    return users


@router.get("/admin")
def admin_only(current_user = Depends(get_all_users)):
    """
    Endpoint: GET /auth/admin
    
    Restricted endpoint requiring admin privileges.
    Raises HTTP 403 Forbidden error if user role is not 'admin'.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not allowed")
    return {"message": "Welcome admin"}
