"""
Database Models Module.

This file defines the structure of database tables using Python classes (Models).
- Models / Tables: Python classes representing database tables where data is stored in rows and columns.
- Column types (Integer, String, Boolean): Specify what kind of data each column can store (numbers, text, boolean flags).
- primary_key=True: The unique identifier for each row/record in the table.
- unique=True: Ensures no two rows can have duplicate values in this column (e.g., duplicate emails).
- index=True: Creates a search index for faster database lookups on this column.
- nullable=True: Allows this column to be empty (None / NULL).
"""

# ── Imports ──
from database import Base
from sqlalchemy import Column, Integer, String, Boolean


# ── Database Models ──

class User(Base):
    """
    Represents the 'users' table in the database.
    Stores user account credentials, roles, and vehicle details.
    """
    __tablename__ = 'users'

    # Primary key ID column (unique for each user, indexed for fast lookup)
    id = Column(Integer, primary_key=True, index=True)
    # Unique username for login, indexed for quick searches
    username = Column(String, unique=True, index=True)
    # Unique email address for the user, indexed for quick searches
    email = Column(String, unique=True, index=True)
    # User authorization role (defaults to "user")
    role = Column(String, default="user")
    # Hashed version of the user's password for security
    hash_password = Column(String)
    # Optional vehicle license plate number associated with the user
    licence_plate = Column(String, nullable=True)


class Violation(Base):
    """
    Represents the 'violations' table in the database.
    Stores records of detected traffic or security violations.
    """
    __tablename__ = 'violations'

    # Primary key ID for each violation record
    id = Column(Integer, primary_key=True, index=True)
    # Optional license plate detected during the violation
    plate = Column(String, nullable=True)
    # Filename of the captured violation image
    filename = Column(String)
    # Timestamp when the violation occurred
    timestamp = Column(String)