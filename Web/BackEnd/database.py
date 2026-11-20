"""
Database Setup Module.

This file sets up our connection to the database using SQLAlchemy.
- SQLAlchemy: A Python tool (ORM) that lets us talk to databases using Python code instead of raw SQL queries.
- Engine: The actual connection manager between Python and the database.
- Session / SessionLocal: A temporary workspace for reading and writing data to/from the database.
- Base: A foundational class that all database models (tables) inherit from so SQLAlchemy knows about them.
"""

# ── Imports ──
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ── Environment Configuration ──
# Read the .env file and set environment variables in os.environ
load_dotenv()

# Retrieve the database connection string (URL) from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

# ── Database Connection & Setup ──
# Create the database engine which manages connection pools to the database
engine = create_engine(DATABASE_URL)

# Create a factory for database sessions (workspaces for db operations)
# autocommit=False: Changes aren't saved automatically; you must explicitly commit.
# autoflush=False: Prevents sending changes to the DB before explicit commits.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── Model Base Class ──
# Base class that all database model classes inherit from
Base = declarative_base()
