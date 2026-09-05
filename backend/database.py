"""
Database connection setup.

SQLite for MVP speed -- zero setup, file-based, no server to install or run.
The whole database is one file: events.db, created automatically on first run.

Concept: SQLAlchemy is the "ORM" layer -- it lets us define database tables
as Python classes (see models.py) instead of writing raw SQL for every
query. `engine` is the actual connection to the database file. `SessionLocal`
creates a new "conversation" with the database each time we need one
(FastAPI will create one per incoming request).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./events.db"

# check_same_thread=False is needed specifically for SQLite + FastAPI,
# since FastAPI may handle a request on a different thread than the one
# that created the connection. Not needed for Postgres/MySQL.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """
    FastAPI dependency: provides a database session to each request,
    and guarantees it's closed afterward even if the request fails.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()