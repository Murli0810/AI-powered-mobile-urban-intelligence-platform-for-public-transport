"""
Pydantic schemas: define the exact shape of data going IN to the API
(EventCreate) and OUT of it (EventOut). FastAPI uses these to validate
incoming requests automatically and to generate the interactive API docs.

Concept: this is intentionally separate from models.py (the DATABASE
table). Keeping "what the API accepts" separate from "how it's stored"
means we can change one without breaking the other -- e.g. if we later
want to accept an optional field the frontend hasn't sent yet.
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class EventCreate(BaseModel):
    event_type: str
    confidence: float
    gps_lat: float
    gps_lon: float
    bus_id: str
    snapshot_url: Optional[str] = None
    extra: Optional[dict] = None  # e.g. {"vehicle_count": 12} or {"plate_number": "JH01AB1234"}


class EventOut(BaseModel):
    id: int
    event_type: str
    confidence: float
    timestamp: datetime
    gps_lat: float
    gps_lon: float
    bus_id: str
    snapshot_url: Optional[str] = None
    extra: Optional[dict] = None

    class Config:
        from_attributes = True  # allows creating this from a SQLAlchemy object directly