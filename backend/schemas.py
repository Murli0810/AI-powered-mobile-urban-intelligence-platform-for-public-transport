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
    extra: Optional[dict] = None

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
        from_attributes = True