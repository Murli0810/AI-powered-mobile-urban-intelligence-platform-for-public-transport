"""
Database table definition for events.

This mirrors the exact JSON event schema we already agreed on with the
frontend (see PRD_Frontend_SIH2026.md, section 5 -- API Contract).
`extra_data` stores type-specific fields (vehicle_count, plate_number, etc.)
as a JSON string, since different event types carry different extra data
and SQLite doesn't need a rigid column for each possibility.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from database import Base
import datetime


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, index=True)       # pothole, vehicle_density, incident, etc.
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    gps_lat = Column(Float, index=True)
    gps_lon = Column(Float, index=True)
    bus_id = Column(String, index=True)
    snapshot_url = Column(String, nullable=True)
    extra_data = Column(Text, nullable=True)       # JSON string: {"plate_number": "...", ...}