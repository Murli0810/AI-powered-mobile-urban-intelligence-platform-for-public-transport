from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from database import Base
import datetime


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, index=True)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    gps_lat = Column(Float, index=True)
    gps_lon = Column(Float, index=True)
    bus_id = Column(String, index=True)
    snapshot_url = Column(String, nullable=True)
    extra_data = Column(Text, nullable=True)