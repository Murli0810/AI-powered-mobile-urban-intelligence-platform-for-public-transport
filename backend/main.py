"""
Main FastAPI application.

Run from the backend/ folder:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for interactive API testing --
FastAPI auto-generates this from the Pydantic schemas, no extra work needed.
"""

import json
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func

import models
import schemas
from database import engine, get_db

# Creates the events.db file and the events table, if they don't already exist.
# Safe to call every time the app starts -- does nothing if the table exists.
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Urban Intelligence Platform API")

# CORS: without this, a browser will BLOCK requests from your frontend
# (e.g. a Next.js app on Vercel) to this API, since they're on different
# domains. allow_origins=["*"] is fine for a hackathon demo -- for a real
# production app you'd restrict this to your actual frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def db_event_to_out(event: models.Event) -> schemas.EventOut:
    """Converts a database row into the API response shape, parsing the
    JSON string back into a real dict for the 'extra' field."""
    extra = json.loads(event.extra_data) if event.extra_data else None
    return schemas.EventOut(
        id=event.id,
        event_type=event.event_type,
        confidence=event.confidence,
        timestamp=event.timestamp,
        gps_lat=event.gps_lat,
        gps_lon=event.gps_lon,
        bus_id=event.bus_id,
        snapshot_url=event.snapshot_url,
        extra=extra,
    )


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Urban Intelligence Platform API"}


@app.post("/events", response_model=schemas.EventOut)
def create_event(event: schemas.EventCreate, db: Session = Depends(get_db)):
    """Ingests one event from an edge device (bus)."""
    db_event = models.Event(
        event_type=event.event_type,
        confidence=event.confidence,
        gps_lat=event.gps_lat,
        gps_lon=event.gps_lon,
        bus_id=event.bus_id,
        snapshot_url=event.snapshot_url,
        extra_data=json.dumps(event.extra) if event.extra else None,
    )
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event_to_out(db_event)


@app.get("/events", response_model=List[schemas.EventOut])
def list_events(
    event_type: Optional[str] = None,
    since: Optional[datetime] = None,
    db: Session = Depends(get_db),
):
    """Matches the frontend's GET /api/events?type=...&since=... contract."""
    query = db.query(models.Event)
    if event_type:
        query = query.filter(models.Event.event_type == event_type)
    if since:
        query = query.filter(models.Event.timestamp >= since)
    events = query.order_by(models.Event.timestamp.desc()).all()
    return [db_event_to_out(e) for e in events]


@app.get("/heatmap")
def get_heatmap(db: Session = Depends(get_db)):
    """
    Aggregates vehicle-density events by rounded coordinate into heatmap
    points. Rounding to 3 decimal places groups nearby points together
    (~111m grid cells) instead of plotting every single detection as a
    separate point -- this is what makes it a "heatmap" instead of a
    scatter plot of thousands of dots.
    """
    results = (
        db.query(
            func.round(models.Event.gps_lat, 3).label("lat"),
            func.round(models.Event.gps_lon, 3).label("lon"),
            func.count().label("count"),
        )
        .filter(models.Event.event_type == "vehicle_density")
        .group_by("lat", "lon")
        .all()
    )
    max_count = max((r.count for r in results), default=1)
    return [
        {"gps_lat": r.lat, "gps_lon": r.lon, "weight": round(r.count / max_count, 2)}
        for r in results
    ]


@app.get("/routes/delays")
def get_route_delays():
    """
    Placeholder for route delay analytics -- needs real route/segment
    definitions to compute meaningfully, which we haven't built yet.
    Returns an empty list for now so the frontend's chart component has
    something valid to render against without crashing.
    """
    return []