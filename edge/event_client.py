"""
Shared utility: sends a detection event from any edge script to the
central backend, over HTTP -- this is the actual "bandwidth-conscious
edge processing" the PS asks for: we send one small JSON object per
detection, never the video itself.

Import this into any edge script:
    from event_client import send_event

Usage:
    send_event(
        event_type="pothole",
        confidence=0.87,
        gps_lat=22.8046, gps_lon=86.2029,
        bus_id="bus_01",
    )
"""

import requests

BACKEND_URL = "http://127.0.0.1:8000"  # change to teammate's IP if testing across machines


def send_event(event_type: str, confidence: float, gps_lat: float, gps_lon: float,
                bus_id: str = "bus_01", snapshot_url: str = None, extra: dict = None):
    """
    Sends one event to the backend. Fails silently (prints a warning)
    rather than crashing the whole edge pipeline if the backend is
    temporarily unreachable -- a dropped alert shouldn't stop the bus's
    video processing.
    """
    payload = {
        "event_type": event_type,
        "confidence": confidence,
        "gps_lat": gps_lat,
        "gps_lon": gps_lon,
        "bus_id": bus_id,
        "snapshot_url": snapshot_url,
        "extra": extra,
    }
    try:
        response = requests.post(f"{BACKEND_URL}/events", json=payload, timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[event_client] Warning: could not send event to backend ({e})")
        return None