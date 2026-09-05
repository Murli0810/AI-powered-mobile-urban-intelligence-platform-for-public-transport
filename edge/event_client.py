import requests

BACKEND_URL = "http://127.0.0.1:8000"


def send_event(event_type: str, confidence: float, gps_lat: float, gps_lon: float,
                bus_id: str = "bus_01", snapshot_url: str = None, extra: dict = None):
    
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