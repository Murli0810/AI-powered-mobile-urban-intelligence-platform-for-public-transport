"""
Central platform: check if a detected event (pothole, pedestrian alert, etc.)
is near a school, using OpenStreetMap's free Overpass API.

This runs on the BACKEND when an event is ingested, not on the edge device --
it's a one-time lookup against map data, not a video-processing task.

NOTE: requires internet access and the `requests` library.
pip install requests
"""

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def is_near_school(lat: float, lon: float, radius_m: int = 150) -> dict:
    """
    Queries OSM for school POIs within radius_m meters of (lat, lon).
    Returns {"near_school": bool, "schools": [...]}
    """
    query = f"""
    [out:json];
    (
      node["amenity"="school"](around:{radius_m},{lat},{lon});
      way["amenity"="school"](around:{radius_m},{lat},{lon});
    );
    out center;
    """
    response = requests.post(OVERPASS_URL, data={"data": query}, timeout=10)
    response.raise_for_status()
    data = response.json()

    schools = [
        {"name": el.get("tags", {}).get("name", "Unnamed school"),
         "lat": el.get("lat") or el.get("center", {}).get("lat"),
         "lon": el.get("lon") or el.get("center", {}).get("lon")}
        for el in data.get("elements", [])
    ]
    return {"near_school": len(schools) > 0, "schools": schools}


def evaluate_infrastructure_need(event_lat, event_lon, pedestrian_alert_count: int, radius_m: int = 150):
    """
    Combines school-proximity with alert frequency to generate a
    recommendation -- this is the kind of "actionable insight" the PS asks
    the central platform to produce.
    """
    result = is_near_school(event_lat, event_lon, radius_m)

    if result["near_school"] and pedestrian_alert_count >= 3:
        return {
            "recommendation": "Install speed breaker / school-zone signage",
            "reason": f"{pedestrian_alert_count} pedestrian alerts recorded near "
                      f"{result['schools'][0]['name']}",
            "priority": "high",
        }
    elif result["near_school"]:
        return {
            "recommendation": "Monitor — school zone, low alert frequency so far",
            "priority": "medium",
        }
    return {"recommendation": None, "priority": "low"}


if __name__ == "__main__":
    # Example: test against a known coordinate (replace with a real one near you)
    result = evaluate_infrastructure_need(
        event_lat=22.8046, event_lon=86.2029, pedestrian_alert_count=5
    )
    print(result)