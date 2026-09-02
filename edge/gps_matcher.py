"""
GPS matching: attach a real-world location to a detection event.

Two paths depending on what your teammate finds on the dashcam phone:

PATH A (preferred): a real GPS log exists (gpx/csv) alongside the video.
    -> use match_gps_to_frame() to look up coordinates by video timestamp.

PATH B (fallback, demo-only): no GPS log recorded.
    -> use simulate_route() to generate a plausible straight-line route
       between two known points (e.g. two points on your college campus),
       so the pipeline and dashboard still work end-to-end for the demo.
       NOTE: this is clearly a simulation and should be disclosed as such
       to judges if used -- it's a stand-in for real hardware, not a hack
       to hide.
"""

import csv
from datetime import datetime, timedelta


def load_gps_log_csv(path: str):
    """
    Expects a CSV with columns: timestamp, lat, lon
    timestamp format: ISO 8601, e.g. 2026-09-01T10:15:00
    Adjust column names/parsing if your app exports a different format.
    """
    log = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            log.append({
                "timestamp": datetime.fromisoformat(row["timestamp"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
            })
    return sorted(log, key=lambda x: x["timestamp"])


def match_gps_to_frame(gps_log, video_start_time: datetime, frame_idx: int, fps: float):
    """
    Given a loaded GPS log, the video's real-world start time, and a frame
    index, return the interpolated (lat, lon) for that moment.
    """
    frame_time = video_start_time + timedelta(seconds=frame_idx / fps)

    # Find the two GPS log entries surrounding frame_time
    before, after = None, None
    for entry in gps_log:
        if entry["timestamp"] <= frame_time:
            before = entry
        elif entry["timestamp"] > frame_time and after is None:
            after = entry
            break

    if before is None:
        return after["lat"], after["lon"]
    if after is None:
        return before["lat"], before["lon"]

    # Linear interpolation between the two surrounding points
    total_gap = (after["timestamp"] - before["timestamp"]).total_seconds()
    if total_gap == 0:
        return before["lat"], before["lon"]
    fraction = (frame_time - before["timestamp"]).total_seconds() / total_gap
    lat = before["lat"] + fraction * (after["lat"] - before["lat"])
    lon = before["lon"] + fraction * (after["lon"] - before["lon"])
    return lat, lon


def simulate_route(start_lat, start_lon, end_lat, end_lon, frame_idx, total_frames):
    """
    FALLBACK ONLY: linearly interpolate a straight-line route between two
    known points, based on how far through the video we are. Use this only
    if no real GPS log is available -- disclose as simulated data in the demo.
    """
    fraction = frame_idx / max(total_frames - 1, 1)
    lat = start_lat + fraction * (end_lat - start_lat)
    lon = start_lon + fraction * (end_lon - start_lon)
    return lat, lon


if __name__ == "__main__":
    # Example: simulate a route across a 902-frame video (sample1.mp4)
    # Replace with real campus coordinates for your demo
    example_lat, example_lon = simulate_route(
        start_lat=22.8046, start_lon=86.2029,
        end_lat=22.8060, end_lon=86.2050,
        frame_idx=451, total_frames=902,
    )
    print(f"Simulated location at frame 451: {example_lat:.6f}, {example_lon:.6f}")