import csv
from datetime import datetime, timedelta


def load_gps_log_csv(path: str):
   
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
    
    frame_time = video_start_time + timedelta(seconds=frame_idx / fps)

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

    total_gap = (after["timestamp"] - before["timestamp"]).total_seconds()
    if total_gap == 0:
        return before["lat"], before["lon"]
    fraction = (frame_time - before["timestamp"]).total_seconds() / total_gap
    lat = before["lat"] + fraction * (after["lat"] - before["lat"])
    lon = before["lon"] + fraction * (after["lon"] - before["lon"])
    return lat, lon


def simulate_route(start_lat, start_lon, end_lat, end_lon, frame_idx, total_frames):

    fraction = frame_idx / max(total_frames - 1, 1)
    lat = start_lat + fraction * (end_lat - start_lat)
    lon = start_lon + fraction * (end_lon - start_lon)
    return lat, lon


if __name__ == "__main__":
    example_lat, example_lon = simulate_route(
        start_lat=22.8046, start_lon=86.2029,
        end_lat=22.8060, end_lon=86.2050,
        frame_idx=451, total_frames=902,
    )
    print(f"Simulated location at frame 451: {example_lat:.6f}, {example_lon:.6f}")