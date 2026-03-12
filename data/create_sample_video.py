"""
Generate a synthetic sample traffic video for testing.

Creates a 10-second video with animated rectangles simulating vehicles
moving through an intersection with stop line and lane markings.
"""

import cv2
import numpy as np
import os

def create_sample_video(
    output_path: str = "sample_videos/traffic.mp4",
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration_sec: int = 10,
):
    """Generate a synthetic traffic video with moving vehicle rectangles."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"ERROR: Cannot create video writer for {output_path}")
        return

    total_frames = fps * duration_sec

    # Define simulated vehicles: (start_x, start_y, speed_x, speed_y, w, h, color)
    vehicles = [
        # Lane 1 — moving downward
        {"x": 350, "y": -80, "vx": 0, "vy": 4, "w": 70, "h": 50, "color": (200, 50, 50)},
        # Lane 2 — moving downward faster
        {"x": 670, "y": -200, "vx": 0, "vy": 6, "w": 80, "h": 55, "color": (50, 200, 50)},
        # Lane 1 — second car
        {"x": 320, "y": -400, "vx": 0, "vy": 5, "w": 65, "h": 48, "color": (50, 50, 200)},
        # Lane 2 — truck
        {"x": 640, "y": -600, "vx": 0, "vy": 3, "w": 90, "h": 70, "color": (200, 200, 50)},
        # Cross traffic — moving right
        {"x": -100, "y": 450, "vx": 5, "vy": 0, "w": 75, "h": 50, "color": (200, 100, 50)},
        # Cross traffic — moving left
        {"x": width + 50, "y": 500, "vx": -4, "vy": 0, "w": 70, "h": 48, "color": (100, 50, 200)},
    ]

    # Define road layout
    stop_line_y = 400
    lane1_x = (200, 500)
    lane2_x = (520, 820)

    for frame_idx in range(total_frames):
        # Create road background
        frame = np.full((height, width, 3), (80, 80, 80), dtype=np.uint8)

        # Road surface (darker)
        cv2.rectangle(frame, (150, 0), (870, height), (50, 50, 50), -1)

        # Lane markings (dashed center line)
        for y in range(0, height, 40):
            # Road edges
            cv2.line(frame, (150, y), (150, min(y + 20, height)), (255, 255, 255), 2)
            cv2.line(frame, (870, y), (870, min(y + 20, height)), (255, 255, 255), 2)
            # Center dashes
            cv2.line(frame, (510, y), (510, min(y + 20, height)), (0, 200, 200), 2)

        # Lane polygons (semi-transparent)
        overlay = frame.copy()
        cv2.rectangle(overlay, (lane1_x[0], 0), (lane1_x[1], height), (100, 50, 0), -1)
        cv2.rectangle(overlay, (lane2_x[0], 0), (lane2_x[1], height), (100, 50, 0), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Stop line
        cv2.line(frame, (100, stop_line_y), (900, stop_line_y), (0, 0, 255), 3)
        cv2.putText(frame, "STOP", (105, stop_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Traffic light indicator (red for first 5 sec, green for next 5)
        light_color = (0, 0, 255) if frame_idx < total_frames // 2 else (0, 255, 0)
        light_label = "RED" if frame_idx < total_frames // 2 else "GREEN"
        cv2.circle(frame, (950, 50), 25, light_color, -1)
        cv2.circle(frame, (950, 50), 25, (255, 255, 255), 2)
        cv2.putText(frame, light_label, (920, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_color, 2)

        # Draw and update vehicles
        for v in vehicles:
            x = int(v["x"] + v["vx"] * frame_idx)
            y = int(v["y"] + v["vy"] * frame_idx)

            # Wrap vehicles that go off-screen
            if v["vy"] > 0 and y > height + 100:
                v["y"] = -100 - v["vy"] * frame_idx + v["y"]
            if v["vy"] < 0 and y < -100:
                v["y"] = height + 100 - v["vy"] * frame_idx + v["y"]
            if v["vx"] > 0 and x > width + 100:
                v["x"] = -100 - v["vx"] * frame_idx + v["x"]
            if v["vx"] < 0 and x < -100:
                v["x"] = width + 100 - v["vx"] * frame_idx + v["x"]

            # Recalculate after potential wrap
            x = int(v["x"] + v["vx"] * frame_idx)
            y = int(v["y"] + v["vy"] * frame_idx)

            # Draw vehicle rectangle
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + v["w"])
            y2 = min(height, y + v["h"])
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), v["color"], -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # Frame info
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}",
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)

        writer.write(frame)

    writer.release()
    print(f"Sample video created: {output_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {duration_sec}s ({total_frames} frames)")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(script_dir, "sample_videos", "traffic.mp4")
    create_sample_video(output)
