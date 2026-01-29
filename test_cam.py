from camera import VideoCamera

cam = VideoCamera()
print("Camera initialized.")

# Simulate grabbing a frame
frame = cam.get_frame()
print(f"Frame captured. Size: {len(frame)} bytes")

# Check stats
print("Stats:", cam.get_stats())