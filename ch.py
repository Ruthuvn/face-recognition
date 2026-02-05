import cv2
import pyttsx3
import torch
import time
import heapq
import numpy as np

# Initialize Text-to-Speech Engine (one-time initialization)
engine = pyttsx3.init()

def speak(instruction, rate=150, volume=1.0):
    engine.setProperty('rate', rate)  # Set speed of speech
    engine.setProperty('volume', volume)  # Set volume level (0.0 to 1.0)
    engine.say(instruction)
    engine.runAndWait()

# Load YOLOv5 Model
def load_model():
    try:
        print("Loading YOLOv5 model...")
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Object Detection Function with Confidence Threshold
def detect_objects(model, frame, conf_threshold=0.5):
    results = model(frame)  # Detect objects
    detected_objects = results.pandas().xyxy[0]  # Pandas DataFrame of results
    filtered_objects = detected_objects[detected_objects['confidence'] > conf_threshold]
    return filtered_objects

# A* Algorithm for Pathfinding
def a_star(grid, start, end):
    # Heuristic function (Euclidean distance)
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    # A* algorithm implementation
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, end), 0, start))  # (f_score, g_score, position)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_list:
        _, _, current = heapq.heappop(open_list)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Return reversed path

        neighbors = [(current[0] + dx, current[1] + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        for neighbor in neighbors:
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1] and grid[neighbor] == 0:
                tentative_g_score = g_score.get(current, float('inf')) + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_list, (f_score[neighbor], tentative_g_score, neighbor))

    return []  # Return empty list if no path found

# Check if there are objects along the path
def check_path_for_objects(detected_objects, path, grid_size):
    objects_in_path = []
    for (y, x) in path:
        grid_x = x * grid_size
        grid_y = y * grid_size
        # Check if there are any detected objects within a radius around the path point
        for _, row in detected_objects.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            # Check if the object's bounding box intersects the path point
            if (xmin <= grid_x <= xmax) and (ymin <= grid_y <= ymax):
                objects_in_path.append(row["name"])
    return objects_in_path

# Main Function
def main():
    # Load YOLOv5 Model
    model = load_model()
    if not model:
        return

    # Open the Video File
    video_path = "home_video.mp4"  # Ensure the video file path is correct
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video file.")
        return

    print("Processing video...")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the grid size for pathfinding (downsample the image for performance)
    grid_size = 20  # Grid size (each cell represents 20x20 pixels)
    grid_width = width // grid_size
    grid_height = height // grid_size

    # Initialize previous frame for smoothness
    previous_frame = None

    # Define the start and end point for pathfinding
    start = (height // 2 // grid_size, width // 2 // grid_size)  # Start point (center of the frame)
    end = (height // grid_size - 1, width // 2 // grid_size)  # End point (center of the bottom)

    # Pathfinding and navigation logic
    path = None
    frame_interval = 5  # Process every 5 frames for better performance
    frames_processed = 0  # Counter to keep track of frames processed
    speak("Starting path navigation. Follow the instructions.")
    
    # To track objects we've already reported
    reported_objects = set()

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frames_processed += 1

        # Process frames periodically (e.g., every 5th frame)
        if frames_processed % frame_interval == 0:
            # Object Detection
            detected_objects = detect_objects(model, frame)

            # Create grid (0 for free space, 1 for obstacles)
            grid = np.zeros((grid_height, grid_width), dtype=int)

            # Mark detected objects as obstacles in the grid
            for _, row in detected_objects.iterrows():
                xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                grid_xmin, grid_ymin = xmin // grid_size, ymin // grid_size
                grid_xmax, grid_ymax = xmax // grid_size, ymax // grid_size
                grid[grid_ymin:grid_ymax, grid_xmin:grid_xmax] = 1  # Mark as obstacle

            # Find path using A* algorithm
            path = a_star(grid, start, end)

            # Check if there are objects along the path
            objects_in_path = check_path_for_objects(detected_objects, path, grid_size)

            # Provide feedback on obstacles
            if objects_in_path:
                for obj in objects_in_path:
                    if obj not in reported_objects:
                        speak(f"Warning! There is a {obj} in front of you.")
                        reported_objects.add(obj)

            # Determine left or right turn (based on objects position)
            left_x = width // 3
            right_x = 2 * width // 3

            for _, row in detected_objects.iterrows():
                object_center_x = (row['xmin'] + row['xmax']) // 2
                if object_center_x < left_x:
                    speak("Turn left. There is an object on the left.")
                    break
                elif object_center_x > right_x:
                    speak("Turn right. There is an object on the right.")
                    break

            # If no objects in the way, guide forward
            if not objects_in_path:
                speak("The path is clear. Move forward.")

            # Update the start position for next pathfinding step
            if path:
                next_point = path[0]
                start = next_point
                speak(f"Move to position ({next_point[0]}, {next_point[1]})")
            else:
                speak("No path found, stopping navigation.")

        # Show the video frame (you can comment out this line if you don't need visualization)
        cv2.imshow("Object Detection and Path Guidance", frame)

        # Exit on 'q' Key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    main()
