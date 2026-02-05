import cv2
import numpy as np
import torch
import pyttsx3
import heapq
import matplotlib.pyplot as plt

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
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

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

# Visual odometry with ORB feature matching and obstacle detection
def process_visual_odometry(prev_frame, prev_keypoints, prev_descriptors, frame, K, trajectory, obstacles):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect and compute features
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # If we have a previous frame, try matching the features
    if prev_frame is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(prev_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) >= 5:
            prev_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            curr_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            E, mask = cv2.findEssentialMat(curr_pts, prev_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None and E.shape == (3, 3):
                _, R, t, _ = cv2.recoverPose(E, curr_pts, prev_pts, K)

                if len(trajectory) == 0:
                    trajectory.append((0, 0))  # Start at the origin
                else:
                    last_pos = np.array(trajectory[-1])
                    new_pos = last_pos + np.array([t[0, 0], t[2, 0]])
                    trajectory.append(tuple(new_pos))

                # Detect obstacles based on feature displacement
                obstacle_threshold = 20
                for i, m in enumerate(matches):
                    if mask[i] and np.linalg.norm(curr_pts[i] - prev_pts[i]) > obstacle_threshold:
                        obstacles.append(tuple(curr_pts[i][0]))

    return gray, keypoints, descriptors, trajectory, obstacles

# Main Function
def main():
    model = load_model()
    if not model:
        return

    video_path = "home_video.mp4"  # Path to your video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video file.")
        return

    print("Processing video...")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize previous frame variables for visual odometry
    prev_frame = None
    prev_keypoints = None
    prev_descriptors = None
    trajectory = []  # Store the 2D positions of the camera
    obstacles = []  # Store detected obstacles positions

    # Define initial grid size for pathfinding
    grid_size = 20
    grid_width = width // grid_size
    grid_height = height // grid_size

    # Start and end point for A* pathfinding
    start = (height // 2 // grid_size, width // 2 // grid_size)
    end = (height // grid_size - 1, width // 2 // grid_size)

    speak("Starting path navigation. Follow the instructions.")

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Visual Odometry Processing
        prev_frame, prev_keypoints, prev_descriptors, trajectory, obstacles = process_visual_odometry(
            prev_frame, prev_keypoints, prev_descriptors, frame, K=np.array([[525.0, 0.0, 319.5],
                                                                          [0.0, 525.0, 239.5],
                                                                          [0.0, 0.0, 1.0]]), trajectory=trajectory, obstacles=obstacles
        )

        # Object Detection and Pathfinding
        detected_objects = detect_objects(model, frame)
        grid = np.zeros((grid_height, grid_width), dtype=int)

        # Mark detected obstacles in grid
        for _, row in detected_objects.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            grid_xmin, grid_ymin = xmin // grid_size, ymin // grid_size
            grid_xmax, grid_ymax = xmax // grid_size, ymax // grid_size
            grid[grid_ymin:grid_ymax, grid_xmin:grid_xmax] = 1

        # Find path using A* algorithm
        path = a_star(grid, start, end)

        # Provide feedback if obstacles are detected
        if len(obstacles) > 0:
            speak("Warning! Obstacles detected ahead.")
        
        if path:
            next_point = path[0]
            speak(f"Move to position ({next_point[0]}, {next_point[1]})")
        else:
            speak("No clear path found, stopping navigation.")

        # Show the frame
        cv2.imshow("Object Detection and Path Guidance", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    # Visualize the trajectory and obstacles in 2D
    trajectory = np.array(trajectory)
    obstacles = np.array(obstacles)

    plt.figure(figsize=(10, 6))
    plt.plot(trajectory[:, 0], -trajectory[:, 1], marker='o', label="Camera Trajectory")
    if len(obstacles) > 0:
        plt.scatter(obstacles[:, 0], -obstacles[:, 1], color='red', label="Obstacles")
    plt.title("Top-Down View of Camera Trajectory and Obstacles")
    plt.xlabel("X (arbitrary units)")
    plt.ylabel("Y (arbitrary units)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
