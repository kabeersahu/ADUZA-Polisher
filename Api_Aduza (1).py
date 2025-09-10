from aduza import *
import csv
import time

# =======================================================
#               CONFIGURATION
# =======================================================

# Path to your CSV file
CSV_FILE_PATH = r"C:\Users\hp\Downloads\polishing_trajectory.csv"
# --- SPEED CONTROL ---
# Time in seconds to complete the move between each waypoint.
# Decrease for a faster trajectory, increase for a slower one.
TIME_PER_SEGMENT = 0.005 # seconds

# --- SMOOTHNESS CONTROL ---
# Number of intermediate steps for the robot's internal calculation.
# Higher values can result in smoother motion.
STEPS_PER_SEGMENT = 1

# =======================================================
#               PROCEDURE AND CODE
# =======================================================

robot = None # Define robot outside try block for access in finally
try:
    # --- Step 1: Initialize the Robot ---
    print("Initializing Robot...")
    robot = Robot(port='COM4')
    robot.reset()
    wait(7) # Give the robot time to complete the reset

    # --- Step 2: Read Waypoints from CSV File ---
    print(f"Reading trajectory from {CSV_FILE_PATH}...")
    waypoints = []
    with open(CSV_FILE_PATH, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader) # Skip the header row
        
        # Read each row and convert string values to floats
        for row in reader:
            waypoint = [float(item) for item in row]
            waypoints.append(waypoint)
    
    if not waypoints:
        print("Warning: No waypoints found in the CSV file.")
    else:
        # Move to the starting point of the trajectory first
        print("Moving to the trajectory's starting position...")
        start_point = waypoints[0]
        robot.goto(
            x=start_point[0], y=start_point[1], z=start_point[2],
            roll=start_point[3], pitch=start_point[4], yaw=start_point[5]
        )
        wait(5) # Wait to ensure it reaches the start

    # --- Step 3: Loop Through Waypoints and Command Robot ---
    print("Starting trajectory following...")
    print(f"Time per segment: {TIME_PER_SEGMENT}s, Steps: {STEPS_PER_SEGMENT}")
    
    for i, point in enumerate(waypoints):
        # Unpack the list into named variables
        target_x, target_y, target_z, target_roll, target_pitch, target_yaw = point
        
        print(f"Moving to waypoint {i+1}/{len(waypoints)}: (x={target_x}, y={target_y}, z={target_z})")
        
        # Command the robot to move to the next point with controlled speed and smoothness
        robot.move_absolute(
            x=target_x, 
            y=target_y, 
            z=target_z, 
            roll=target_roll, 
            pitch=target_pitch, 
            yaw=target_yaw,
            T=TIME_PER_SEGMENT,
            steps=STEPS_PER_SEGMENT
        )
        # Note: We do not need an extra wait() here because the `move_absolute`
        # function should take `T` seconds to execute.
    
    print("Trajectory complete.")
    wait(1)
    
    # Optionally, return to home position
    print("Returning to home position...")
    robot.reset()


except FileNotFoundError:
    print(f"ERROR: The file '{CSV_FILE_PATH}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # --- Step 4: Shutdown ---
    # This block will run even if an error occurs
    if robot:
        print("Closing connection.")
        robot.close()