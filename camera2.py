import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os  # <-- ADDED: For creating folders
import datetime  # <-- ADDED: For unique filenames

def valid_depth(depth_frame, x, y, ksize=5):
    """
    Return a robust depth reading (in meters) around (x,y):
    - Try center pixel first
    - If 0 or invalid, take median over a ksize x ksize window (odd).
    """
    h, w = depth_frame.get_height(), depth_frame.get_width()
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    z = depth_frame.get_distance(x, y)
    if z > 0:
        return z

    # Fallback: median in a small neighborhood
    r = ksize // 2
    xs = np.clip(np.arange(x - r, x + r + 1), 0, w - 1)
    ys = np.clip(np.arange(y - r, y + r + 1), 0, h - 1)
    vals = []
    for yy in ys:
        for xx in xs:
            d = depth_frame.get_distance(int(xx), int(yy))
            if d > 0:
                vals.append(d)
    if len(vals):
        return float(np.median(vals))
    return 0.0

def find_largest_contour(mask, min_area=2000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None
    return cnt

def main():
    pipe = rs.pipeline()
    cfg  = rs.config()

    # Stream setup
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Align depth to color
    align_to = rs.stream.color
    align = rs.align(align_to)

    profile = pipe.start(cfg)

    # --- ADDED: Setup for Point Cloud saving ---
    pc = rs.pointcloud()
    points = rs.points()
    # -----------------------------------------

    # Depth colorizer for visualization
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 2)  # Jet

    prev_time = time.time()
    fps = 0.0

    print("Streaming... Press 's' to save a .ply file, 'q' to quit.")

    try:
        while True:
            frames = pipe.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())
            depth_viz   = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            # Threshold mask
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            cnt = find_largest_contour(mask, min_area=2000)
            overlay = color_image.copy()
            centroid_text = "Centroid: N/A"

            if cnt is not None:
                # Draw contour
                cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)

                # Centroid via moments
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    z_m = valid_depth(depth_frame, cx, cy, ksize=5)

                    # Draw centroid
                    cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
                    centroid_text = f"Centroid: x={cx}, y={cy}, Z={z_m:.3f} m"

            # Show centroid text
            cv2.putText(overlay, centroid_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # FPS calculation
            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = now
            cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Display
            cv2.imshow("RGB + Centroid", overlay)
            cv2.imshow("Depth (viz)", depth_viz)
            cv2.imshow("Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # --- ADDED: Save functionality on 's' key press ---
            elif key == ord('s'):
                print("Saving point cloud...")
                
                # Define save directory and create it
                save_dir = r"F:\RealSenseData\Captures"
                os.makedirs(save_dir, exist_ok=True)
                
                # Generate a unique filename with a timestamp
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{ts}.ply"
                save_path = os.path.join(save_dir, filename)

                # Generate and save the point cloud
                pc.map_to(color_frame)
                points = pc.calculate(depth_frame)
                points.export_to_ply(save_path, color_frame)
                
                print(f" Point cloud saved to: {save_path}")
            # ----------------------------------------------------

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()