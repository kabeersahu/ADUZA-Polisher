import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
import os # For file path handling
import imageio # For saving animation frames to video (requires pip install imageio[ffmpeg])
import time # For timing operations
from scipy.interpolate import splrep, splev # For trajectory smoothing
from tqdm import tqdm # For progress bar

# --- GLOBAL CONFIGURATION PARAMETERS ---
# Adjust these parameters to suit your specific data and requirements
# --- Stage 1: PCD to Mesh Reconstruction Parameters ---
PCD_FILE_PATH = r"F:\RealSenseData\yoyo\object_top_patch_mesh.ply"
PCD_SCALE_FACTOR = 1000.0
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 7.0
NORMAL_EST_RADIUS = 5.0
NORMAL_EST_MAX_NN = 30
POISSON_RECON_DEPTH = 9
OUTPUT_MESH_FILENAME = 'reconstructed_poisson_mesh.ply'

# --- Stage 2: TIF Characterization Parameters ---
MEASUREMENT_DATA_FILE = 'tif_measurement_data.npy'
MRF_TOOL_DIAMETER_MM = 8.0
STATIC_DWELL_TIME_SEC = 10.0
DEFAULT_VOLUMETRIC_REMOVAL_RATE_MM3_PER_SEC = 1.0e-6
SIM_GRID_SIZE = 100
SIM_EXTENT_MM = MRF_TOOL_DIAMETER_MM * 1.5
SIM_DEPTH_UM = 10.0
SIM_NOISE_LEVEL = 0.02
SIM_SUPER_GAUSSIAN_POWER = 4

# --- Stage 3: Trajectory Generation Parameters ---
DESIRED_OVERLAP_PERCENTAGE = 0.65
LINEAR_SPEED_MM_PER_SEC = 0.1
PATH_POINT_SPACING_MM = 1
INSET_FROM_EDGE_MM = 4.0
TRAJECTORY_SMOOTHING_FACTOR = 0.05
TRAJECTORY_SMOOTHING_POINTS_PER_SEGMENT = 8
OUTPUT_TRAJECTORY_FILENAME = r'C:\Users\hp\Downloads\polishing_trajectory.csv' # Output file for XYZ + RPY

# --- Stage 4: Stewart Platform Kinematics and Animation Parameters ---
# Define platform joints in local mobile frame (MPF) and base joints in world frame (SBF)
Pi_platform_cpu = np.array([
    [25, 164.54, 0], [-25, 164.54, 0], [-155, -60.62, 0],
    [-130, -103.92, 0], [130, -103.92, 0], [155, -60.62, 0],
], dtype=np.float64)
Bi_base_cpu = np.array([
    [42.5, 134.23, 107], [-42.5, 134.23, 107], [-137.5, -30.31, 107],
    [-95, -103.92, 107], [95, -103.92, 107], [137.5, -30.31, 107],
], dtype=np.float64)
tool_position_in_mobile_cpu = np.array([0, 0, 175])
camera_position_in_mobile_cpu = np.array([50, 50, 0])                 
R_cam_to_mobile_cpu = np.eye(3)
T_cam_to_mobile_cpu = np.eye(4)
T_cam_to_mobile_cpu[:3, :3] = R_cam_to_mobile_cpu
T_cam_to_mobile_cpu[:3, 3] = camera_position_in_mobile_cpu

TOOL_CYLINDER_DIAMETER = 8
TOOL_CYLINDER_LENGTH = 175
ANIMATION_FPS = 60
ANIMATION_OUTPUT_FILENAME = r'C:\Users\hp\Downloads\mrf_polishing_animation.mp4'
ANIMATION_DOWNSAMPLE_FACTOR = 50 # This controls animation frames, not total calculation

MIN_LEG_LENGTH_MM = 10.0
MAX_LEG_LENGTH_MM = 300.0

# --- NEW: SAFETY WORKSPACE AND ORIENTATION LIMITS ---
# Define the safe operational envelope for the MOBILE PLATFORM's ORIGIN.
# Any point in the trajectory that requires the platform to exceed these
# limits will be discarded.
X_LIMITS_MM = [-70.0, 70.0]  # [min_x, max_x]
Y_LIMITS_MM = [-70.0, 70.0]  # [min_y, max_y]
Z_LIMITS_MM = [200.0, 340.0]   # [min_z, max_z]

# Define the safe orientation limits for the MOBILE PLATFORM in degrees.
ROLL_LIMITS_DEG = [-23.0, 23.0]   # [min_roll, max_roll]
PITCH_LIMITS_DEG = [-23.0, 23.0]  # [min_pitch, max_pitch]
YAW_LIMITS_DEG = [-27.0, 27.0]    # [min_yaw, max_yaw]

# --- Unified NumPy/CuPy import for GPU acceleration (only for TIF fitting now) ---
_use_cupy = False # Flag to track if CuPy is successfully imported and used
try:
    import cupy as cp
    print("Attempting to use CuPy for GPU acceleration...")
    _np_lib_tif = cp # Alias specifically for TIF fitting
    _use_cupy = True # Set flag to true if import is successful
    print("CuPy successfully imported and initialized. TIF fitting may use GPU.")

except ImportError as e:
    print(f"CuPy NOT FOUND or FAILED TO IMPORT: {e}. Falling back to NumPy (CPU).")
    print("Please ensure CuPy is installed (pip install cupy-cuda12x for CUDA 12.x) and your CUDA Toolkit is correctly set up for your GPU.")
    _np_lib_tif = np # Fallback to standard numpy
    
# Ensure global constants are standard NumPy arrays for core kinematics (reverted from CuPy objects)
# These will always be NumPy arrays in this version.
Pi_platform = Pi_platform_cpu
Bi_base = Bi_base_cpu
tool_position_in_mobile = tool_position_in_mobile_cpu
R_cam_to_mobile = R_cam_to_mobile_cpu
T_cam_to_mobile = T_cam_to_mobile_cpu
camera_position_in_mobile = camera_position_in_mobile_cpu


print("--- Starting Integrated MRF Polishing Pipeline ---")
print("-------------------------------------------------")

# --- Stage 1: PCD to Mesh Reconstruction ---
print("\n### Stage 1: Load Existing Poisson Mesh ###")

MESH_FILE_PATH = r"F:\RealSenseData\yoyo\object_surface_mesh.ply"

if not os.path.exists(MESH_FILE_PATH):
    print(f"Error: Mesh file '{MESH_FILE_PATH}' not found. Exiting pipeline.")
    exit()

poisson_mesh = o3d.io.read_triangle_mesh(MESH_FILE_PATH)
if not poisson_mesh.has_vertex_normals():
    poisson_mesh.compute_vertex_normals()

poisson_mesh.scale(1000.0, center=[0, 0, 0])

import numpy as np
flip_z = np.eye(3)
flip_z[2, 2] = -1
poisson_mesh.rotate(flip_z, center=[0, 0, 0])

print(f"Loaded mesh with {len(poisson_mesh.vertices)} vertices and {len(poisson_mesh.triangles)} faces")


# --- Separate Discrete Mesh Visualization (NEW) ---
print("\n--- Discrete Visualization of the Reconstructed Mesh (Open3D) ---")

# --- NEW: Create a coordinate frame to visualize the axes ---
# You can adjust the 'size' to make the axes bigger or smaller
coordinate_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])
if poisson_mesh and poisson_mesh.has_vertices():
    print("Opening Open3D viewer for the reconstructed mesh. Close window to continue.")
    print(f"Mesh min bound (for trajectory generation check): {poisson_mesh.get_min_bound()}")
    print(f"Mesh max bound (for trajectory generation check): {poisson_mesh.get_max_bound()}")
    o3d.visualization.draw_geometries(
    [poisson_mesh, coordinate_axes],
    window_name="Reconstructed Mesh (Discrete View)",
    width=800, height=600,  # <--- add these
    zoom=0.8,
    front=[0.4257, -0.2125, -0.8795],
    lookat=[0.05, 0.05, 0.05],
    up=[-0.0618, -0.9795, 0.1812]
)
else:
    print("No mesh to display for discrete visualization.")
print("Discrete mesh visualization complete (window closed or no mesh to display).")
# --- End Discrete Mesh Visualization ---

# --- Stage 2: TIF Characterization ---
print("\n### Stage 2: TIF Characterization ###")

def rpy_to_rot(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx

def build_transform(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = np.asarray(t)
    return T

def transform_point_and_normal(T, point_cam, normal_cam):
    p_cam_h = np.array([*point_cam, 1.0])
    p_world = (T @ p_cam_h)[:3]
    n_world = T[:3,:3] @ np.array(normal_cam)
    n_world /= (np.linalg.norm(n_world) + 1e-9)
    return p_world, n_world

def generate_synthetic_super_gaussian_tif_data(grid_size, extent, depth_um, noise_level, power):
    x = np.linspace(-extent / 2, extent / 2, grid_size)
    y = np.linspace(-extent / 2, extent / 2, grid_size)
    X, Y = np.meshgrid(x, y)
    sigma = MRF_TOOL_DIAMETER_MM / 4.0
    super_gaussian_profile = -depth_um * np.exp(-((X**2 + Y**2) / (2 * sigma**2))**(power/2))
    noise = np.random.normal(0, depth_um * noise_level, X.shape)
    Z = super_gaussian_profile + noise
    Z_offset = Z - np.max(Z)
    return X, Y, Z_offset

def super_gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, power_x, power_y, theta, offset):
    # This function uses the _np_lib_tif alias for CuPy if available, else np
    _local_np_lib = cp if _use_cupy else np 
    
    x_dev = _local_np_lib.array(xy[0]) 
    y_dev = _local_np_lib.array(xy[1])
    
    amplitude_dev = _local_np_lib.array(amplitude)
    xo_dev = _local_np_lib.array(xo)
    yo_dev = _local_np_lib.array(yo)
    theta_dev = _local_np_lib.array(theta)
    offset_dev = _local_np_lib.array(offset)
    
    sigma_x_dev = _local_np_lib.array(sigma_x)
    sigma_y_dev = _local_np_lib.array(sigma_y)
    power_x_dev = _local_np_lib.array(power_x)
    power_y_dev = _local_np_lib.array(power_y)

    x_rot = (x_dev - xo_dev) * _local_np_lib.cos(theta_dev) - (y_dev - yo_dev) * _local_np_lib.sin(theta_dev)
    y_rot = (x_dev - xo_dev) * _local_np_lib.sin(theta_dev) + (y_dev - yo_dev) * _local_np_lib.cos(theta_dev)
    g = offset_dev + amplitude_dev * _local_np_lib.exp( - ((_local_np_lib.abs(x_rot) / sigma_x_dev)**power_x_dev + (_local_np_lib.abs(y_rot) / sigma_y_dev)**power_y_dev))
    
    # Only call .get() if _use_cupy is True (i.e., if _local_np_lib is cp)
    return g.get().ravel() if _use_cupy else g.ravel()

print(f"2.1 Loading TIF measurement data from '{MEASUREMENT_DATA_FILE}' or generating synthetic data...")
try:
    loaded_data = np.load(MEASUREMENT_DATA_FILE) 
    X_meas = loaded_data['X']
    Y_meas = loaded_data['Y']
    Z_meas_raw = loaded_data['Z']
    print(f"Data loaded from '{MEASUREMENT_DATA_FILE}'. Z_meas_raw shape: {Z_meas_raw.shape}")
except FileNotFoundError:
    print(f"Measurement data file '{MEASUREMENT_DATA_FILE}' not found. Generating synthetic TIF data.")
    X_meas, Y_meas, Z_meas_raw = generate_synthetic_super_gaussian_tif_data(
        grid_size=SIM_GRID_SIZE, extent=SIM_EXTENT_MM, depth_um=SIM_DEPTH_UM,
        noise_level=SIM_NOISE_LEVEL, power=SIM_SUPER_GAUSSIAN_POWER
    )
    np.savez(MEASUREMENT_DATA_FILE, X=X_meas, Y=Y_meas, Z=Z_meas_raw)
    print(f"Synthetic data generated and saved to {MEASUREMENT_DATA_FILE}.")
except Exception as e:
    print(f"Error loading measurement data: {e}")
    print("Please check the file format and structure. Exiting pipeline.")
    exit()

print("2.2 Processing Measurement Data (Centering, Leveling)...")
Z_leveled = Z_meas_raw - np.max(Z_meas_raw)
min_z_idx = np.unravel_index(np.argmin(Z_leveled), Z_leveled.shape)
center_x_val = X_meas[min_z_idx]
center_y_val = Y_meas[min_z_idx]
X_centered = X_meas - center_x_val
Y_centered = Y_meas - center_y_val
Z_processed = Z_leveled
print(f"Data leveled (max Z set to 0) and centered around min removal point.")

print("2.3 Modeling the TIF by fitting a 2D Super-Gaussian...")
x_flat = X_centered.ravel()
y_flat = Y_centered.ravel()
z_flat = Z_processed.ravel()

initial_amplitude = np.min(z_flat)
initial_xo = 0.0
initial_yo = 0.0
initial_sigma_x = MRF_TOOL_DIAMETER_MM / 3.0
initial_sigma_y = MRF_TOOL_DIAMETER_MM / 3.0
initial_power_x = 4.0
initial_power_y = 4.0
initial_theta = 0.0
initial_offset = 0.0

p0 = [initial_amplitude, initial_xo, initial_yo,
      initial_sigma_x, initial_sigma_y, initial_power_x, initial_power_y,
      initial_theta, initial_offset]
bounds = ([-np.inf, -np.inf, -np.inf, 0.01, 0.01, 2.0, 2.0, -np.pi/2, -np.inf],
          [0, np.inf, np.inf, np.inf, np.inf, 10.0, 10.0, np.pi/2, np.inf])

TIF_EFFECTIVE_DIAMETER_MM_CALCULATED = MRF_TOOL_DIAMETER_MM
volumetric_removal_rate_mm3_per_sec = DEFAULT_VOLUMETRIC_REMOVAL_RATE_MM3_PER_SEC
fit_successful = False

try:
    popt, pcov = curve_fit(super_gaussian_2d, (x_flat, y_flat), z_flat, p0=p0, bounds=bounds, maxfev=50000)
    print("Super-Gaussian fit successful.")
    
    amplitude_fit, xo_fit, yo_fit, sigma_x_fit, sigma_y_fit, power_x_fit, power_y_fit, theta_fit, offset_fit = popt
    
    Z_fitted_tif = super_gaussian_2d((X_centered, Y_centered), *popt).reshape(X_centered.shape)
    
    peak_removal_rate_mm_per_sec = abs(amplitude_fit) / STATIC_DWELL_TIME_SEC / 1000
    dx = (X_centered[0, 1] - X_centered[0, 0])
    dy = (Y_centered[1, 0] - Y_centered[0, 0])
    volume_removed_um_mm2 = np.sum(np.abs(Z_processed)) * dx * dy
    volume_removed_mm3 = volume_removed_um_mm2 / 1000
    volumetric_removal_rate_mm3_per_sec = volume_removed_mm3 / STATIC_DWELL_TIME_SEC

    TIF_EFFECTIVE_DIAMETER_MM_CALCULATED = 4 * max(sigma_x_fit, sigma_y_fit)
    
    fit_successful = True

except RuntimeError as e:
    print(f"Error: Super-Gaussian fit failed. {e}")
    print("Using default TIF parameters for trajectory generation. Consider reviewing TIF data/fit.")

if fit_successful:
    print("\n--- TIF Characterization Results ---")
    print(f"  Fitted Peak Removal Depth: {amplitude_fit:.4f} µm")
    print(f"  Peak Removal Rate: {peak_removal_rate_mm_per_sec:.6f} mm/sec")
    print(f"  Fitted Sigma_x: {sigma_x_fit:.4f} mm")
    print(f"  Fitted Sigma_y: {sigma_y_fit:.4f} mm")
    print(f"  Fitted Power_x: {power_x_fit:.2f}")
    print(f"  Fitted Power_y: {power_y_fit:.2f}")
    print(f"  Volumetric Removal Rate: {volumetric_removal_rate_mm3_per_sec:.8f} mm^3/sec")
    print(f"  **Calculated TIF Effective Diameter for Trajectory: {TIF_EFFECTIVE_DIAMETER_MM_CALCULATED:.4f} mm**")
else:
    print("\n--- TIF Characterization Results (Default Parameters Used) ---")
    print(f"  Volumetric Removal Rate: {volumetric_removal_rate_mm3_per_sec:.8f} mm^3/sec")
    print(f"  **Calculated TIF Effective Diameter for Trajectory: {TIF_EFFECTIVE_DIAMETER_MM_CALCULATED:.4f} mm**")


# --- Stage 3: Trajectory Generation (in Surface World Frame) ---
print("\n### Stage 3: Trajectory Generation (in Surface World Frame - SWF) ###")

TIF_EFFECTIVE_DIAMETER_MM_STAGE3 = TIF_EFFECTIVE_DIAMETER_MM_CALCULATED

print("3.1 Configured Trajectory Parameters:")
print(f"  TIF Effective Diameter: {TIF_EFFECTIVE_DIAMETER_MM_STAGE3:.4f} mm (from TIF fit)")
print(f"  Desired Overlap: {DESIRED_OVERLAP_PERCENTAGE * 100}%")
print(f"  Linear Speed: {LINEAR_SPEED_MM_PER_SEC} mm/sec")
PATH_POINT_SPACING_MM = 1 # Reverted for denser trajectory as it's the original intention
INSET_FROM_EDGE_MM = 4.0

TRAJECTORY_SMOOTHING_FACTOR = 0.05
TRAJECTORY_SMOOTHING_POINTS_PER_SEGMENT = 8 # Reverted for denser trajectory as it's the original intention

OUTPUT_TRAJECTORY_FILENAME = r'C:\Users\hp\Downloads\polishing_trajectory.csv'

def smooth_trajectory_spline(raw_trajectory_points_np, poisson_mesh, kdtree_mesh, mesh_vertices, mesh_normals):
    """
    Smooths a 3D trajectory using B-spline interpolation for position (X,Y,Z)
    and re-queries normals from the mesh for the smoothed path.
    """
    _temp_np = np.array(raw_trajectory_points_np)
    if _temp_np.shape[0] < 4:
        print("Warning: Not enough points for spline smoothing. Skipping smoothing.")
        return _temp_np

    x_coords = _temp_np[:, 0]
    y_coords = _temp_np[:, 1]
    z_coords = _temp_np[:, 2]

    t_vals = np.cumsum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2 + np.diff(z_coords)**2))
    t_vals = np.insert(t_vals, 0, 0)
    
    if t_vals[-1] > 0:
        t_vals = t_vals / t_vals[-1]
    else:
        t_vals = np.linspace(0, 1, len(x_coords))

    total_original_segments = len(raw_trajectory_points_np) - 1
    if total_original_segments < 1:
        return _temp_np
    
    num_interp_points = total_original_segments * TRAJECTORY_SMOOTHING_POINTS_PER_SEGMENT
    if num_interp_points == 0:
        return _temp_np

    t_interp = np.linspace(0, 1, num_interp_points)

    s_x = TRAJECTORY_SMOOTHING_FACTOR * len(x_coords)
    s_y = TRAJECTORY_SMOOTHING_FACTOR * len(y_coords)
    s_z = TRAJECTORY_SMOOTHING_FACTOR * len(z_coords)

    try:
        tck_x = splrep(t_vals, x_coords, k=3, s=s_x)
        tck_y = splrep(t_vals, y_coords, k=3, s=s_y)
        tck_z = splrep(t_vals, z_coords, k=3, s=s_z)
    except Exception as e:
        print(f"Warning: Spline fitting failed ({e}). This might happen if points are too close or collinear for cubic spline. Returning unsmoothed trajectory.")
        return _temp_np

    smoothed_x = splev(t_interp, tck_x)
    smoothed_y = splev(t_interp, tck_y)
    smoothed_z = splev(t_interp, tck_z)

    smoothed_trajectory_xyz = np.vstack((smoothed_x, smoothed_y, smoothed_z)).T
    
    smoothed_trajectory_normals = np.zeros_like(smoothed_trajectory_xyz)
    for i, point in enumerate(smoothed_trajectory_xyz):
        [k, idx, dist] = kdtree_mesh.search_knn_vector_3d(point, 1) 
        if idx and len(idx) > 0:
            normal_on_surface = mesh_normals[idx[0]] 
            if normal_on_surface[2] < 0:
                normal_on_surface = -normal_on_surface
            normal_on_surface /= np.linalg.norm(normal_on_surface)
            smoothed_trajectory_normals[i] = normal_on_surface
        else:
            smoothed_trajectory_normals[i] = np.array([0, 0, 1])

    smoothed_trajectory_np = np.hstack((smoothed_trajectory_xyz, smoothed_trajectory_normals))

    return smoothed_trajectory_np


print("3.2 Generating polishing path on the reconstructed surface...")

if poisson_mesh is None or not poisson_mesh.has_vertices():
    print("Error: No valid mesh found for trajectory generation. Exiting Stage 3.")
    generated_trajectory_points_np = np.array([])
else:
    min_bound = poisson_mesh.get_min_bound()
    max_bound = poisson_mesh.get_max_bound()

    start_x = min_bound[0] + INSET_FROM_EDGE_MM
    end_x = max_bound[0] - INSET_FROM_EDGE_MM
    start_y = min_bound[1] + INSET_FROM_EDGE_MM
    end_y = max_bound[1] - INSET_FROM_EDGE_MM

    print(f"  Calculated path X range: [{start_x:.2f}, {end_x:.2f}] mm")
    print(f"  Calculated path Y range: [{start_y:.2f}, {end_y:.2f}] mm")

    if start_x >= end_x or start_y >= end_y:
        print(f"Warning: Inset {INSET_FROM_EDGE_MM}mm is too large for the mesh dimensions. No valid polishing area remains after inset.")
        print(f"         Mesh extent X: {max_bound[0] - min_bound[0]:.2f} mm")
        print(f"         Mesh extent Y: {max_bound[1] - min_bound[1]:.2f} mm")
        print(f"         Consider reducing INSET_FROM_EDGE_MM or using a larger PCD_SCALE_FACTOR.")
        generated_trajectory_points_np = np.array([])
    else:
        line_step_y = TIF_EFFECTIVE_DIAMETER_MM_STAGE3 * (1 - DESIRED_OVERLAP_PERCENTAGE)

        raw_zig_zag_points = []

        mesh_vertices = np.asarray(poisson_mesh.vertices)
        mesh_normals = np.asarray(poisson_mesh.vertex_normals)
        mesh_pcd_for_kdtree = o3d.geometry.PointCloud()
        mesh_pcd_for_kdtree.points = o3d.utility.Vector3dVector(mesh_vertices)
        mesh_pcd_for_kdtree.normals = o3d.utility.Vector3dVector(mesh_normals)
        kdtree_mesh = o3d.geometry.KDTreeFlann(mesh_pcd_for_kdtree)

        y_current = start_y
        while y_current <= end_y:
            if int(round((y_current - start_y) / line_step_y)) % 2 == 0:
                x_points_on_line = np.arange(start_x, end_x + PATH_POINT_SPACING_MM / 2, PATH_POINT_SPACING_MM)
            else:
                x_points_on_line = np.arange(end_x, start_x - PATH_POINT_SPACING_MM / 2, -PATH_POINT_SPACING_MM)

            for x_current in x_points_on_line:
                query_point = np.array([x_current, y_current, poisson_mesh.get_center()[2]])

                [k, idx, dist] = kdtree_mesh.search_knn_vector_3d(query_point, 1)

                if idx and len(idx) > 0:
                    closest_vertex_idx = idx[0]
                    z_on_surface = mesh_vertices[closest_vertex_idx, 2]
                    normal_on_surface = mesh_normals[closest_vertex_idx]

                    if normal_on_surface[2] < 0:
                        normal_on_surface = -normal_on_surface
                    normal_on_surface /= np.linalg.norm(normal_on_surface)

                    raw_zig_zag_points.append([x_current, y_current, z_on_surface,
                                                        normal_on_surface[0], normal_on_surface[1], normal_on_surface[2]])

            y_current += line_step_y
        
        raw_zig_zag_points_np = np.array(raw_zig_zag_points)

        print(f"  Applying spline smoothing to the trajectory (Smoothing Factor: {TRAJECTORY_SMOOTHING_FACTOR}, Points per segment: {TRAJECTORY_SMOOTHING_POINTS_PER_SEGMENT})...")
        generated_trajectory_points_np = smooth_trajectory_spline(
            raw_zig_zag_points_np, poisson_mesh, kdtree_mesh, mesh_vertices, mesh_normals
        )
        if generated_trajectory_points_np.shape[0] == raw_zig_zag_points_np.shape[0] and raw_zig_zag_points_np.shape[0] > 0:
            print("  Spline smoothing might have failed or returned original points. Check warnings above.")


        if generated_trajectory_points_np.size > 0:
            print(f"Generated {len(generated_trajectory_points_np)} smoothed trajectory points.")
            # Do NOT save here. Saving will happen after kinematics calculation.
            # np.savetxt(OUTPUT_TRAJECTORY_FILENAME, generated_trajectory_points_np,
            #            delimiter=',', header='X,Y,Z,NX,NY,NZ', comments='')
            # print(f"Generated trajectory saved to '{OUTPUT_TRAJECTORY_FILENAME}'")
        else:
            print("Warning: No trajectory points were generated after smoothing. Check mesh bounds, inset, and path parameters, or smoothing parameters.")

# --- Additional Visualization: Trajectory on Mesh (NEW) ---
print("\n--- Additional Visualization: Trajectory on Reconstructed Mesh (Open3D) ---")
if poisson_mesh and poisson_mesh.has_vertices() and generated_trajectory_points_np.size > 0:
    points_cpu = generated_trajectory_points_np[:, :3]
    lines = [[i, i+1] for i in range(len(points_cpu) - 1)]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_cpu),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    print("Opening Open3D viewer for Trajectory on Reconstructed Mesh. Close window to continue.")
    o3d.visualization.draw_geometries([poisson_mesh, line_set, coordinate_axes],
                                     window_name="Trajectory on Reconstructed Mesh",
                                     zoom=0.8,
                                     front=[0.4257, -0.2125, -0.8795],
                                     lookat=[0.05, 0.05, 0.05],
                                     up=[-0.0618, -0.9795, 0.1812])
else:
    print("Cannot display trajectory on mesh: Mesh or trajectory data is missing or empty.")
print("Trajectory on mesh visualization complete (window closed or no data to display).")


# --- Stage 4: Stewart Platform Kinematics and Animation ---
print("\n### Stage 4: Stewart Platform Kinematics and Animation ###")

# --- Functions for Kinematics (All use numpy directly, for stability) ---
def transform_points(points, T_matrix): # T_matrix is now explicitly numpy
    points_np = np.asarray(points)
    T_matrix_np = np.asarray(T_matrix)

    ones = np.ones((points_np.shape[0], 1))
    points_homogeneous = np.hstack((points_np, ones))
    transformed = (T_matrix_np @ points_homogeneous.T).T[:, :3]
    return transformed # Always returns numpy array

def get_camera_to_world_transform(mobile_x, mobile_y, mobile_z, mobile_R):
    mobile_R_np = np.asarray(mobile_R)

    T_mobile_to_world = np.eye(4)
    T_mobile_to_world[:3, :3] = mobile_R_np
    T_mobile_to_world[:3, 3] = np.array([mobile_x, mobile_y, mobile_z])
    
    transformed = T_mobile_to_world @ T_cam_to_mobile_cpu # T_cam_to_mobile_cpu is np
    return transformed # Always returns numpy array

def transform_camera_pose_to_world(x_cam, y_cam, z_cam, nx_cam, ny_cam, nz_cam,
                                   mobile_x, mobile_y, mobile_z, mobile_R):
    T_cam_to_world_result = get_camera_to_world_transform(mobile_x, mobile_y, mobile_z, mobile_R)

    pos_cam_homogeneous = np.array([x_cam, y_cam, z_cam, 1.0])
    normal_cam = np.array([nx_cam, ny_cam, nz_cam])

    pos_world_homogeneous = T_cam_to_world_result @ pos_cam_homogeneous
    normal_world = T_cam_to_world_result[:3, :3] @ normal_cam
    
    return pos_world_homogeneous[0], pos_world_homogeneous[1], pos_world_homogeneous[2], \
           normal_world[0], normal_world[1], normal_world[2]

def minimize_rpy_orientation_for_tool(x_tool, y_tool, z_tool, nx_tool, ny_tool, nz_tool):
    # All inputs are scalars or NumPy arrays, processed with np
    z_axis = np.array([nx_tool, ny_tool, nz_tool], dtype=np.float64)
    if np.linalg.norm(z_axis) < 1e-6:
        print(f"Warning: Zero normal vector for target ({x_tool:.2f},{y_tool:.2f},{z_tool:.2f}). Returning None.")
        return None, None, None, None
    z_axis /= np.linalg.norm(z_axis)

    world_x_ref = np.array([1.0, 0.0, 0.0])
    x_proj = world_x_ref - np.dot(world_x_ref, z_axis) * z_axis
    
    if np.linalg.norm(x_proj) < 1e-6:
        world_y_ref = np.array([0.0, 1.0, 0.0])
        x_proj = world_y_ref - np.dot(world_y_ref, z_axis) * z_axis
        if np.linalg.norm(x_proj) < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis = x_proj / np.linalg.norm(x_proj)
    else:
        x_axis = x_proj / np.linalg.norm(x_proj)
    
    y_axis = np.cross(z_axis, x_axis)
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    
    mobile_xyz = np.array([x_tool, y_tool, z_tool])
    tool_offset_world = rotation_matrix @ tool_position_in_mobile_cpu # tool_position_in_mobile_cpu is np
    mobile_position = mobile_xyz - tool_offset_world

    T_mobile_to_world_result = np.eye(4)
    T_mobile_to_world_result[:3, :3] = rotation_matrix
    T_mobile_to_world_result[:3, 3] = mobile_position

    Pi_world_current = (T_mobile_to_world_result @ np.hstack((Pi_platform_cpu, np.ones((6,1)))).T).T[:, :3] # Pi_platform_cpu is np

    for i in range(6):
        leg_vector = Pi_world_current[i] - Bi_base_cpu[i] # Bi_base_cpu is np
        leg_length = np.linalg.norm(leg_vector)
        
        if not (MIN_LEG_LENGTH_MM <= leg_length <= MAX_LEG_LENGTH_MM):
            print(f"Warning: Target pose ({x_tool:.2f},{y_tool:.2f},{z_tool:.2f}) requires leg {i+1} length {leg_length:.2f}mm, which is outside physical limits [{MIN_LEG_LENGTH_MM}, {MAX_LEG_LENGTH_MM}]mm. Skipping this pose.")
            return None, None, None, None
            
    rpy = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)

    return rotation_matrix, T_mobile_to_world_result, rpy, mobile_position # All returns are np arrays


def create_3d_plotly_visualization_for_frame(mobile_position_cpu, R_mat_cpu, tool_world_cpu, rpy_cpu,
                                             x_world_target_cpu, y_world_target_cpu, z_world_target_cpu, 
                                             nx_world_target_cpu, ny_world_target_cpu, nz_world_target_cpu,
                                             camera_position_current_frame_cpu, camera_axes_current_frame_cpu, # NEW CAMERA INPUTS
                                              mesh_obj=None, T_mesh_to_sbf_cpu_for_plotting=None):
    """
    Creates the Plotly traces for a single frame, given pre-calculated CPU-side pose data.
    This function no longer performs kinematic calculations.
    """
    frame_data_traces = []

    # --- Mesh Trace (T_mesh_to_sbf_cpu_for_plotting is assumed to be NumPy here) ---
    if mesh_obj is not None and mesh_obj.has_vertices() and T_mesh_to_sbf_cpu_for_plotting is not None:
        mesh_vertices_sbf = transform_points(np.asarray(mesh_obj.vertices), T_mesh_to_sbf_cpu_for_plotting) # transform_points returns np
        frame_data_traces.append(go.Mesh3d(
            x=mesh_vertices_sbf[:, 0], y=mesh_vertices_sbf[:, 1], z=mesh_vertices_sbf[:, 2],
            i=np.asarray(mesh_obj.triangles)[:, 0],
            j=np.asarray(mesh_obj.triangles)[:, 1],
            k=np.asarray(mesh_obj.triangles)[:, 2],
            color='lightblue', opacity=0.8, name='Polishing Surface', showlegend=True
        ))

    # All subsequent plot data uses CPU arrays directly from function inputs or global_cpu arrays
    frame_data_traces.append(go.Scatter3d(
        x=Bi_base_cpu[:, 0], y=Bi_base_cpu[:, 1], z=Bi_base_cpu[:, 2],
        mode='markers', marker=dict(size=8, color='red'), name='Base Joints (Fixed)', showlegend=True
    ))
    
    # Recalculate Pi_world_cpu_final from R_mat_cpu and mobile_position_cpu here for plotting
    T_mobile_to_world_cpu_for_plotting = np.eye(4)
    T_mobile_to_world_cpu_for_plotting[:3, :3] = R_mat_cpu
    T_mobile_to_world_cpu_for_plotting[:3, 3] = mobile_position_cpu
    Pi_world_cpu_final = transform_points(Pi_platform_cpu, T_mobile_to_world_cpu_for_plotting)


    frame_data_traces.append(go.Scatter3d(
        x=Pi_world_cpu_final[:, 0], y=Pi_world_cpu_final[:, 1], z=Pi_world_cpu_final[:, 2],
        mode='markers', marker=dict(size=8, color='blue'), name='Platform Joints', showlegend=True
    ))
    for i in range(6):
        frame_data_traces.append(go.Scatter3d(
            x=[Bi_base_cpu[i, 0], Pi_world_cpu_final[i, 0]], y=[Bi_base_cpu[i, 1], Pi_world_cpu_final[i, 1]],
            z=[Bi_base_cpu[i, 2], Pi_world_cpu_final[i, 2]], mode='lines', line=dict(color='black', width=4),
            name=f'Actuator {i+1}' if i==0 else None, showlegend=True if i==0 else False, legendgroup='actuators'
        ))
    platform_vertices = np.vstack([Pi_world_cpu_final, Pi_world_cpu_final[0]])
    frame_data_traces.append(go.Scatter3d(
        x=platform_vertices[:, 0], y=platform_vertices[:, 1], z=platform_vertices[:, 2],
        mode='lines', line=dict(color='blue', width=3), name='Platform', showlegend=True
    ))
    base_vertices_closed = np.vstack([Bi_base_cpu, Bi_base_cpu[0]])
    frame_data_traces.append(go.Scatter3d(
        x=base_vertices_closed[:, 0], y=base_vertices_closed[:, 1], z=base_vertices_closed[:, 2],
        mode='lines', line=dict(color='red', width=3), name='Base', showlegend=True
    ))

    arrow_length_world = 50
    frame_data_traces.extend([
        go.Scatter3d(x=[0, arrow_length_world], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=6), name='World X', showlegend=True),
        go.Scatter3d(x=[0, 0], y=[0, arrow_length_world], z=[0, 0], mode='lines', line=dict(color='green', width=6), name='World Y', showlegend=True),
        go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, arrow_length_world], mode='lines', line=dict(color='blue', width=6), name='World Z', showlegend=True)
    ])

    platform_arrow_length = 50
    frame_data_traces.extend([
        go.Scatter3d(x=[mobile_position_cpu[0], mobile_position_cpu[0] + R_mat_cpu[0, 0] * platform_arrow_length], y=[mobile_position_cpu[1], mobile_position_cpu[1] + R_mat_cpu[1, 0] * platform_arrow_length], z=[mobile_position_cpu[2], mobile_position_cpu[2] + R_mat_cpu[2, 0] * platform_arrow_length], mode='lines', line=dict(color='red', width=4, dash='dash'), name='Platform X', showlegend=True),
        go.Scatter3d(x=[mobile_position_cpu[0], mobile_position_cpu[0] + R_mat_cpu[0, 1] * platform_arrow_length], y=[mobile_position_cpu[1], mobile_position_cpu[1] + R_mat_cpu[1, 1] * platform_arrow_length], z=[mobile_position_cpu[2], mobile_position_cpu[2] + R_mat_cpu[2, 1] * platform_arrow_length], mode='lines', line=dict(color='green', width=4, dash='dash'), name='Platform Y', showlegend=True),
        go.Scatter3d(x=[mobile_position_cpu[0], mobile_position_cpu[0] + R_mat_cpu[0, 2] * platform_arrow_length], y=[mobile_position_cpu[1], mobile_position_cpu[1] + R_mat_cpu[1, 2] * platform_arrow_length], z=[mobile_position_cpu[2], mobile_position_cpu[2] + R_mat_cpu[2, 2] * platform_arrow_length], mode='lines', line=dict(color='blue', width=4, dash='dash'), name='Platform Z', showlegend=True)
    ])

    frame_data_traces.append(go.Scatter3d(
        x=[tool_world_cpu[0]], y=[tool_world_cpu[1]], z=[tool_world_cpu[2]],
        mode='markers', marker=dict(size=15, color='purple', symbol='diamond'), name='Tool Tip (Frame Origin)', showlegend=True
    ))
    tool_arrow_length = 40
    frame_data_traces.extend([
        go.Scatter3d(x=[tool_world_cpu[0], tool_world_cpu[0] + R_mat_cpu[0, 0] * tool_arrow_length], y=[tool_world_cpu[1], tool_world_cpu[1] + R_mat_cpu[1, 0] * tool_arrow_length], z=[tool_world_cpu[2], tool_world_cpu[2] + R_mat_cpu[2, 0] * tool_arrow_length], mode='lines', line=dict(color='red', width=5), name='Tool X', showlegend=True),
        go.Scatter3d(x=[tool_world_cpu[0], tool_world_cpu[0] + R_mat_cpu[0, 1] * tool_arrow_length], y=[tool_world_cpu[1], tool_world_cpu[1] + R_mat_cpu[1, 1] * tool_arrow_length], z=[tool_world_cpu[2], tool_world_cpu[2] + R_mat_cpu[2, 1] * tool_arrow_length], mode='lines', line=dict(color='green', width=5), name='Tool Y', showlegend=True),
        go.Scatter3d(x=[tool_world_cpu[0], tool_world_cpu[0] + R_mat_cpu[0, 2] * tool_arrow_length], y=[tool_world_cpu[1], tool_world_cpu[1] + R_mat_cpu[1, 2] * tool_arrow_length], z=[tool_world_cpu[2], tool_world_cpu[2] + R_mat_cpu[2, 2] * tool_arrow_length], mode='lines', line=dict(color='blue', width=5), name='Tool Z', showlegend=True)
    ])

    tool_diameter_visual = TOOL_CYLINDER_DIAMETER
    tool_length_visual = TOOL_CYLINDER_LENGTH
    tool_radius_visual = tool_diameter_visual / 2

    num_segments_cyl = 20
    theta_cyl = np.linspace(0, 2 * np.pi, num_segments_cyl)
    x_cyl = tool_radius_visual * np.cos(theta_cyl)
    y_cyl = tool_radius_visual * np.sin(theta_cyl)
    
    cylinder_points_local_cpu = np.zeros((num_segments_cyl * 2 + 2, 3))
    cylinder_points_local_cpu[:num_segments_cyl, 0] = tool_radius_visual * np.cos(theta_cyl)
    cylinder_points_local_cpu[:num_segments_cyl, 1] = tool_radius_visual * np.sin(theta_cyl)
    cylinder_points_local_cpu[:num_segments_cyl, 2] = 0.0
    cylinder_points_local_cpu[num_segments_cyl:num_segments_cyl*2, 0] = tool_radius_visual * np.cos(theta_cyl)
    cylinder_points_local_cpu[num_segments_cyl:num_segments_cyl*2, 1] = tool_radius_visual * np.sin(theta_cyl)
    cylinder_points_local_cpu[num_segments_cyl:num_segments_cyl*2, 2] = -tool_length_visual
    cylinder_points_local_cpu[num_segments_cyl*2, :] = [0, 0, 0]
    cylinder_points_local_cpu[num_segments_cyl*2 + 1, :] = [0, 0, -tool_length_visual]

    T_tool_to_world_current_visual_cpu = np.eye(4)
    T_tool_to_world_current_visual_cpu[:3, :3] = R_mat_cpu
    T_tool_to_world_current_visual_cpu[:3, 3] = tool_world_cpu

    cylinder_points_world_visual = transform_points(cylinder_points_local_cpu, T_tool_to_world_current_visual_cpu)
    
    faces_cyl = []
    for i in range(num_segments_cyl):
        j = (i + 1) % num_segments_cyl
        faces_cyl.append([i, j, i + num_segments_cyl])
        faces_cyl.append([j, j + num_segments_cyl, i + num_segments_cyl])
    top_center_idx = num_segments_cyl * 2
    for i in range(num_segments_cyl):
        j = (i + 1) % num_segments_cyl
        faces_cyl.append([top_center_idx, i, j])
    bottom_center_idx = num_segments_cyl * 2 + 1
    for i in range(num_segments_cyl):
        j = (i + 1) % num_segments_cyl
        faces_cyl.append([bottom_center_idx, num_segments_cyl + j, num_segments_cyl + i])

    frame_data_traces.append(go.Mesh3d(
        x=cylinder_points_world_visual[:, 0], y=cylinder_points_world_visual[:, 1], z=cylinder_points_world_visual[:, 2],
        i=np.array(faces_cyl)[:, 0], j=np.array(faces_cyl)[:, 1], k=np.array(faces_cyl)[:, 2],
        color='cyan', opacity=0.7, name='Cylindrical Tool', showlegend=True
    ))

    frame_data_traces.append(go.Scatter3d(
        x=[mobile_position_cpu[0], tool_world_cpu[0]], y=[mobile_position_cpu[1], tool_world_cpu[1]],
        z=[mobile_position_cpu[2], tool_world_cpu[2]], mode='lines', line=dict(color='purple', width=3, dash='dot'),
        name='Tool Offset', showlegend=True
    ))

    frame_data_traces.append(go.Scatter3d(
        x=[camera_position_current_frame_cpu[0]], y=[camera_position_current_frame_cpu[1]], z=[camera_position_current_frame_cpu[2]],
        mode='markers', marker=dict(size=15, color='gold', symbol='diamond'), name='Camera (Moving)', showlegend=True
    ))
    cam_arrow_length = 40
    frame_data_traces.extend([
        go.Scatter3d(x=[camera_position_current_frame_cpu[0], camera_position_current_frame_cpu[0] + camera_axes_current_frame_cpu[0, 0] * cam_arrow_length], y=[camera_position_current_frame_cpu[1], camera_position_current_frame_cpu[1] + camera_axes_current_frame_cpu[1, 0] * cam_arrow_length], z=[camera_position_current_frame_cpu[2], camera_position_current_frame_cpu[2] + camera_axes_current_frame_cpu[2, 0] * cam_arrow_length], mode='lines', line=dict(color='gold', width=4), name='Camera X', showlegend=True, opacity=0.9),
        go.Scatter3d(x=[camera_position_current_frame_cpu[0], camera_position_current_frame_cpu[0] + camera_axes_current_frame_cpu[0, 1] * cam_arrow_length], y=[camera_position_current_frame_cpu[1], camera_position_current_frame_cpu[1] + camera_axes_current_frame_cpu[1, 1] * cam_arrow_length], z=[camera_position_current_frame_cpu[2], camera_position_current_frame_cpu[2] + camera_axes_current_frame_cpu[2, 1] * cam_arrow_length], mode='lines', line=dict(color='lime', width=4), name='Camera Y', showlegend=True, opacity=0.9),
        go.Scatter3d(x=[camera_position_current_frame_cpu[0], camera_position_current_frame_cpu[0] + camera_axes_current_frame_cpu[0, 2] * cam_arrow_length], y=[camera_position_current_frame_cpu[1], camera_position_current_frame_cpu[1] + camera_axes_current_frame_cpu[1, 2] * cam_arrow_length], z=[camera_position_current_frame_cpu[2], camera_position_current_frame_cpu[2] + camera_axes_current_frame_cpu[2, 2] * cam_arrow_length], mode='lines', line=dict(color='deepskyblue', width=4), name='Camera Z', showlegend=True, opacity=0.9)
    ])

    frustum_points = np.array([[0, 0, 0], [-20, -20, 40], [20, -20, 40], [20, 20, 40], [-20, 20, 40]])
    frustum_homogeneous = np.hstack((frustum_points, np.ones((5, 1))))
    T_cam_to_world_current_frame_cpu_for_frustum = np.eye(4) # Re-define this locally
    T_cam_to_world_current_frame_cpu_for_frustum[:3,:3] = camera_axes_current_frame_cpu
    T_cam_to_world_current_frame_cpu_for_frustum[:3,3] = camera_position_current_frame_cpu
    
    frustum_world_current_frame = (T_cam_to_world_current_frame_cpu_for_frustum @ frustum_homogeneous.T).T[:, :3]
    for i in range(1, 5):
        frame_data_traces.append(go.Scatter3d(
            x=[frustum_world_current_frame[0, 0], frustum_world_current_frame[i, 0]], y=[frustum_world_current_frame[0, 1], frustum_world_current_frame[i, 1]],
            z=[frustum_world_current_frame[0, 2], frustum_world_current_frame[i, 2]], mode='lines', line=dict(color='yellow', width=3),
            name='Camera Frustum' if i==1 else None, showlegend=True if i==1 else False, legendgroup='frustum', opacity=0.7
        ))
    far_plane_indices = [1, 2, 3, 4, 1]
    for i in range(len(far_plane_indices) - 1):
        idx1, idx2 = far_plane_indices[i], far_plane_indices[i + 1]
        frame_data_traces.append(go.Scatter3d(
            x=[frustum_world_current_frame[idx1, 0], frustum_world_current_frame[idx2, 0]], y=[frustum_world_current_frame[idx1, 1], frustum_world_current_frame[idx2, 1]],
            z=[frustum_world_current_frame[idx1, 2], frustum_world_current_frame[idx2, 2]], mode='lines', line=dict(color='yellow', width=3), showlegend=False, legendgroup='frustum', opacity=0.7
        ))

    frame_data_traces.append(go.Scatter3d(
        x=[x_world_target_cpu], y=[y_world_target_cpu], z=[z_world_target_cpu],
        mode='markers', marker=dict(size=12, color='magenta', symbol='diamond'), name='Target Pose', showlegend=True
    ))

    normal_length_target = 50
    frame_data_traces.extend([
        go.Scatter3d(x=[x_world_target_cpu, x_world_target_cpu + nx_world_target_cpu * normal_length_target], y=[y_world_target_cpu, y_world_target_cpu + ny_world_target_cpu * normal_length_target],
        z=[z_world_target_cpu, z_world_target_cpu + nz_world_target_cpu * normal_length_target], mode='lines', line=dict(color='magenta', width=6),
        name='Target Normal', showlegend=True, opacity=0.8
    )])
    
    return frame_data_traces


# NEW: Function to calculate all kinematics for the entire trajectory in an optimized way
def calculate_all_kinematics(trajectory_data_SWF):
    all_mobile_positions_list = []
    all_rotation_matrices_list = []
    all_tool_positions_list = []
    all_rpy_angles_list = []
    all_camera_positions_list = [] # New list for camera positions
    all_camera_axes_list = []      # New list for camera axes
    all_x_world_targets_list = []  # New list for x_world_target_cpu
    all_y_world_targets_list = []  # New list for y_world_target_cpu
    all_z_world_targets_list = []  # New list for z_world_target_cpu
    all_nx_world_targets_list = [] # New list for nx_world_target_cpu
    all_ny_world_targets_list = [] # New list for ny_world_target_cpu
    all_nz_world_targets_list = [] # New list for nz_world_target_cpu


    print("\nStarting full kinematic calculation for all trajectory points...")
    
    # --- THIS ENTIRE BLOCK IS NOW CORRECTLY INDENTED ---
    if trajectory_data_SWF.size > 0:
        for i, point_swf_normal_swf in tqdm(enumerate(trajectory_data_SWF), total=len(trajectory_data_SWF), desc="Calculating All Kinematics with Safety Checks"):
            x_pt, y_pt, z_pt, nx_pt, ny_pt, nz_pt = point_swf_normal_swf

            # Mobile pose at the moment the image was captured (set your real values!)
            mobile_pose_at_capture_xyz = (0.0, 0.0, 265.92)   # in mm
            mobile_pose_at_capture_rpy = (0.0, 0.0, 0.0)     # in radians

# Build transform from camera to world
            R_mobile_capture = rpy_to_rot(*mobile_pose_at_capture_rpy)
            T_mobile_to_world_capture = build_transform(R_mobile_capture, mobile_pose_at_capture_xyz)
            T_cam_to_world = T_mobile_to_world_capture @ T_cam_to_mobile_cpu

# Transform the point+normal from camera ’ world
            p_world, n_world = transform_point_and_normal(
                T_cam_to_world,
                (x_pt, y_pt, z_pt),
                (nx_pt, ny_pt, nz_pt)
            )

            x_world_cpu, y_world_cpu, z_world_cpu = p_world
            nx_world_cpu, ny_world_cpu, nz_world_cpu = n_world

            R_mat, T_mobile_to_world, rpy, mobile_position = minimize_rpy_orientation_for_tool(
                x_world_cpu, y_world_cpu, z_world_cpu, nx_world_cpu, ny_world_cpu, nz_world_cpu
            )

            if R_mat is not None:
                # --- SAFETY CHECKS START HERE ---
                # 1. Check Cartesian Position Limits (X, Y, Z)
                if not (X_LIMITS_MM[0] <= mobile_position[0] <= X_LIMITS_MM[1]):
                    print(f"\nWarning: Point {i} skipped. X position {mobile_position[0]:.2f} is outside limits {X_LIMITS_MM}.")
                    continue # Skip to the next point
                
                if not (Y_LIMITS_MM[0] <= mobile_position[1] <= Y_LIMITS_MM[1]):
                    print(f"\nWarning: Point {i} skipped. Y position {mobile_position[1]:.2f} is outside limits {Y_LIMITS_MM}.")
                    continue

                if not (Z_LIMITS_MM[0] <= mobile_position[2] <= Z_LIMITS_MM[1]):
                    print(f"\nWarning: Point {i} skipped. Z position {mobile_position[2]:.2f} is outside limits {Z_LIMITS_MM}.")
                    continue

                # 2. Check Orientation Limits (Roll, Pitch, Yaw)
                if not (ROLL_LIMITS_DEG[0] <= rpy[0] <= ROLL_LIMITS_DEG[1]):
                    print(f"\nWarning: Point {i} skipped. Roll angle {rpy[0]:.2f} is outside limits {ROLL_LIMITS_DEG}.")
                    continue
                    
                if not (PITCH_LIMITS_DEG[0] <= rpy[1] <= PITCH_LIMITS_DEG[1]):
                    print(f"\nWarning: Point {i} skipped. Pitch angle {rpy[1]:.2f} is outside limits {PITCH_LIMITS_DEG}.")
                    continue

                if not (YAW_LIMITS_DEG[0] <= rpy[2] <= YAW_LIMITS_DEG[1]):
                    print(f"\nWarning: Point {i} skipped. Yaw angle {rpy[2]:.2f} is outside limits {YAW_LIMITS_DEG}.")
                    continue
                # --- SAFETY CHECKS END HERE ---

                # If all checks passed, append the data to the lists
                all_mobile_positions_list.append(mobile_position)
                all_rotation_matrices_list.append(R_mat)
                all_rpy_angles_list.append(rpy)
                
                tool_world = mobile_position + R_mat @ tool_position_in_mobile_cpu 
                all_tool_positions_list.append(tool_world)

                T_current_mobile_to_world = np.eye(4)
                T_current_mobile_to_world[:3, :3] = R_mat
                T_current_mobile_to_world[:3, 3] = mobile_position
                
                T_cam_to_world_current_frame = T_current_mobile_to_world @ T_cam_to_mobile_cpu
                camera_position_current_frame = T_cam_to_world_current_frame[:3, 3]
                camera_axes_current_frame = T_cam_to_world_current_frame[:3, :3]

                all_camera_positions_list.append(camera_position_current_frame)
                all_camera_axes_list.append(camera_axes_current_frame)
                
                all_x_world_targets_list.append(x_world_cpu)
                all_y_world_targets_list.append(y_world_cpu)
                all_z_world_targets_list.append(z_world_cpu)
                all_nx_world_targets_list.append(nx_world_cpu)
                all_ny_world_targets_list.append(ny_world_cpu)
                all_nz_world_targets_list.append(nz_world_cpu)

    print("Full kinematic calculation complete. Results are in CPU NumPy arrays.")
    all_mobile_positions_np = np.array(all_mobile_positions_list)
    all_rotation_matrices_np = np.array(all_rotation_matrices_list)
    all_tool_positions_np = np.array(all_tool_positions_list)
    all_rpy_angles_np = np.array(all_rpy_angles_list)
    all_camera_positions_np = np.array(all_camera_positions_list)
    all_camera_axes_np = np.array(all_camera_axes_list)
    
    all_x_world_targets_np = np.array(all_x_world_targets_list)
    all_y_world_targets_np = np.array(all_y_world_targets_list)
    all_z_world_targets_np = np.array(all_z_world_targets_list)
    all_nx_world_targets_np = np.array(all_nx_world_targets_list)
    all_ny_world_targets_np = np.array(all_ny_world_targets_list)
    all_nz_world_targets_np = np.array(all_nz_world_targets_list)

    return all_mobile_positions_np, all_rotation_matrices_np, all_tool_positions_np, all_rpy_angles_np, \
           all_camera_positions_np, all_camera_axes_np, \
           all_x_world_targets_np, all_y_world_targets_np, all_z_world_targets_np, \
           all_nx_world_targets_np, all_ny_world_targets_np, all_nz_world_targets_np

# --- Main Execution Flow of Stage 4.5 and 4.6 ---
print("\n4.5 Transforming trajectory points from Surface World Frame (SWF) / Camera Frame (CF) to Stewart Base Frame (SBF)...")

INITIAL_MOBILE_X_SBF_ANIM = 0.0
INITIAL_MOBILE_Y_SBF_ANIM = 0.0
INITIAL_MOBILE_Z_SBF_ANIM = 265.92
INITIAL_MOBILE_R_SBF_ANIM = np.eye(3)

T_CF_to_SBF_fixed_for_mesh_cpu = np.eye(4)
T_CF_to_SBF_fixed_for_mesh_cpu[:3, :3] = INITIAL_MOBILE_R_SBF_ANIM @ R_cam_to_mobile_cpu
T_CF_to_SBF_fixed_for_mesh_cpu[:3, 3] = np.array([INITIAL_MOBILE_X_SBF_ANIM, INITIAL_MOBILE_Y_SBF_ANIM, INITIAL_MOBILE_Z_SBF_ANIM]) + (INITIAL_MOBILE_R_SBF_ANIM @ camera_position_in_mobile_cpu)

T_CF_to_SBF_fixed_for_mesh = T_CF_to_SBF_fixed_for_mesh_cpu 

print(f"  Fixed Transformation matrix for mesh (from Camera Input Frame to Stewart Base Frame):\n{T_CF_to_SBF_fixed_for_mesh}")


trajectory_data_SWF = generated_trajectory_points_np 

start_kinematics_time = time.time()

all_mobile_positions_np, all_rotation_matrices_np, all_tool_positions_np, all_rpy_angles_np, \
all_camera_positions_np, all_camera_axes_np, \
all_x_world_targets_np, all_y_world_targets_np, all_z_world_targets_np, \
all_nx_world_targets_np, all_ny_world_targets_np, all_nz_world_targets_np = \
    calculate_all_kinematics(trajectory_data_SWF)

print(f"Total calculated poses: {len(all_mobile_positions_np)} points.")

# --- Save Trajectory with XYZ and RPY Angles (NEW LOCATION) ---
# --- Save Trajectory with MOBILE PLATFORM XYZ and RPY Angles ---
if all_mobile_positions_np.size > 0:
    # Combine the calculated mobile platform XYZ with the calculated RPY angles
    # all_mobile_positions_np is already (N, 3) and all_rpy_angles_np is (N, 3)
    platform_pose_data = np.hstack((all_mobile_positions_np, all_rpy_angles_np))

    # Using a more descriptive header
    np.savetxt(OUTPUT_TRAJECTORY_FILENAME, platform_pose_data,
               delimiter=',', header='Platform_X,Platform_Y,Platform_Z,Roll_deg,Pitch_deg,Yaw_deg', comments='')
    print(f"\nTrajectory with MOBILE PLATFORM origin XYZ and RPY angles saved to '{OUTPUT_TRAJECTORY_FILENAME}'")
else:
    print("\nNo trajectory data to save to CSV.")
# --- End Save Trajectory ---


# --- 4.6 Animation Generation ---
print("\n4.6 Generating Stewart Platform Polishing Animation...")

start_animation_frame_gen_time = time.time()

if all_mobile_positions_np.size == 0:
    print("No valid platform poses to animate. Exiting animation stage.")
else:
    if ANIMATION_DOWNSAMPLE_FACTOR > 1:
        animation_mobile_positions = all_mobile_positions_np[::ANIMATION_DOWNSAMPLE_FACTOR]
        animation_rotation_matrices = all_rotation_matrices_np[::ANIMATION_DOWNSAMPLE_FACTOR]
        animation_tool_positions = all_tool_positions_np[::ANIMATION_DOWNSAMPLE_FACTOR]
        animation_rpy_angles = all_rpy_angles_np[::ANIMATION_DOWNSAMPLE_FACTOR]
        animation_camera_positions = all_camera_positions_np[::ANIMATION_DOWNSAMPLE_FACTOR]
        animation_camera_axes = all_camera_axes_np[::ANIMATION_DOWNSAMPLE_FACTOR]
        animation_x_world_targets = all_x_world_targets_np[::ANIMATION_DOWNSAMPLE_FACTOR]
        animation_y_world_targets = all_y_world_targets_np[::ANIMATION_DOWNSAMPLE_FACTOR]
        animation_z_world_targets = all_z_world_targets_np[::ANIMATION_DOWNSAMPLE_FACTOR]
        animation_nx_world_targets = all_nx_world_targets_np[::ANIMATION_DOWNSAMPLE_FACTOR]
        animation_ny_world_targets = all_ny_world_targets_np[::ANIMATION_DOWNSAMPLE_FACTOR]
        animation_nz_world_targets = all_nz_world_targets_np[::ANIMATION_DOWNSAMPLE_FACTOR]
        print(f"  Downsampled animation to {len(animation_mobile_positions)} frames.")
    else:
        animation_mobile_positions = all_mobile_positions_np
        animation_rotation_matrices = all_rotation_matrices_np
        animation_tool_positions = all_tool_positions_np
        animation_rpy_angles = all_rpy_angles_np
        animation_camera_positions = all_camera_positions_np
        animation_camera_axes = all_camera_axes_np
        animation_x_world_targets = all_x_world_targets_np
        animation_y_world_targets = all_y_world_targets_np
        animation_z_world_targets = all_z_world_targets_np
        animation_nx_world_targets = all_nx_world_targets_np
        animation_ny_world_targets = all_ny_world_targets_np
        animation_nz_world_targets = all_nz_world_targets_np


    frame_data_for_animation = []
    if animation_mobile_positions.size > 0:
        for i in tqdm(range(len(animation_mobile_positions)), desc="Generating Plotly Animation Frames"):
            mobile_pos_sbf = animation_mobile_positions[i]
            R_mat_sbf = animation_rotation_matrices[i]
            tool_pos_sbf = animation_tool_positions[i]
            rpy_sbf = animation_rpy_angles[i]
            
            x_world_target_cpu = animation_x_world_targets[i]
            y_world_target_cpu = animation_y_world_targets[i]
            z_world_target_cpu = animation_z_world_targets[i]
            nx_world_target_cpu = animation_nx_world_targets[i]
            ny_world_target_cpu = animation_ny_world_targets[i]
            nz_world_target_cpu = animation_nz_world_targets[i]

            camera_pos_current_frame = animation_camera_positions[i]
            camera_axes_current_frame = animation_camera_axes[i]

            current_frame_traces = create_3d_plotly_visualization_for_frame(
                mobile_pos_sbf, R_mat_sbf, tool_pos_sbf, rpy_sbf,
                x_world_target_cpu=x_world_target_cpu, y_world_target_cpu=y_world_target_cpu, z_world_target_cpu=z_world_target_cpu,
                nx_world_target_cpu=nx_world_target_cpu, ny_world_target_cpu=ny_world_target_cpu, nz_world_target_cpu=nz_world_target_cpu,
                camera_position_current_frame_cpu=camera_pos_current_frame, camera_axes_current_frame_cpu=camera_axes_current_frame,
                mesh_obj=poisson_mesh, T_mesh_to_sbf_cpu_for_plotting=T_CF_to_SBF_fixed_for_mesh
            )
            frame_data_for_animation.append(current_frame_traces)
    else:
        print("No valid platform poses to animate after downsampling. Skipping animation generation.")


end_animation_frame_gen_time = time.time()
print(f"  Animation frames generation took {end_animation_frame_gen_time - start_animation_frame_gen_time:.2f} seconds.")

if not frame_data_for_animation:
    print("No valid platform poses to animate. Exiting animation stage.")
else:
    fig_anim = go.Figure(data=frame_data_for_animation[0], frames=[go.Frame(data=f, name=str(i)) for i, f in enumerate(frame_data_for_animation)])

    fig_anim.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 1000/ANIMATION_FPS, "redraw": True}, "fromcurrent": True}]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )],
        sliders=[dict(
            steps=[dict(args=[[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        label=str(k),
                        method="animate") for k, f in enumerate(fig_anim.frames)],
            transition=dict(duration=0),
            x=0,
            y=0,
            currentvalue={"prefix": "Frame: "},
            len=0.8
        )],
        scene=dict(
            xaxis=dict(range=[-300, 300], title='X (mm)'),
            yaxis=dict(range=[-300, 300], title='Y (mm)'),
            zaxis=dict(range=[0, 600], title='Z (mm)'),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=0.8)
            )
        ),
        title=f'Stewart Platform Polishing Animation (Total Frames: {len(frame_data_for_animation)})',
        width=1200,
        height=900,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    print(f"  Animation frames generation took {end_animation_frame_gen_time - start_animation_frame_gen_time:.2f} seconds.")

    print(f"  Displaying interactive animation. Attempting to save to '{ANIMATION_OUTPUT_FILENAME.replace('.mp4', '.html')}' and '{ANIMATION_OUTPUT_FILENAME}'...")
    
    output_dir = os.path.dirname(ANIMATION_OUTPUT_FILENAME)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"  Created output directory: {output_dir}")
        except OSError as e:
            print(f"  Error creating directory {output_dir}: {e}. Please check permissions.")
            output_dir = "."
            ANIMATION_OUTPUT_FILENAME = os.path.join(output_dir, os.path.basename(ANIMATION_OUTPUT_FILENAME))
            print(f"  Attempting to save to current working directory: {ANIMATION_OUTPUT_FILENAME}")
            

    # --- HTML Save ---
    html_output_filename = ANIMATION_OUTPUT_FILENAME.replace('.mp4', '.html')
    html_output_filename_web_safe = html_output_filename.replace('\\', '/')
    try:
        fig_anim.write_html(html_output_filename_web_safe, auto_play=True)
        print(f"  Interactive HTML animation saved to '{html_output_filename}'.")
    except Exception as e:
        print(f"  Error saving HTML animation: {e}")
        print(f"  Failed to save HTML to '{html_output_filename}'.")


    # --- MP4 Save ---
    try:
        fig_anim.write_image(ANIMATION_OUTPUT_FILENAME, format='mp4', engine='kaleido',
                              scale=1.5, width=1200, height=900,
                              animation_duration=(len(animation_frames) / ANIMATION_FPS) * 1000,
                              animation_frame_duration=int(1000/ANIMATION_FPS))
        print(f"  MP4 animation saved to '{ANIMATION_OUTPUT_FILENAME}'.")
    except Exception as e:
        print(f"  Error saving MP4 animation using Kaleido: {e}")
        print("  Please ensure 'kaleido' is installed correctly (pip install kaleido).")
        print("  If it is installed, check for display server issues or try running from a non-headless environment.")
        print("  You can still view the interactive HTML version in your browser (if it saved successfully).")
        print("  Skipping MP4 save.")

print("\n--- MRF Polishing Pipeline Complete ---")