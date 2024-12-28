import os
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
from tqdm import tqdm
from sklearn.neighbors import KDTree
import re
from scipy.spatial.transform import Rotation as Rot

import re

def extract_az_el(image_name):
    """Extract azimuth and elevation (including floats) from the image filename."""
    import re
    match = re.match(r'(\d+)_az([-+]?\d+\.\d+)_el([-+]?\d+\.\d+)\.jpg', image_name)
    if match:
        azimuth = float(match.group(2))
        elevation = float(match.group(3))
        return azimuth, elevation
    else:
        raise ValueError(f"Filename {image_name} does not match expected pattern.")

def rotation_matrix_from_az_el(azimuth, elevation):
    """Generate a rotation matrix based on azimuth and elevation."""
    # Convert angles to radians
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    

    
    # Combined rotation matrix
    rotation_matrix = Rot.from_euler('xyz',[0,elevation,azimuth]).as_matrix()
    
    return rotation_matrix


def rgb2gray(image):
    """Convert an RGB image to grayscale."""
    return image.convert('L')

def gaussian_pyramid(image, num_octaves=4, num_scales=5, sigma=1.6):
    """Generate a Gaussian pyramid using NumPy."""
    if isinstance(image, np.ndarray) and image.ndim == 3:
        image = image.mean(axis=-1)  # Convert to grayscale if needed
    pyramid = []
    for _ in range(num_octaves):
        octave = [image]
        for _ in range(num_scales - 1):
            image = gaussian_filter(image, sigma)
            octave.append(image)
            sigma *= 2
        pyramid.append(octave)
        # Downsample the image (reduce resolution by half)
        image = image[::2, ::2]
    return pyramid

def dog_pyramid(gaussian_pyramid):
    """Generate a Difference of Gaussian (DoG) pyramid."""
    dog_pyr = []
    for octave in gaussian_pyramid:
        dog_octave = [octave[i + 1] - octave[i] for i in range(len(octave) - 1)]
        dog_pyr.append(dog_octave)
    return dog_pyr

def detect_keypoints(dog_pyramid, threshold=0.005):
    """Detect keypoints in the Difference-of-Gaussian pyramid."""
    keypoints = []
    for o, octave in enumerate(dog_pyramid):
        for i, image in enumerate(octave):
            image = np.array(image)
            local_max = (image == maximum_filter(image, size=3))  # Find local maxima
            local_min = (image == minimum_filter(image, size=3))  # Find local minima
            keypoint_mask = (local_max | local_min) & (np.abs(image) > threshold)
            y, x = np.nonzero(keypoint_mask)  # Find coordinates of keypoints
            keypoints.extend([(x[j], y[j], o, i) for j in range(len(x))])
    return keypoints

def eliminate_low_contrast_keypoints(keypoints, dog_pyramid, contrast_threshold=0.008):
    """Eliminate low-contrast keypoints."""
    high_contrast_keypoints = []
    for kp in keypoints:
        x, y, o, i = kp
        if np.abs(dog_pyramid[o][i][y, x]) > contrast_threshold:
            high_contrast_keypoints.append(kp)
    return high_contrast_keypoints

def assign_orientations(keypoints, gaussian_pyramid):
    """Assign orientations to keypoints."""
    oriented_keypoints = []
    for kp in keypoints:
        x, y, o, i = kp
        image = gaussian_pyramid[o][i]
        
        # Skip keypoints near the borders of the image
        if x <= 0 or y <= 0 or x >= image.shape[1] - 1 or y >= image.shape[0] - 1:
            continue
        
        magnitude = np.sqrt((image[y+1, x] - image[y-1, x])**2 + (image[y, x+1] - image[y, x-1])**2)
        orientation = np.arctan2(image[y+1, x] - image[y-1, x], image[y, x+1] - image[y, x-1])
        oriented_keypoints.append((x, y, o, i, magnitude, orientation))
    return oriented_keypoints

def generate_descriptors(oriented_keypoints, gaussian_pyramid):
    """Generate descriptors efficiently."""
    descriptors = []
    for kp in oriented_keypoints:
        x, y, o, i, magnitude, orientation = kp
        image = gaussian_pyramid[o][i]
        # Extract patch efficiently using slicing
        patch = image[max(0, y-4):y+4, max(0, x-4):x+4]
        if patch.shape != (8, 8):  # Skip patches that are too small
            continue
        grad_y, grad_x = np.gradient(patch)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_ori = np.arctan2(grad_y, grad_x) - orientation
        descriptor = (grad_mag * np.cos(grad_ori)).flatten()
        descriptors.append(descriptor)
    return descriptors

def process_single_image(image_path):
    """Process a single image."""
    image = Image.open(image_path).convert('L')  # Grayscale
    image = np.array(image, dtype=np.float32)

    # Generate Gaussian and DoG pyramids
    gaussian_pyr = gaussian_pyramid(image)
    dog_pyr = dog_pyramid(gaussian_pyr)

    # Detect and filter keypoints
    keypoints = detect_keypoints(dog_pyr)
    high_contrast_kp = eliminate_low_contrast_keypoints(keypoints, dog_pyr)
    oriented_kp = assign_orientations(high_contrast_kp, gaussian_pyr)

    # Generate descriptors
    descriptors = generate_descriptors(oriented_kp, gaussian_pyr)
    return oriented_kp, descriptors

def process_images_parallel(image_directory):
    """Process images in parallel."""
    image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('jpg', 'png'))]
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_single_image, image_files), total=len(image_files), desc="Processing Images"))
    return {os.path.basename(image_files[i]): results[i] for i in range(len(image_files))}

def match_keypoints(descriptors1, descriptors2, threshold=0.7):
    """Match keypoints using KDTree for efficiency."""
    tree = KDTree(descriptors2)
    matches = []
    for i, desc1 in enumerate(descriptors1):
        dist, ind = tree.query([desc1], k=1)
        if dist[0][0] < threshold:
            matches.append((i, ind[0][0]))
    return matches

def triangulate_points(matched_keypoints, keypoints1, keypoints2, camera_matrix1, camera_matrix2):
    """Triangulate 3D points from matched keypoints."""
    points3D = []
    for kp1_idx, kp2_idx in matched_keypoints:
        x1, y1= keypoints1[kp1_idx][:2]
        x2, y2= keypoints2[kp2_idx][:2]
        A = np.array([
            [x1 * camera_matrix1[2, :] - camera_matrix1[0, :]],
            [y1 * camera_matrix1[2, :] - camera_matrix1[1, :]],
            [x2 * camera_matrix2[2, :] - camera_matrix2[0, :]],
            [y2 * camera_matrix2[2, :] - camera_matrix2[1, :]]
        ]).reshape(-1, 4)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]
        points3D.append(X[:3])
    return np.array(points3D)

def create_point_cloud(points3D):
    """Visualize a 3D point cloud."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


if __name__ == "__main__":
    # Directory containing your images
    image_dir = "/storage/emulated/0/skanfon/img/"

    # Process all images in the directory
    image_descriptors = process_images_parallel(image_dir)

    # Match keypoints between pairs of images and triangulate points to create 3D points
    all_points3D = []
    image_files = list(image_descriptors.keys())
    for i in tqdm(range(len(image_files) - 1)):
        keypoints1, descriptors1 = image_descriptors[image_files[i]]
        keypoints2, descriptors2 = image_descriptors[image_files[i+1]]
        
        # Extract azimuth and elevation
        az1, el1 = extract_az_el(image_files[i])
        az2, el2 = extract_az_el(image_files[i+1])
        
        matches = match_keypoints(np.array(descriptors1), np.array(descriptors2))
        f_pixels = 26000  # Focal length in pixels
        cx = 3072 / 2  # Principal point X-coordinate
        cy = 4096 / 2  # Principal point Y-coordinate
        
        camera_matrix1 = rotation_matrix_from_az_el(az1, el1) @ np.array([[f_pixels, 0, cx], 
                                                                          [0, f_pixels, cy], 
                                                                          [0, 0, 1]])
        camera_matrix2 = rotation_matrix_from_az_el(az2, el2) @ np.array([[f_pixels, 0, cx], 
                                                                  [0, f_pixels, cy], 
                                                                  [0, 0, 1]])

        points3D = triangulate_points(matches, keypoints1, keypoints2, camera_matrix1, camera_matrix2)
        all_points3D.extend(points3D)

    # Create and visualize point cloud
    all_points3D = np.array(all_points3D)
    np.save("/storage/emulated/0/skanfon/3dcloud.npy", all_points3D)
    create_point_cloud(all_points3D)
