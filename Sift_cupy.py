import os
import cupy as cp
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from cupyx.scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
from sklearn.neighbors import KDTree
from scipy.spatial.transform import Rotation as Rot


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
    azimuth = cp.deg2rad(azimuth)
    elevation = cp.deg2rad(elevation)
    rotation_matrix = Rot.from_euler('xyz', [0, elevation, azimuth]).as_matrix()
    return rotation_matrix


def gaussian_pyramid(image, num_octaves=4, num_scales=5, sigma=1.6):
    """Generate a Gaussian pyramid using CuPy."""
    pyramid = []
    for _ in range(num_octaves):
        octave = [image]
        for _ in range(num_scales - 1):
            image = gaussian_filter(image, sigma)
            octave.append(image)
            sigma *= 2
        pyramid.append(octave)
        image = image[::2, ::2]  # Downsample
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
            local_max = image == maximum_filter(image, size=3)
            local_min = image == minimum_filter(image, size=3)
            keypoint_mask = (local_max | local_min) & (cp.abs(image) > threshold)
            y, x = cp.nonzero(keypoint_mask)
            keypoints.extend([(x[j], y[j], o, i) for j in range(len(x))])
    return keypoints


def eliminate_low_contrast_keypoints(keypoints, dog_pyramid, contrast_threshold=0.008):
    """Eliminate low-contrast keypoints."""
    high_contrast_keypoints = []
    for kp in keypoints:
        x, y, o, i = kp
        if cp.abs(dog_pyramid[o][i][y, x]) > contrast_threshold:
            high_contrast_keypoints.append(kp)
    return high_contrast_keypoints


def assign_orientations(keypoints, gaussian_pyramid):
    """Assign orientations to keypoints."""
    oriented_keypoints = []
    for kp in keypoints:
        x, y, o, i = kp
        image = gaussian_pyramid[o][i]
        if x <= 0 or y <= 0 or x >= image.shape[1] - 1 or y >= image.shape[0] - 1:
            continue
        grad_y, grad_x = cp.gradient(image)
        magnitude = cp.sqrt(grad_x[y, x]**2 + grad_y[y, x]**2)
        orientation = cp.arctan2(grad_y[y, x], grad_x[y, x])
        oriented_keypoints.append((x, y, o, i, magnitude, orientation))
    return oriented_keypoints


def generate_descriptors(oriented_keypoints, gaussian_pyramid):
    """Generate descriptors efficiently using CuPy."""
    descriptors = []
    for kp in oriented_keypoints:
        x, y, o, i, magnitude, orientation = kp
        image = gaussian_pyramid[o][i]
        patch = image[max(0, y-4):y+4, max(0, x-4):x+4]
        if patch.shape != (8, 8):
            continue
        grad_y, grad_x = cp.gradient(patch)
        grad_mag = cp.sqrt(grad_x**2 + grad_y**2)
        grad_ori = cp.arctan2(grad_y, grad_x) - orientation
        descriptor = (grad_mag * cp.cos(grad_ori)).flatten()
        descriptors.append(descriptor)
    return descriptors


def process_single_image(image_path):
    """Process a single image."""
    image = Image.open(image_path).convert('L')
    image = cp.array(image, dtype=cp.float32)

    gaussian_pyr = gaussian_pyramid(image)
    dog_pyr = dog_pyramid(gaussian_pyr)

    keypoints = detect_keypoints(dog_pyr)
    high_contrast_kp = eliminate_low_contrast_keypoints(keypoints, dog_pyr)
    oriented_kp = assign_orientations(high_contrast_kp, gaussian_pyr)

    descriptors = generate_descriptors(oriented_kp, gaussian_pyr)
    return oriented_kp, descriptors


def process_images(image_directory):
    """Process images sequentially."""
    image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('jpg', 'png'))]
    results = {}
    for image_file in tqdm(image_files, desc="Processing Images"):
        results[os.path.basename(image_file)] = process_single_image(image_file)
    return results


def triangulate_points(matches, keypoints1, keypoints2, camera_matrix1, camera_matrix2):
    """Triangulate 3D points from matched keypoints."""
    points3D = []
    for kp1_idx, kp2_idx in matches:
        x1, y1 = keypoints1[kp1_idx][:2]
        x2, y2 = keypoints2[kp2_idx][:2]
        A = cp.array([
            x1 * camera_matrix1[2, :] - camera_matrix1[0, :],
            y1 * camera_matrix1[2, :] - camera_matrix1[1, :],
            x2 * camera_matrix2[2, :] - camera_matrix2[0, :],
            y2 * camera_matrix2[2, :] - camera_matrix2[1, :]
        ])
        _, _, Vt = cp.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]
        points3D.append(X[:3])
    return cp.array(points3D)


def create_point_cloud(points3D):
    """Visualize a 3D point cloud."""
    points3D = cp.asnumpy(points3D)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


if __name__ == "__main__":
    image_dir = "/path/to/your/image/directory/"
    image_descriptors = process_images(image_dir)

    all_points3D = []
    image_files = list(image_descriptors.keys())
    for i in range(len(image_files) - 1):
        keypoints1, descriptors1 = image_descriptors[image_files[i]]
        keypoints2, descriptors2 = image_descriptors[image_files[i+1]]

        az1, el1 = extract_az_el(image_files[i])
        az2, el2 = extract_az_el(image_files[i+1])

        matches = KDTree(cp.asnumpy(cp.stack(descriptors2))).query(cp.asnumpy(cp.stack(descriptors1)), k=1, return_distance=False)
        matches = [(i, match[0]) for i, match in enumerate(matches) if len(match) > 0]

        f_pixels = 26000
        cx, cy = 3072 / 2, 4096 / 2
        camera_matrix1 = rotation_matrix_from_az_el(az1, el1) @ cp.array([[f_pixels, 0, cx], [0, f_pixels, cy], [0, 0, 1]])
        camera_matrix2 = rotation_matrix_from_az_el(az2, el2) @ cp.array([[f_pixels, 0, cx], [0, f_pixels, cy], [0, 0, 1]])

        points3D = triangulate_points(matches, keypoints1, keypoints2, camera_matrix1, camera_matrix2)
        all_points3D.append(points3D)

    all_points3D = cp.concatenate(all_points3D, axis=0)
    create_point_cloud(all_points3D)
