import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
import re
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Directory to images
image_dir = "/storage/emulated/0/skanfon/img/"
output_file = "/storage/emulated/0/skanfon/bckprjct.npy"
radius = 1000.0

def point_to_line_distance(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    v = np.array([x2 - x1, y2 - y1, z2 - z1])
    w = np.array([x3 - x1, y3 - y1, z3 - z1])
    v_magnitude = np.linalg.norm(v)
    if v_magnitude == 0:
        raise ValueError("The two points defining the line are identical.")
    cross_product = np.cross(v, w)
    cross_magnitude = np.linalg.norm(cross_product)
    return cross_magnitude / v_magnitude

def rotation_matrix_from_euler(a, b, c):
    return Rot.from_euler('xyz', [a, b, c]).as_matrix()

def convert_rgb_to_grayscale(image):
    return np.asarray(image.convert('L'), dtype=np.float32)

def process_file(file):
    try:
        if not file.endswith(".jpg"):
            return None

        match = re.match(r"(-?\d+\.\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)\.jpg", file)
        if not match:
            return None

        a, b, c = map(float, match.groups())
        a, b, c = np.deg2rad([a, b, c])
        rotmat = rotation_matrix_from_euler(a, b, c)
        normvec = np.array([0, 0, 1])
        mainvec = np.dot(rotmat, normvec) * radius

        img_path = os.path.join(image_dir, file)
        img = convert_rgb_to_grayscale(Image.open(img_path).resize((480, 640), Image.LANCZOS))

        h, w = 640,480
        vox_local = np.zeros((h,w, 64, 1))

        for i in range(h):
            for j in range(w):
                for zz in range(len(vox_local[0, 0])):
                    x, y, z = np.dot(rotmat, np.array([i - h // 2, j - w // 2, 0])) + mainvec
                    if np.abs(point_to_line_distance(
                        i - len(vox_local) // 2, j - len(vox_local[0]) // 2, 0,
                        x, y, z,
                        i - len(vox_local) // 2, j - len(vox_local[0]) // 2, zz
                    )) < 0.5:
                        vox_local[i, j, zz, 0] += img[i, j]
        return vox_local
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None

def main():
    if not os.path.exists(image_dir):
        print(f"Directory does not exist: {image_dir}")
        return

    files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    if not files:
        print("No files found in the directory.")
        return

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files)))

    # Combine results from all processes
    vox = np.zeros((640,480, 64, 1))
    for result in results:
        if result is not None:
            vox += result

    np.save(output_file, vox)
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    main()
