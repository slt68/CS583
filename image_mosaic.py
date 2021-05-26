import numpy as np
import imageio
import cv2 as CV

from PIL import Image
from hw3 import track_object, visualize

def bilinear_interp(image, points):
    """Given an image and an array of row/col (Y/X) points, perform bilinear
    interpolation and return the pixel values in the image at those points."""
    points = np.asarray(points)
    if points.ndim == 1:
        points = points[np.newaxis]

    valid = np.all(points < [image.shape[0]-1, image.shape[1]-1], axis=-1)
    valid *= np.all(points >= 0, axis=-1)
    valid = valid.astype(np.float32)
    points = np.minimum(points, [image.shape[0]-2, image.shape[1]-2])
    points = np.maximum(points, 0)

    fpart, ipart = np.modf(points)
    tl = ipart.astype(np.int32)
    br = tl+1
    tr = np.concatenate([tl[..., 0:1], br[..., 1:2]], axis=-1)
    bl = np.concatenate([br[..., 0:1], tl[..., 1:2]], axis=-1)

    b = fpart[..., 0:1]
    a = fpart[..., 1:2]

    top = (1-a) * image[tl[..., 0], tl[..., 1]] + \
        a * image[tr[..., 0], tr[..., 1]]
    bot = (1-a) * image[bl[..., 0], bl[..., 1]] + \
        a * image[br[..., 0], br[..., 1]]
    return ((1-b) * top + b * bot) * valid[..., np.newaxis]


def load_image(filename):
    """Loads the provided image file, and returns it as a numpy array."""
    im = Image.open(filename)
    return np.array(im)


def project_to_cyl(image, s, k1, k2):
    corners = np.array([(0, 0), (image.shape[1], 0),  (image.shape[1], image.shape[0]), (0, image.shape[0])])
    xc = image.shape[1] / 2
    yc = image.shape[0] / 2

    pts = np.mgrid[:image.shape[0], :image.shape[1]
          ].transpose(1, 2, 0).astype(np.float32)


    # convert from cylindrical coordinates to h/theta
    theta = (pts[:, 0] - xc) / s
    h = (pts[:, 1] - yc) / s

    # get point on the cylinder
    x_hat = np.sin(theta)
    y_hat = h
    z_hat = np.cos(theta)

    # normalize cylinder to input image
    x_norm = x_hat / z_hat
    y_norm = y_hat / z_hat

    r2 = x_norm**2 + y_norm**2

    # apply radial distortion correction
    x_distort = x_norm * (1 + k1 * r2 + k2 * r2**2)
    y_distort = y_norm * (1 + k1 * r2 + k2 * r2 ** 2)

    x = s * x_distort + xc
    y = s * y_distort + yc

    bilinear_interp(image,)


if __name__ == "__main__":
    f = open('test_data/test_files.txt', 'r')
    lines = f.readlines()
    f.close()
    images = []
    for line in lines:
        images.append(load_image(line.strip().split(' ')[0]))

    f = open('test_data/test_params.txt')
    camera_params = f.readline().strip().split(' ')
    f.close()

    s = int(camera_params[0])
    xc = float(camera_params[2])
    yc = float(camera_params[3])

    cyl_images = [project_to_cyl(x, s, xc, yc) for x in images]
