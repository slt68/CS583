import numpy as np
import imageio
import cv2 as CV
from scipy.ndimage.filters import convolve

from PIL import Image
from hw3 import track_object, visualize

#alamkin: adding lucas kanade algo from hw3.py and blending from hw2
# -- BEGIN --
def blend_with_mask(source, target, mask):
    """
    Blends the source image with the target image according to the mask.
    Pixels with value "1" are source pixels, "0" are target pixels, and
    intermediate values are interpolated linearly between the two.

    Args:
        source:     The source image.
        target:     The target image.
        mask:       The mask to use

    Returns:
        A new image representing the linear combination of the mask (and it's inverse)
        with source and target, respectively.
    """

    # TODO: First, convert the mask image to be a floating point between 0 and 1
    mask = mask.astype(np.float32)
    m = mask / np.max(mask)

    # TODO: Next, use it to make a linear combination of the pixels
    result = (1-m)*source + m*target

    # TODO: Convert the result to be the same type as source and return the result
    result = result.astype(source.dtype)

    return result

def translate(image, displacement):
    """Takes an image and a displacement of the form X,Y and translates the
    image by the displacement. The shape of the output is the same as the
    input, with missing pixels filled in with zeros."""
    pts = np.mgrid[:image.shape[0], :image.shape[1]
                   ].transpose(1, 2, 0).astype(np.float32)
    pts -= displacement[::-1]

    return bilinear_interp(image, pts)

def convolve_img(image, kernel):
    """Convolves an image with a convolution kernel. Kernel should either have
    the same number of dimensions and channels (last dimension shape) as the
    image, or should have 1 less dimension than the image."""
    if kernel.ndim == image.ndim:
        if image.shape[-1] == kernel.shape[-1]:
            return np.dstack([convolve(image[..., c], kernel[..., c]) for c in range(kernel.shape[-1])])
        elif image.ndim == 2:
            return convolve(image, kernel)
        else:
            raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
                kernel.shape, image.shape))
    elif kernel.ndim == image.ndim - 1:
        return np.dstack([convolve(image[..., c], kernel) for c in range(image.shape[-1])])
    else:
        raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
            kernel.shape, image.shape))


def gaussian_kernel(ksize=5):
    """
    Computes a 2-d gaussian kernel of size ksize and returns it.
    """
    kernel = np.exp(-np.linspace(-(ksize//2), ksize//2, ksize)
                    ** 2 / 2) / np.sqrt(2*np.pi)
    kernel = np.outer(kernel, kernel)
    kernel /= kernel.sum()
    return kernel


def lucas_kanade(H, I):
    """Given images H and I, compute the displacement that should be applied to
    H so that it aligns with I."""
    # motion in dark regions is difficult to estimate. Generate a binary mask
    # indicating pixels that are valid (average color value > 0.25) in both H
    # and I.
    mask = (H.mean(-1) > 0.25) * (I.mean(-1) > 0.25)
    mask = mask[:, :, np.newaxis]

    # Compute the partial image derivatives w.r.t. X, Y, and Time (t).
    # In other words, compute I_y, I_x, and I_t
    # To achieve this, use a _normalized_ 3x3 sobel kernel and the convolve_img
    # function above. NOTE: since you're convolving the kernel, you need to 
    # multiply it by -1 to get the proper direction.
    xsobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) * -1/8
    ysobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) * -1/8

    Ix = convolve_img(I, xsobel)
    Iy = convolve_img(I, ysobel)
    It = I - H

    # Compute the various products (Ixx, Ixy, Iyy, Ixt, Iyt) necessary to form
    # AtA. Apply the mask to each product.
    Ixx = (Ix * Ix) * mask
    Ixy = (Ix * Iy) * mask
    Iyy = (Iy * Iy) * mask

    Ixt = (Ix * It) * mask
    Iyt = (Iy * It) * mask

    # Build the AtA matrix and Atb vector. You can use the .sum() function on numpy arrays to help.
    AtA = np.array([[Ixx.sum(), Ixy.sum()], [Ixy.sum(), Iyy.sum()]])
    Atb = np.array([Ixt.sum(), Iyt.sum()])

    # Solve for the displacement using linalg.solve

    displacement = np.linalg.solve(AtA, Atb)

    # return the displacement and some intermediate data for unit testing..
    return displacement, AtA, Atb


def iterative_lucas_kanade(H, I, steps):
    # Run the basic Lucas Kanade algorithm in a loop `steps` times.
    # Start with an initial displacement of 0 and accumulate displacements.
    disp = np.zeros((2,), np.float32)
    for i in range(steps):
        # Translate the H image by the current displacement (using the translate function above)
        trans_H = translate(H, disp)
        # run Lucas Kanade and update the displacement estimate
        disp += lucas_kanade(trans_H, I)[0]
    # Return the final displacement
    return disp


def gaussian_pyramid(image, levels):
    """
    Builds a Gaussian pyramid for an image with the given number of levels, then return it.
    Inputs:
        image: a numpy array (i.e., image) to make the pyramid from
        levels: how many levels to make in the gaussian pyramid
    Retuns:
        An array of images where each image is a blurred and shruken version of the first.
    """

    # Compute a gaussian kernel using the gaussian_kernel function above. You can leave the size as default.
    kernel = gaussian_kernel()
    # Add image to the the list as the first level
    pyr = [image]
    for level in range(1, levels):
        # Convolve the previous image with the gussian kernel
        tmp = convolve_img(pyr[-1], kernel)
        # decimate the convolved image by downsampling the pixels in both dimensions.
        # Note: you can use numpy advanced indexing for this (i.e., ::2)

        # add the sampled image to the list
        pyr.append(tmp[::2, ::2])
    return pyr


def pyramid_lucas_kanade(H, I, initial_d, levels, steps):
    """Given images H and I, and an initial displacement that roughly aligns H
    to I when applied to H, run Iterative Lucas Kanade on a pyramid of the
    images with the given number of levels to compute the refined
    displacement."""

    initial_d = np.asarray(initial_d, dtype=np.float32)

    # Build Gaussian pyramids for the two images.

    H_pyr = gaussian_pyramid(H, levels)
    I_pyr = gaussian_pyramid(I, levels)

    # Start with an initial displacement (scaled to the coarsest level of the
    # pyramid) and compute the updated displacement at each level using Lucas
    # Kanade.

    scalar = 2.

    disp = initial_d / 2.**(levels)

    for level in range(levels):
        # Get the two images for this pyramid level.
        H_cur = H_pyr[-(level+1)]
        I_cur = I_pyr[-(level+1)]

        # Scale the previous level's displacement and apply it to one of the
        # images via translation.

        I_trans = translate(I_cur, -disp * scalar)

        # Use the iterative Lucas Kanade method to compute a displacement
        # between the two images at this level.
        cur_disp = iterative_lucas_kanade(H_cur, I_trans, steps)

        # Update the displacement based on the one you just computed.
        disp = (disp * scalar) + cur_disp

    # Return the final displacement.
    return disp

#-- END --

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


def project_to_cyl(image, s, k1, k2, image_filename):
    corners = np.array([(0, 0), (image.shape[1], 0),  (image.shape[1], image.shape[0]), (0, image.shape[0])])
    xc = image.shape[1] / 2
    yc = image.shape[0] / 2

    #alamkin - what's this for? 
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
    # alamkin - shouldn't r2**2 be r2**4?
    x_distort = x_norm * (1 + k1 * r2 + k2 * r2**2)
    y_distort = y_norm * (1 + k1 * r2 + k2 * r2 ** 2)

    #alamkin - should x and y be a (480,1) and not (480,2)?
    x = s * x_distort + xc
    y = s * y_distort + yc

    #alamkin - generate new image by getting pixel val from org image 
    # create array of x y corrdinates
    xy_coor = np.column_stack((np.array(x), np.array(y)))

    # iterate through image points and apply bilinear interpolation 
    # (shouldn't output image be same size as input image?)
    height = image.shape[0]
    width = image.shape[1]
    res = np.zeros((height, width))
    idx=0
    
    '''
    for i in range(height):
        for j in range(width):
                res[i][j] = bilinear_interp(image, xy_coor[idx])
                idx++
    '''

    # return res

    # write reprojected image to file (temporary)
    out_fn = image_filename.split(".")[0] + "_cyl." + image_filename.split(".")[1]
    imageio.imwrite(out_fn, res)
    
if __name__ == "__main__":
    f = open('test_data/test_files.txt', 'r')
    lines = f.readlines()
    f.close()
    images = []
    img_fns = []
    for line in lines:
        images.append(load_image(line.strip().split(' ')[0]))
        #alamkin - get image filenames (temporary)
        img_fns.append(line.strip().split(' ')[0])

    f = open('test_data/test_params.txt')
    camera_params = f.readline().strip().split(' ')
    f.close()

    s = int(camera_params[0])
    xc = float(camera_params[2])
    yc = float(camera_params[3])

    #alamkin - writing images to files first to ensure images are correctly reprojected
    #cyl_images = [project_to_cyl(x, s, xc, yc) for x in images]
    for i, x in enumerate(images):
        project_to_cyl(x, s, xc, yc, img_fns[i]) 

    #alamkin - align each per of images and blend with mask
    # (what should be initial displacement?)
    initial_d = np.array([0,0])
    # using default from hw3
    steps = 5  
    # all image dimensions should be equal
    '''
    levels = 4 #int(np.floor(np.log2(min(cyl_images[0].shape[1], cyl_images[0].shape[0]))))
    for i in range(0,len(cyl_images),2):
          disp = pyramid_lucas_kanade(cyl_images[i], cyl_images[i+1], initial_d), levels, steps)
          blend_with_mask(cyl_images[i], cyl_images[i+1], mask):
    '''


