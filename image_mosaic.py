import numpy as np
import imageio
import cv2 as CV
from scipy.ndimage.filters import convolve
import os
from PIL import Image

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

    # return the displacement
    return displacement


def iterative_lucas_kanade(H, I, steps):
    # Run the basic Lucas Kanade algorithm in a loop `steps` times.
    # Start with an initial displacement of 0 and accumulate displacements.
    disp = np.zeros((2,), np.float32)
    for i in range(steps):
        # Translate the H image by the current displacement (using the translate function above)
        trans_H = translate(H, disp)
        # run Lucas Kanade and update the displacement estimate
        disp += lucas_kanade(trans_H, I)
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

#
# def reproject(img):
#
#     on_cyl = img / (np.sqrt(img[:, :, 0]**2 + img[:,:, 2]**2))
#

def project_to_cyl(image, s, k1, k2):

    #alamkin - for some reason, image DSC00109.png has a shape = (480, 640, 4) 
    #and needs to be downsized
    new_image = np.zeros((image.shape[0], image.shape[1],3))
    if image.shape[2] == 4:
        for i in range(len(image)):
            for j in range(len(image[i])):
                new_image[i][j] = np.array(image[i][j][:3])
        image = new_image        

    xc = image.shape[1] / 2
    yc = image.shape[0] / 2

    #alamkin - what's this for? 
    pts = np.mgrid[:image.shape[0], :image.shape[1]
          ].transpose(1, 2, 0).astype(np.float32)

    # get cylindrical coordinates
    x_cyl = pts[:, :, 1]
    y_cyl = pts[:, :, 0]

    # convert from cylindrical coordinates to h/theta
    theta = (x_cyl - xc) / s
    h = (y_cyl - yc) / s

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

    # set original coordinates to warped coordinates
    pts[:, :, 1] = x
    pts[:, :, 0] = y

    return bilinear_interp(image, pts)


def stitch(proj_imgs, final_disps):
    final_heigth = int(np.ceil(proj_imgs[0].shape[0] + (np.max(final_disps, axis = 0)[1] - np.min(final_disps, axis = 0)[1])))
    final = None
    for idx, img in enumerate(proj_imgs):

        if idx == 0:
            final = img

        width = int(np.ceil(final.shape[1] + abs(final_disps[idx][0])))

        mask = np.zeros([final_heigth, width, 3])
        cur = mask.copy()
        next = mask.copy()

        cur[:final.shape[0], :final.shape[1], :] = final
        next_image = proj_imgs[(idx + 1) % 15]

        disp = final_disps[idx]

        next[-next_image.shape[0]:, -next_image.shape[1]:, :] = next_image
        mask[np.where(next > 0)] = 1
        final = blend_with_mask(cur, next, mask)

    return final

    
if __name__ == "__main__":
    f = open('test_data/test_files.txt', 'r')
    lines = f.readlines()
    f.close()
    images = []
    img_fns = []
    in_disps = []
    for line in lines:
        words = line.strip().split(' ')
        file_name = words[0]
        images.append(load_image(file_name))
        img_fns.append(file_name)
        in_disps.append([int(x) for x in words[1:]])

    f = open('test_data/test_params.txt')
    camera_params = f.readline().strip().split(' ')
    f.close()

    s = int(camera_params[0])
    k1 = float(camera_params[2])
    k2 = float(camera_params[3])

    
    proj_imgs = []
    #alamkin - writing images to files first to ensure images are correctly reprojected
    #cyl_images = [project_to_cyl(x, s, xc, yc) for x in images]
    for i in range(len(images)):
        proj_img = project_to_cyl(images[i], s, k1, k2)
        proj_imgs.append(proj_img)
        img_name = img_fns[i].split('.')
        imageio.imwrite('{}_cyl.{}'.format(img_name[0], img_name[1]), proj_img.astype(np.uint8))


    #alamkin - align each per of images 
    '''
    proj_images_fns = []
    proj_imgs = []
    
    for fn in os.listdir("./test_data"):
        if fn.endswith(".png_cyl.png"):
            proj_images_fns.append(fn)

    proj_images_fns = sorted(proj_images_fns)

    for fn in proj_images_fns:
        proj_imgs.append(load_image("./test_data/" + fn))
    '''

    steps = 5
    levels = 4 
    final_disps = []

    for i in range(len(proj_imgs)):
        final_disps.append(pyramid_lucas_kanade(proj_imgs[i], proj_imgs[(i+1)%15], in_disps[i], levels, steps))

    #printing displacements for review
    #for i in range(len(img_fns)):
    #    print(img_fns[i] + ': ' + str(in_disps[i]) + ' --> ' + str(final_disps[i]))
    
    #alamkin - stich photos together
    mosiac_after_blend = stitch(proj_imgs, final_disps)

    #writing intermediate image to file for presentation
    imageio.imwrite('mosiac_after_blending.png', mosiac_after_blend.astype(np.uint8))

    #alamkin - blend resulting photo
    #mosiac = blend_with_mask()
    
    #writing final image to file
    #imageio.imwrite('final_mosiac.png'.format(img_fns[i]), mosiac.astype(np.uint8))


