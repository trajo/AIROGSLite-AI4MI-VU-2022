import cv2
import numpy as np
import math


def get_retinal_image_diameter_as_horizontal_segment(raw_jpg: np.ndarray, threshold: int):
    """
    returns the bounds in pixel indices of the image

    Parameters
    ----------
    raw_jpg: numpy.ndarray
        image as found in dataset

    Returns
    -------
    top: int
        top bound
    bottom: int
        bottom bound
    left: int
        left bound
    right: int
        right bound
    threshold: int
        threshold value used for the cutoff
    """
    hor = np.max(raw_jpg, axis=(0, 2))
    horbounds = np.where(hor > threshold)[0]

    ver = np.max(raw_jpg, axis=(1, 2))
    verbounds = np.where(ver > threshold)[0]
    
    return verbounds[0], verbounds[-1], horbounds[0], horbounds[-1]


def get_square_retinal_img(raw_jpg: np.ndarray, top: int, bottom: int, left: int, right: int):
    """
    returns the square around the retina

    Parameters
    ----------
    raw_jpg: numpy.ndarray
        image as found in dataset
    top: int
        top bound
    bottom: int
        bottom bound
    left: int
        left bound
    right: int
        right bound

    Returns
    -------
    square: numpy.ndarray
        the new square image
    cutting: tuple
        how much was cut from the top, bottom, left, right
    padding: tuple
        how much was padded at top, bottom, left, right after the cutting has been performed
    """
    retina = raw_jpg[top:bottom+1, left:right+1]
    remtop = top
    rembottom = raw_jpg.shape[0] - (bottom+1)
    remleft = left
    remright = raw_jpg.shape[1] - (right+1)

    diff = retina.shape[1] - retina.shape[0]
    addtop = addbottom = addleft = addright = 0

    if diff == 0:
        return retina, (remtop, rembottom, remleft, remright), (addtop, addbottom, addleft, addright)
    elif diff > 0:
        addtop += diff // 2
        addbottom += diff - (diff // 2)
    else:
        diff = - diff
        addleft += diff // 2
        addright += diff - (diff // 2)

    square = cv2.copyMakeBorder(
        retina,
        top=addtop,
        bottom=addbottom,
        left=addleft,
        right=addright,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    return square, (remtop, rembottom, remleft, remright), (addtop, addbottom, addleft, addright)


def resize_square(img: np.ndarray, side: int):
    """
    resizes sqaure image to fixed resolution

    Parameters
    ----------
    img: numpy.ndarray
        the original square image
    side: int
        number of pixels per side

    Returns
    -------
    square_resized: numpy.ndarray
        resized square image
    """
    return cv2.resize(img, (side, side), interpolation=cv2.INTER_LINEAR)


def square_resize(raw_jpg: np.ndarray, side: int, threshold: int):
    """
    cuts out square around retina and resizes image to fixed resolution

    Parameters
    ----------
    raw_jpg: numpy.ndarray
        image as found in dataset
    side: int
        number of pixels per side
    threshold: int
        threshold value used for determining how much can be cut from the image edges

    Returns
    -------
    square_resized: numpy.ndarray
        resized square image
    scaling_factor: float
        multiply new image size by this factor to get original resolution
    cutting: numpy.ndarray
        resized square image
    padding: numpy.ndarray
        resized square image
    """
    top, bottom, left, right = get_retinal_image_diameter_as_horizontal_segment(raw_jpg, threshold)
    retinal_img_sq, cutting, padding = get_square_retinal_img(raw_jpg, top, bottom, left, right)
    assert retinal_img_sq.shape[0] == retinal_img_sq.shape[1]
    new_img = resize_square(retinal_img_sq, side)
    assert side == new_img.shape[0]
    assert side == new_img.shape[1]
    scaling_factor = retinal_img_sq.shape[0] / side

    # consistency test / sanity check: checking that the original resolution can be reconstructed
    added_x = -(padding[0] + padding[1]) + (cutting[0] + cutting[1])
    added_y = -(padding[2] + padding[3]) + (cutting[2] + cutting[3])
    rec_res_x, rec_res_y = round((side * scaling_factor) + added_x), round((side * scaling_factor) + added_y)
    assert rec_res_x == raw_jpg.shape[0]
    assert rec_res_y == raw_jpg.shape[1]

    return new_img, scaling_factor, cutting, padding


def make_square(raw_jpg: np.ndarray, threshold: int):
    """
    cuts out square around retina and resizes image to fix resolution

    Parameters
    ----------
    raw_jpg: numpy.ndarray
        image as found in dataset
    side: int
        number of pixels per side
    threshold: int
        threshold value used for determining how much can be cut from the image edges

    Returns
    -------
    square_resized: numpy.ndarray
        resized square image
    """
    top, bottom, left, right = get_retinal_image_diameter_as_horizontal_segment(raw_jpg, threshold)
    return get_square_retinal_img(raw_jpg, top, bottom, left, right)


def crop_od(original_img, odc_x, odc_y, sidelength):
    '''
    will in reality return square of size sidelength+1 
    '''
    radius = int(sidelength/2)
    
    top = max(0, odc_y - radius)
    bottom = min(original_img.shape[0] - 1, odc_y + radius)

    left = max(0, odc_x - radius)
    right = min(original_img.shape[1] - 1, odc_x + radius)
    
    return original_img[top:bottom+1, left:right+1]
