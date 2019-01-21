import math
import numpy as np
import scipy.ndimage as ndi
import cv2
import sys

# # Implement the functions below.


def extract_red(image):
    temp_image = np.copy(image)
    Red = temp_image[:,:,2]

    return Red


def extract_green(image):

    temp_image = np.copy(image)
    Green = temp_image[:,:,1]

    return Green



def extract_blue(image):
    temp_image = np.copy(image)
    Blue = temp_image[:,:,0]

    return Blue


def swap_green_blue(image):

    temp_image = np.copy(image)
    image[:,:,1]= extract_blue(temp_image)
    image[:,:,0]= extract_green(temp_image)


    return image


def copy_paste_middle(src, dst, shape):

    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:"""
    temp_img1 = np.copy(src)
    temp_img2 = np.copy(dst)
    height=np.size(temp_img1,0)
    width=np.size(temp_img1,1)
    min1_w=(width/2)-(shape[0]/2)
    max1_w=(width/2)+(shape[0]/2)
    min1_h=(height/2)-(shape[1]/2)
    max1_h=(height/2)+(shape[1]/2)
    cropped_src = temp_img1[int(min1_h):int(max1_h),int(min1_w):int(max1_w)]

    height=np.size(temp_img2,0)
    width=np.size(temp_img2,1)
    min2_w=(width/2)-(shape[0]/2)
    max2_w=(width/2)+(shape[0]/2)
    min2_h=(height/2)-(shape[1]/2)
    max2_h=(height/2)+(shape[1]/2)

    temp_img2[int(min2_h):int(max2_h),int(min2_w):int(max2_w)]= cropped_src


    #   Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

    # Note: Where 'middle' is ambiguous because of any difference in the oddness
    #   or evenness of the size of the copied region and the image size, the function
        #rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        #into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        #the src into the range [1:2,1:2] of the dst. """


    #    src (numpy.array): 2D array where the rectangular shape will be copied from.
    #    dst (numpy.array): 2D array where the rectangular shape will be copied to.
    #    shape (tuple): Tuple containing the height (int) and width (int) of the section to be

    return temp_img2



def image_stats(image):

    #
    temp_image = np.copy(image)
    #mono_color=extract_green(temp_image)
    temp_image = temp_image.astype(np.float64)

    tup=(temp_image.min(),temp_image.max(),temp_image.mean(),temp_image.std())

    return tup


def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:"""

    temp_image = np.copy(image)
    temp_image = temp_image.astype(np.float64)
    #mono_color=extract_green(temp_image)
    #mono_color2=extract_red(image)
    temp_image=((temp_image-temp_image.mean())/temp_image.std())*scale+ temp_image.mean()
    #temp_image = temp_image.astype(np.uint8)
    temp_image = np.uint8(temp_image)

    return temp_image


def shift_image_left(image, shift):

    #im = cv2.imread('image.jpg')
    temp_image = np.copy(image)
    bordersize=shift
    replicate = cv2.copyMakeBorder(image,0,0,0,bordersize,cv2.BORDER_REPLICATE)
    row, col= replicate.shape[:2]
    temp_image = replicate[:,shift:col]
    #    image (numpy.array): Input 2D image.
    #    shift (int): Displacement value representing the number of pixels to shift the input image.
    #        This parameter may be 0 representing zero displacement.

    return temp_image



def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255]"""
    temp_img1 = np.copy(img1)
    temp_img2 = np.copy(img2)
    temp_img1 = temp_img1.astype(np.float64)
    temp_img2 = temp_img2.astype(np.float64)
    #diff=cv2.subtract(temp_img1,temp_img2)
    diff=temp_img1-temp_img2
    diff=(diff-diff.min())*(255/(diff.max()-diff.min()))
    diff = diff.astype(np.uint8)
    #diff = np.uint8(diff)



    return diff



def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
     channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise."""
    #image = image.astype(np.float64)
    temp_image = np.copy(image)

    mono_color=temp_image[:,:,channel]
    mono_color = mono_color.astype(np.float64)
    filtered=ndi.gaussian_filter(mono_color,sigma)

    filtered=(filtered-filtered.min())*(255/(filtered.max()-filtered.min()))
    filtered = filtered.astype(np.uint8)
    #The returned array values must not be clipped or normalized and scaled. This means that
    #there could be values that are not in [0, 255].

    #Note: This function makes no defense against the creation
    #of out-of-range pixel values.  Consider converting the input image to
    #a float64 type before passing in an image.
    temp_image[:,:,channel]=filtered
    #temp_image=cv2.cvtColor(filtered,temp_image)


    return temp_image
