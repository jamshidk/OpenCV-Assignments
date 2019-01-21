import numpy as np
import cv2

from ps1 import *


def main():

    # TODO: Insert your image file paths here:
    img1_filename = "\ps01\ps1-1-a-1.png"
    img2_filename = "\ps01\ps1-1-a-2.png"

    # # 1a
    img1 = cv2.imread('ps1-1-a-1.png')
    img2 = cv2.imread('ps1-1-a-2.png')

    assert 100 < img1.shape[0] <= 512, "Check your image 1 dimensions"
    assert 100 < img1.shape[1] <= 512, "Check your image 1 dimensions"
    assert 100 < img2.shape[0] <= 512, "Check your image 2 dimensions"
    assert 100 < img2.shape[1] <= 512, "Check your image 2 dimensions"
    assert img1.shape[1] > img1.shape[0], "Image 1 should be a wide image"
    assert img2.shape[0] > img2.shape[1], "Image 2 should be a tall image"


    cv2.imwrite('output/ps1-1-a-1.png', img1)
    cv2.imwrite('output/ps1-1-a-2.png', img2)


    # # 2 Color Planes

    # 2a

    swapped_green_blue_img = swap_green_blue(img1)
    cv2.imwrite('output/ps1-2-a-1.png', swapped_green_blue_img)

    # 2b
    img1_green = extract_green(img1)

    assert len(img1_green.shape) == 2, "The monochrome image must be a 2D array"
    cv2.imwrite('output/ps1-2-b-1.png', img1_green)


    # 2c
    img1_red = extract_red(img1)
    assert len(img1_red.shape) == 2, "The monochrome image must be a 2D array"
    cv2.imwrite('output/ps1-2-c-1.png', img1_red)

    # # 3 Replacement of Pixels

    # 3a

    # TODO: Choose the monochrome image for img1.
    mono1 = img1_green

    mono2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    replaced_img = copy_paste_middle(mono1, mono2, (100, 100))
    cv2.imwrite('output/ps1-3-a-1.png', replaced_img)
    cv2.imshow('normalized_img',replaced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # 4 Arithmetic and Geometric operations

    # # 4a
    #min_green, max_green, mean_green, stddev_green = image_stats(img1_green)

    print("The min pixel value of img1_green is", image_stats(img1_green)[0])
    print("The max pixel value of img1_green is", image_stats(img1_green)[1])
    print("The mean pixel value of img1_green is", image_stats(img1_green)[2])
    print("The std dev of img1_green is", image_stats(img1_green)[3])

    # # 4b
    normalized_img = center_and_normalize(img1_green, 10)
    cv2.imshow('normalized_img',center_and_normalize(img1_green, 10))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('output/ps1-4-b-1.png', normalized_img)

    # # 4c
    shift_green = shift_image_left(img1_green, 2)

    cv2.imwrite('output/ps1-4-c-1.png', shift_green)

    #cv2.imshow('normalized_img',shift_green)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    # # 4d
    diff_green = difference_image(img1_green, shift_green)

    cv2.imwrite('output/ps1-4-d-1.png', diff_green)
    cv2.imshow('normalized_img',diff_green)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # # 5 Noise

    # TODO: Choose a sigma value:
    sigma = 10

    # # 5a
    channel = 1
    noisy_green = add_noise(img1, channel, sigma)
    cv2.imwrite('output/ps1-5-a-1.png', noisy_green)
    cv2.imshow('noise',noisy_green)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # # 5b
    channel = 0
    noisy_blue = add_noise(img2, channel, sigma)
    cv2.imwrite('output/ps1-5-b-1.png', noisy_blue)
    cv2.imshow('noise',noisy_blue)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
