"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""

import argparse
import json
import os
import glob
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)

    if show:
        show_image(img)

    return img


def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    crops = img[1:34, 1:24]
    cv2.imshow('crops', crops)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    enrollment()

    detection()

    recognition()

    raise NotImplementedError


def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.

    raise NotImplementedError


def detection():
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    raise NotImplementedError


def recognition():
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.

    raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    characters = []
    test_characters = []

    all_character_imgs = glob.glob(args.character_folder_path + "/*")
    test_character_imgs = glob.glob(args.test_img)

    for each_character in all_character_imgs:
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=True)])

    character_name = "{}".format(os.path.split(test_character_imgs[0])[-1].split('.')[0])
    test_characters.append([character_name, read_image(test_character_imgs[0], show=False)])

    char_Enroll = []

    x = characters[0]

    #Row by Row
    row_total = []
    for x in characters:
        x_Range = []
        tmp = x[1]
        white_Pix = np.sum(tmp[0])
        for i in range(len(x[1])):
            if 1000 < white_Pix - np.sum(tmp[i]) < 20000:
                x_Range.append(i)
        row_total.append(x_Range)

    print(row_total)
    #Column by Column
    column_total = []

    tmp = x[1]
    for i in range(len(tmp[0])):
        tmp_Y = []
        for y in tmp:
            tmp_Y.append(y[i])
        column_total.append(tmp_Y)

    white_Pix = np.sum(column_total[0])

    final_Col = []
    for i in range(len(column_total)):
        if 200 < white_Pix - np.sum(column_total[i]) < 5000:
            final_Col.append(i)

    print(final_Col)
    # test_img = read_image(args.test_img)
    #
    # results = ocr(test_img, characters)
    #
    # save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
