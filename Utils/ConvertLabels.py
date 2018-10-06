from PIL import Image
from tqdm import tqdm
import numpy as np
import os
from time import time

"""
This file encodes the labels and prepares the masks utilizing scripts from:
Reference:
V. Nekrasov, “Preparing Masks.” [Online]. Available: https://gist.github.com/DrSleep/4bce37254c5900545e6b65f6a0858b9c. [Accessed: 18-Jun-2018].
"""

# the colour palette used for generating masks
palette = {(0, 0, 0): 0,
           (255, 255, 255): 1
           }


def main():
    # path of the folder containing the extracted 2D images
    rootPath = 'D:/DeepModelFullData'

    # generate masked labels for the training, validation and testing sets
    generateMaskedLabels(rootPath, "TrainingData_Full")
    generateMaskedLabels(rootPath, "ValidationData_Full")
    generateMaskedLabels(rootPath, "TestingData_Full")

    # create new text files in the same directory having the masked labels paths to feed the deep learning algorithm with the masked images.
    generateNewTextFiles(rootPath, "TrainingData_Full")
    generateNewTextFiles(rootPath, "ValidationData_Full")
    generateNewTextFiles(rootPath, "TestingData_Full")


def convert_from_color_segmentation(arr_3d):
    """Converts the label into a mask using the output classes 0,1 mapped from the palette. This is taken from:
    Reference:
    V. Nekrasov, “Preparing Masks.” [Online]. Available: https://gist.github.com/DrSleep/4bce37254c5900545e6b65f6a0858b9c. [Accessed: 18-Jun-2018].
    """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def generateMaskedLabels(rootPath, folderPath):
    """Generates masked labels and saves them in a separate folder called y_tf in the same directory. """

    print("Converting the labels of {}".format(folderPath))
    new_label_dir = "{}/{}/y_tf/".format(rootPath, folderPath)
    label_dir = "{}/{}/y/".format(rootPath, folderPath)
    label_files = os.listdir(label_dir)

    start = time()

    for l_f in tqdm(label_files):
        arr = np.array(Image.open(label_dir + l_f))
        arr = arr[:, :, :3]
        arr_2d = convert_from_color_segmentation(arr)
        Image.fromarray(arr_2d).save(new_label_dir + l_f)

    end = time()
    print("Time taken to convert the {} labels: {} seconds.".format(folderPath, (end - start)))
    print()


def generateNewTextFiles(rootPath, folderPath):
    """This method creates a new text file, having the previous file name added by number 2. The files are created by updating the paths in the provided text file to point to the masked labels and provide it to the deep learning algorithm to allow the successful reading of images."""
    if "TrainingData" in folderPath:
        fileName = "train"
    elif "ValidationData" in folderPath:
        fileName = "val"
    else:
        fileName = "test"

    path = "{}/{}/{}".format(rootPath, folderPath, fileName)
    existedPath = "{}.txt".format(path)
    newPath = "{}2.txt".format(path)

    with open(existedPath, 'r') as file:
        f = file.read()

    f = f.replace('/y/', '/y_tf/')

    with open(newPath, 'w') as file:
        file.write(f)


if __name__ == "__main__":
    main()
