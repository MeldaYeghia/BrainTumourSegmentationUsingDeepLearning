import os, glob, itertools
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib
from scipy import stats
import SimpleITK as sitk


def writeScoreFile(path, scores):
    """Writes a given list (scores) into a txt file in the specified path (path)."""
    with open(path, 'w') as f:
        print(scores, file=f)


def writeScoreFileSingleLine(path, scores):
    """Writes a given list (scores) into a txt file in the specified path (path), writing each element in the list in a new line."""
    with open(path, 'w') as f:
        for str in scores:
            f.write("%s\n" % str)


def createFolder(path):
    """Creates a folder given a path."""
    if not os.path.exists(path):
        os.makedirs(path)


def createWSFolders(pathOfSegsFolder, pathOfScores):
    """Creates the required folders for the output of the Watershed algorithm given the required paths."""
    pathofSegs = "{}/Segs".format(pathOfSegsFolder)
    pathofGTs = "{}/GTs".format(pathOfSegsFolder)
    pathofContours1 = "{}/Contours1".format(pathOfSegsFolder)
    pathofContours2 = "{}/Contours2".format(pathOfSegsFolder)

    if not os.path.exists(pathOfSegsFolder):
        os.makedirs(pathofSegs)
        os.makedirs(pathofGTs)
        os.makedirs(pathofContours1)
        os.makedirs(pathofContours2)

    createFolder(pathOfScores)
    return pathofSegs, pathofGTs, pathofContours1, pathofContours2


def saveContouredSample(img, seg, gt, name, path, approach):
    """Contours the segmentation (in blue) and ground truth (in red) on a given image and saves it given the path. """
    npImg = np.array(img)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    if "Watershed" in approach:
        origin = "lower"

    elif "DeepLab" in approach:
        origin = None

    axes[0].imshow(npImg, cmap=plt.cm.gray, origin=origin)
    axes[0].set_title('Red: Ground Truth\nBlue: {} approach'.format(approach))
    axes[0].contour(seg, [0.5], linewidths=1.2, colors='b')
    axes[0].contour(gt, [0.5], linewidths=1.2, colors='r')
    axes[0].axis('off')
    fig.delaxes(axes[1])
    plt.savefig("{}/{}.png".format(path, name))
    plt.close()
    # plt.show()


def readFiles(path):
    """Reads all the data from the specified path and returns a list of paths for HGG and LGG samples"""

    for filename in glob.glob(os.path.join(path, '*')):
        if "HGG" in filename:

            HGG_scans = []
            for HGG_folders in glob.glob(os.path.join(filename, '*')):
                patientList = []

                for HGG_subfile in glob.glob(os.path.join(HGG_folders, '*')):
                    # get only.mha files
                    for HGG_sub_sub_file in glob.glob(os.path.join(HGG_subfile, '*.mha')):
                        patientList.append(HGG_sub_sub_file)

                HGG_scans.append(patientList)


        elif "LGG" in filename:
            LGG_scans = []
            for LGG_folders in glob.glob(os.path.join(filename, '*')):
                patientList = []

                for LGG_subfile in glob.glob(os.path.join(LGG_folders, '*')):
                    # get only.mha files
                    for LGG_sub_sub_file in glob.glob(os.path.join(LGG_subfile, '*.mha')):
                        patientList.append(LGG_sub_sub_file)

                LGG_scans.append(patientList)

    return HGG_scans, LGG_scans


def mergeLists(HGG, LGG):
    """Merges HGG and LGG lists"""
    HGG_LGG = []
    for item in itertools.chain(HGG, LGG):
        HGG_LGG.append(item)
    return HGG_LGG


def readImage(imagePath):
    """This method reads the image and returns it as an array, given its path."""
    image = sitk.ReadImage(imagePath)
    imgData = sitk.GetArrayFromImage(image)
    return imgData


def getOccupiedSlices(labelData):
    """This method returns a list containing the indexes of slices in which there exists a tumour segmentation label in the ground truth data, passed as a parameter. """

    occupied = []
    shape = labelData.shape

    for i in range(shape[1]):  # 240
        labelSlice = labelData[:, i, :]
        allZeros = not np.any(labelSlice)

        if not allZeros:
            occupied.append(i)
    return occupied


def create2Dimgs(pathOfData, X_set, y_set, pathOfSet, task, numOfSlices):
    """Extracts 2D images and their corresponding labels, names them according to its original path in the BraTS 2015, saves them and saves their paths to a text file."""
    pathOfData = "{}/{}".format(pathOfData, pathOfSet)
    listForTextFile = []
    listOfXPaths = []
    listOfYPaths = []
    listOfLbls = []

    if not os.path.exists(pathOfData):
        os.makedirs("{}/X".format(pathOfData))
        os.makedirs("{}/y".format(pathOfData))
        os.makedirs("{}/y_tf".format(pathOfData))

    for trainingSampleX, trainingSampleY in zip(X_set, y_set):

        labelData = readImage(trainingSampleY)

        isHGG = checkScanType(trainingSampleY)
        if isHGG == True:
            tumourType = "HGG"
        else:
            tumourType = "LGG"

        patientInfo = getPatientInfo(trainingSampleY)

        labelName = "{}_{}_seg_".format(tumourType, patientInfo)


        for trainingSampleType in trainingSampleX:
            imageData = readImage(trainingSampleType)
            scanModality = checkScanModality(trainingSampleType)
            imageName = "{}_{}_{}_".format(tumourType, patientInfo, scanModality)

            listForTextFile_part, listOfXPaths_part, listOfYPaths_part, listOfLbls_part = getImageSlices(imageData,
                                                                                                         labelData,
                                                                                                         imageName,
                                                                                                         labelName,
                                                                                                         pathOfData,
                                                                                                         numOfSlices)
            for val1, val2, val3, val4 in zip(listForTextFile_part, listOfXPaths_part, listOfYPaths_part,
                                              listOfLbls_part):
                listForTextFile.append(val1)
                listOfXPaths.append(val2)
                listOfYPaths.append(val3)
                listOfLbls.append(val4)
    #write the paths to files
    writeToFiles(pathOfData, listForTextFile, listOfXPaths, listOfYPaths, listOfLbls, task)


def getRandomlySelectedIndexes(tumourSlices, numberOfSlices):
    """Get random indexes but from non-tumour and tumour slices (beginning, middle, and end of tumours slices) ."""
    numOfSlices = len(tumourSlices)
    idxOfSplit = int(numOfSlices / 3)
    listOfSplitIdx = []
    for i in range(len(tumourSlices)):
        if i % idxOfSplit == 0:
            listOfSplitIdx.append(i)

    if numOfSlices >= 45:
        numOfInterimSlices = 15
    elif numOfSlices >= 30:
        numOfInterimSlices = 10
    elif numOfSlices >= 21:
        numOfInterimSlices = 7
    elif numOfSlices >= 15:
        numOfInterimSlices = 5
    elif numOfSlices >= 9:
        numOfInterimSlices = 3
    else:
        numOfInterimSlices = 1

    startingPt1 = tumourSlices[listOfSplitIdx[0]]
    startingPt2 = tumourSlices[listOfSplitIdx[1]]
    startingPt3 = tumourSlices[listOfSplitIdx[2]]
    lastIdx = tumourSlices[-1]

    randIndexes1 = random.sample(range(startingPt1, startingPt2),
                                 numOfInterimSlices)
    randIndexes2 = random.sample(range(startingPt2, startingPt3), numOfInterimSlices)
    randIndexes3 = random.sample(range(startingPt3, lastIdx + 1), numOfInterimSlices)

    obtainedIdxs = numOfInterimSlices * 3
    desiredNumSlices = numberOfSlices
    remainingNumSlices = desiredNumSlices - obtainedIdxs

    beg = np.arange(0, startingPt1)
    end = np.arange(lastIdx + 1, 240)
    remainingIdx = np.append(beg, end)
    randomIdxOframaining = random.sample(range(0, len(remainingIdx)), remainingNumSlices)

    actualRandomSliceIdx = []
    for val in randomIdxOframaining:
        actualRandomSliceIdx.append(remainingIdx[val])

    allSliceIndexes_p1 = np.append(randIndexes1, randIndexes2)
    allSliceIndexes_p2 = np.append(allSliceIndexes_p1, randIndexes3)
    allSliceIndexes_p3 = np.append(allSliceIndexes_p2, actualRandomSliceIdx)
    return allSliceIndexes_p3


def getImageSlices(imgData, lblData, currentImageName, currentLabelName, pathOfData, numOfSlices):
    """Extract slices of the image and its label, given the numberOfSlices"""

    if (numOfSlices == 240): #extract all slices of the scan
        indexes = np.arange(240)
    else:                   # extract less slices than 240, which is only used for initial experiments
        # get the tumour slices
        tumourSlices = getOccupiedSlices(lblData)
        #get systematically random indexes
        indexes = getRandomlySelectedIndexes(tumourSlices, numOfSlices)

    listForTextFile = []
    listOfXPaths = []
    listOfYPaths = []
    listOfLbls = []

    for i in indexes:
        lblSlice = getSliceAt(lblData, i)
        imgSlice = getSliceAt(imgData, i)

        lblSliceName = "{}{}".format(currentLabelName, i)
        imgSliceName = "{}{}".format(currentImageName, i)

        lblSlice = preprocessLabelSlice(lblSlice, None)
        imgSlice=preprocessImgs_DL(imgSlice)

        imgSlice = np.flipud(imgSlice)
        lblSlice = np.flipud(lblSlice)
        yPath = "/y/{}.png".format(lblSliceName)
        xPath = "/X/{}.jpg".format(imgSliceName)
        forTextFile = "{} {}".format(xPath, yPath)
        listForTextFile.append(forTextFile)
        listOfXPaths.append(xPath)
        listOfYPaths.append(yPath)
        listOfLbls.append(lblSliceName)
        #saves the images
        matplotlib.image.imsave("{}{}".format(pathOfData, yPath), lblSlice, cmap='gray')
        matplotlib.image.imsave("{}{}".format(pathOfData, xPath), imgSlice, cmap='gray')
    return listForTextFile, listOfXPaths, listOfYPaths, listOfLbls


def getLabelNamesFrom(lblNamesFile):
    "Reads a file and saves its contents into a list."
    f = open(lblNamesFile, 'r')
    listOfLblNames = []

    for line in f:
        lbl = line.strip("\n")

        listOfLblNames.append(lbl)

    return listOfLblNames


def writeToFiles(pathOfData, listForTextFile, listOfXPaths, listOfYPaths, listOfLbls, task):
    """Writes the paths of extracted images to files, to use them in the DL approach."""
    with open('{}/{}.txt'.format(pathOfData, task), 'w') as f:
        for str in listForTextFile:
            f.write("%s\n" % str)

    with open('{}/{}_X.txt'.format(pathOfData, task), 'w') as f:
        for str in listOfXPaths:
            f.write("%s\n" % str)

    with open('{}/{}_Y.txt'.format(pathOfData, task), 'w') as f:
        for str in listOfYPaths:
            f.write("%s\n" % str)

    with open('{}/labelsNames.txt'.format(pathOfData), 'w') as f:
        for str in listOfLbls:
            f.write("%s\n" % str)


def changePathsMasksToLabels(path, type):
    """Utilized to modify the paths that are saved to the text files while creating the 2D images, from the masks paths to the labels paths."""
    with open(path, 'r') as file:
        f = file.read()

    f = f.replace('/y_tf/', '/y/')

    path = path.replace(type, '{}Watershed'.format(type))

    with open(path, 'w') as file:
        file.write(f)
    return path


def getFolderName(name):
    """Builds the folder name of a sample, as it is in the BraTS dataset, given the name of the corresponding 2D image file. Returns the required folder names to read the data."""
    typeOfTumour = name[:3]
    lastIndexOf_ = name.rfind('_')
    rest = name[4:lastIndexOf_]
    lastIndex = rest.rfind('_')
    folderName = rest[:lastIndex]
    return typeOfTumour, folderName


def getListOfScans(names):
    """Gets the scan folder names in BraTS dataset given the names of the images."""
    listOfPatients = []
    listOfTumourTypes = []
    for name in names:
        typeOfTumour, folderName = getFolderName(name)
        if folderName not in listOfPatients:
            listOfPatients.append(folderName)
            listOfTumourTypes.append(typeOfTumour)

    return listOfTumourTypes, listOfPatients


def fetchParticularPatientsDataPaths(rootPath, listOfTumourTypes, listOfPatients):
    """Get the scan paths of the images in the test set given the folders they belong to in the BraTS dataset."""
    patientsList = []
    for tumourType, patient in zip(listOfTumourTypes, listOfPatients):
        patientPath = "{}/{}/{}".format(rootPath, tumourType, patient)

        patientList = []
        for patientScans in glob.glob(os.path.join(patientPath, '*')):

            # get only.mha files
            for scanType in glob.glob(os.path.join(patientScans, '*.mha')):
                patientList.append(scanType)

        patientsList.append(patientList)
    return patientsList


def getScanModalityName(scanType):
    """Returns the scan modality name contained in a string given the string."""
    if "Flair" in scanType:
        modalityName = "Flair"
    elif "T1c" in scanType:
        modalityName = "T1c"
    elif "T1" in scanType:
        modalityName = "T1"
    elif "T2" in scanType:
        modalityName = "T2"
    return modalityName


def normalizeImageSlice(imgSlice):
    """Saturates and normalizes the given image slice.
     The main ideas of the preprocessing and some parts of the following code is used from:
     Reference:
    “Images and words, Emmanuelle Gouillart’s blog.” [Online]. Available: http://emmanuelle.github.io/segmentation-of-3-d-tomography-images-with-python-and-scikit-image.html. [Accessed: 23-Mar-2018].
    """

    # get the 0.5% of the lightest and darkest grey intensities
    vmin, vmax = stats.scoreatpercentile(imgSlice, (0.5, 99.5))

    # if vmin and vmax are 0, which means the darkest and lightest 0.5% of values are the same, equal to 0
    if vmin == vmax:
        vmin = np.min(imgSlice)
        vmax = np.max(imgSlice)

    # saturate the image slice using vmin, vmax
    imgSlice = np.clip(imgSlice, vmin, vmax)
    # self.dispImg(imgSlice,"After saturation")

    # apply minmax normalization to rescale the intensity ranges of the image slice
    imgSlice = (imgSlice - vmin) / (vmax - vmin)

    return imgSlice


def preprocessLabelSlice(slice, bbox):
    """ This method preprocesses the labels as they consist of 5 classes (from 0 to 4). It sets all the values that are greater than 0 into 1, indicating the tumour area."""
    mask = slice > 0
    slice[mask] = 1
    if bbox != None:
        slice = slice[bbox]
    return slice


def preprocessImgs_DL(imgSlice):
    """Pre-processes 2D images for the deep learning algorithm (saturates and normalizes them only)."""
    allZeros = not np.any(imgSlice)
    # in case of black images ( all zeros) don't pre-process it
    if (not allZeros):
        imgSlice = normalizeImageSlice(imgSlice)

    return imgSlice


def getSliceAt(data, sliceIdx):
    """This method gets the slice at the index passed as a parameter."""
    slice = data[:, sliceIdx, :]
    return slice


def checkScanType(scanPath):
    """Checks whether the given path is for HGG or LGG scan. Returns a boolean value indicating if it is an HGG or not."""
    isHGG = True
    if "LGG" in scanPath:
        isHGG = False

    elif "HGG" in scanPath:
        isHGG = True

    return isHGG


def getPatientInfo(scanPath):
    """Returns the sample number as it is called in the BraTS 2015 dataset, from a given path."""
    patientNum = ""
    splitPath = scanPath.split('\\')
    for item in splitPath:
        if item.startswith("brats_"):
            patientNum = item
    return patientNum


def checkScanModality(scanPath):
    """Returns the scan modality name by checking if that string is contained in the given path. """
    modality = ""
    if "Flair" in scanPath:
        modality = "Flair"
    elif "T1c" in scanPath:
        modality = "T1c"
    elif "T1" in scanPath:
        modality = "T1"
    elif "T2" in scanPath:
        modality = "T2"
    return modality


def checkIfLabel(scanPath):
    """Checks whether a given scan path is the label path, and returns a boolean value indicating if it is label or not."""
    isLabel = False
    if "3more" in scanPath:
        isLabel = True
    else:
        isLabel = False
    return isLabel
