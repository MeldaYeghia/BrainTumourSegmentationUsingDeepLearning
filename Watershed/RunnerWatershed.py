from Watershed.WatershedSegmentation import Segmentation
from time import time
from Utils.utils import *
from Utils.Metrics import *
import Watershed.WatershedSegmentation as WS

"""
Run Watershed algorithm on the entire test set (55 samples/220 3D scans) 
"""


def main():
    """The main function to run Watershed Algorithm"""
    # path of the folder to save the output
    pathofSegsFolder = "../ResultingSegmentations/Watershed/TestSet"

    # path of the test images
    pathsOfImgs = "../DeepLearningModel/dataset/Test_Full/testOneSample.txt"  # labelsNames.txt

    # path of the BraTS 2015 training dataset folder
    rootPath="D:/DATASET/BRATS2015_Training"

    # path of scores file
    pathOfScores = "../Plots/Watershed"
    pathofSegs, pathofGTs, pathofContours1, pathofContours2 = createWSFolders(pathofSegsFolder, pathOfScores)

    segObject = Segmentation()

    # get the names of the utilized test images
    names = getLabelNamesFrom(pathsOfImgs)
    namesIndex = 0
    # get the corresponding scans folder names given the names of images
    listOfTumourTypes, listOfPatients = getListOfScans(names)
    # get the corresponding scan paths of the images in the test set.
    scansPaths = fetchParticularPatientsDataPaths(rootPath, listOfTumourTypes, listOfPatients)
    start = time()
    Dices = []
    Precisions = []
    Recalls = []

    AllTPs = AllFPs = AllFNs = 0

    for indx, scanVal in enumerate(scansPaths):
        print("patient index: {} ".format(indx))
        seg = scanVal[4]  # ground truth
        scanVal = scanVal[0:4]  # 4 scan modalities (FLAIR, T1, T1c, T2)
        for scanType in scanVal:  # process all the slices for all 4 MRI modalities in the sample
            modalityName = getScanModalityName(scanType)
            # process and segment each scan in a sample and get the scores
            returnedNamesIndex, dice, precision, recall, TPs, FPs, TNs, FNs = segObject.startProcessing(scanType, seg,
                                                                                                        pathofSegs,
                                                                                                        pathofGTs,
                                                                                                        pathofContours1,
                                                                                                        pathofContours2,
                                                                                                        modalityName,
                                                                                                        names,
                                                                                                        namesIndex)
            namesIndex = returnedNamesIndex
            Dices.append(dice)
            Precisions.append(precision)
            Recalls.append(recall)
            print("Dice: {}, Precision: {}, Recall: {}".format(dice, precision, recall))
            AllTPs += TPs
            AllFPs += FPs
            AllFNs += FNs

    # calculates the average scores
    avgDice = sum(Dices) / float(len(Dices))
    avgPrecision = sum(Precisions) / float(len(Precisions))
    avgRecall = sum(Recalls) / float(len(Recalls))

    print("\nAverage Dice: {}, Average Precision: {}, Average Recall: {}".format(avgDice, avgPrecision, avgRecall))
    end = time()
    print("Time taken to segment all the scans in the test set: %f seconds" % (end - start))

    # Write all the generated scores into files
    writeScoreFile('{}/Dice.txt'.format(pathOfScores), Dices)
    writeScoreFile('{}/Precision.txt'.format(pathOfScores), Precisions)
    writeScoreFile('{}/Recall.txt'.format(pathOfScores), Recalls)
    writeScoreFileSingleLine('{}/InterimDices.txt'.format(pathOfScores), WS.InterimDices)
    writeScoreFileSingleLine('{}/CorrespondingSegNames.txt'.format(pathOfScores), WS.SegNames)


if __name__ == "__main__":
    main()
