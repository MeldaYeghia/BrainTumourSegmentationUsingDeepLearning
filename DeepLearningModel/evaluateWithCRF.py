"""
Evaluation script for the DeepLab-ResNet network with CRF post-processing on the extracted validation and testing sets
   of the BraTS 2015 dataset.

This script evaluates the model using 52800 test images.

The code utilizes part of the evaluation script from DeepLabv2 ResNet Tensorflow library, by V. Nekrasov.
V. Nekrasov, “Tensorflow DeepLabv2 ResNet.” [Online]. Available: https://github.com/DrSleep/tensorflow-deeplab-resnet. [Accessed: 18-Jun-2018].

"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from Utils.Metrics import *
from Utils.utils import *
from DeepLearningModel.deeplab_resnet import DeepLabResNetModel, ImageReader
from DeepLearningModel.deeplab_resnet.utils import dense_crf

# path of the validation/testing data
DATA_DIRECTORY = "./TestingData"
# path of the text file containing the validation/testing data paths
DATA_LIST_PATH = './dataset/Test_Full/test.txt'

IGNORE_LABEL = 2
NUM_CLASSES = 2
NUM_STEPS = 52800  # Number of images in the test set. To be changed when different set is used such as the validation set, or subset of the test set
# path of the trained model
RESTORE_FROM = './model/model.ckpt-220000'

Num_Slices = 240  # number of slices in the MRI scan on the coronal view
# path of the text file containing the images names
labelNamesPath = "./dataset/Test_Full/labelsNames.txt"
# path of the resulting segmentations
pathOfSegsFolder = "../ResultingSegmentations/ResNet+CRF/TestSet"

# path of a file to save additional parameters at the end of evaluation
savingOutput = 'SavedParametersFromTestingCRF.txt'

# path of the folder in the Plots directory in which the results will be saved
scoresFolderName = 'ResNetCRF'

savingDiceScoresPath = '../Plots/{}/Dice.txt'.format(scoresFolderName)
savingPrecisionScoresPath = '../Plots/{}/Precision.txt'.format(scoresFolderName)
savingRecallScoresPath = '../Plots/{}/Recall.txt'.format(scoresFolderName)
savingDicesOfStepsPath = '../Plots/{}/DicesOfSteps.txt'.format(scoresFolderName)
savingNamesOfStepsPath = '../Plots/{}/NamesOfSteps.txt'.format(scoresFolderName)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the BRATS 2015 dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    return parser.parse_args()


def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    """Create the model and start the evaluation process."""
    createFolder("../Plots/{}".format(scoresFolderName))

    DicesOfSteps = []
    NamesOfSteps = []
    pathofSegs = "{}/Segs".format(pathOfSegsFolder)
    pathofGTs = "{}/GTs".format(pathOfSegsFolder)
    pathofContours = "{}/Contours".format(pathOfSegsFolder)

    if not os.path.exists(pathOfSegsFolder):
        os.makedirs(pathofSegs)
        os.makedirs(pathofGTs)
        os.makedirs(pathofContours)

    lblNames = getLabelNamesFrom(labelNamesPath)

    args = get_arguments()

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            None,  # No defined input size.
            False,  # No random scale.
            False,  # No random mirror.
            args.ignore_label,
            coord)
        image, label = reader.image, reader.label
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)  # Add one batch dimension.

    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3, ])
    img = tf.cast(image_batch, dtype=tf.uint8)
    # CRF unary potential
    unaryCRFInp = tf.nn.softmax(raw_output)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    TPs = TNs = FPs = FNs = 0
    AllTPs = AllTNs = AllFPs = AllFNs = 0
    Precisions = []
    Dices = []
    Recalls = []

    # Measure time
    MaxScanDur = 0
    MinScanDur = 100000000
    MaxSampleDur = 0
    MinSampleDur = 1000000000

    startTime = time.time()
    # Iterate over training steps.
    sampleStartTime = time.time()
    for step, lblName in zip(range(args.num_steps), lblNames):
        scanStartTime = time.time()
        unaryPreds, gt, image = sess.run([unaryCRFInp, label_batch, img])
        gt = gt[0, :, :, 0]  # get the image info as a numpy array from the tensor

        # set the CRF pairwise potential's parameters (2 of them searched as described in the report).
        posGaussian = 1
        compatGaussian = 1
        posBilateral = 2
        compatBilateral = 3
        rgbBilateral = 8
        # apply crf-inference post-processing
        preds = dense_crf(unaryPreds, posGaussian, compatGaussian, posBilateral, compatBilateral, rgbBilateral, image)
        preds = np.argmax(preds, axis=3)
        preds = preds[0, :, :]
        image = image[0, :, :, 0]

        uniqueImageName = '{}_{}'.format(lblName, step)
        NamesOfSteps.append(uniqueImageName)
        method = "DeepLab"

        # save the generated segmentation, ground truth and the contoured image with both
        matplotlib.image.imsave("{}/{}.png".format(pathofSegs, uniqueImageName), preds, cmap='gray')
        matplotlib.image.imsave("{}/{}.png".format(pathofGTs, lblName), gt, cmap='gray')
        saveContouredSample(image, preds, gt, uniqueImageName, pathofContours, method)

        # calculate 4 cardinalities
        TP, TN, FP, FN = calculateCardinalities(preds, gt)

        # find the maximum or minumum time that a slice takes to be segmented and saved
        scanDuration = time.time() - scanStartTime
        if scanDuration > MaxScanDur:
            MaxScanDur = scanDuration
        elif scanDuration < MinScanDur:
            MinScanDur = scanDuration

        print("image slice: ", step)
        print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
        dice = calculateDiceSimilarity(TP, FP, FN)
        DicesOfSteps.append(dice)

        TPs += TP
        TNs += TN
        FPs += FP
        FNs += FN

        if step != 0 and ((
                                  step + 1) % Num_Slices == 0):  # 1 scan is completed with its 240 slices, calculate dice, precision and recall

            diceScore = calculateDiceSimilarity(TPs, FPs, FNs)
            precisionScore = calculatePrecision(TPs, FPs)
            recallScore = calculateRecall(TPs, FNs)
            print("Dice Score: {}\nPrecision: {}\nRecall: {}".format(diceScore, precisionScore, recallScore))
            AllTPs += TPs
            AllFNs += FNs
            AllFPs += FPs
            AllTNs += TNs
            TPs = TNs = FPs = FNs = 0

            sampleDuration = time.time() - sampleStartTime

            # find the maximum or minumum time that 1 scan takes to be segmented and saved
            if sampleDuration > MaxSampleDur:
                MaxSampleDur = sampleDuration
            elif sampleDuration < MinSampleDur:
                MinSampleDur = sampleDuration

            Precisions.append(precisionScore)
            Recalls.append(recallScore)
            Dices.append(diceScore)
            sampleStartTime = time.time()

    averageDices = sum(Dices) / float(len(Dices))
    print("Average dice score for all the validation set: {}".format(averageDices))
    avgPrecisionScore = sum(Precisions) / float(len(Precisions))
    print("Average precision for all the validation set: {}".format(avgPrecisionScore))
    avgRecall = sum(Recalls) / float(len(Recalls))
    print("Average recall for all the validation set: {}".format(avgRecall))

    endTime = time.time()
    print("Time taken to evaluate the current set: {} seconds".format(endTime - startTime))
    print(
        "Minimum time taken to segment and evaluate one scan: {} seconds.\nMaximum time taken to segment and evaluate one scan: {} seconds.".format(
            MinScanDur, MaxScanDur))
    print(
        "Minimum time taken to segment and evaluate one sample with 4 scan modalities: {} seconds.\nMaximum time taken to segment and evaluate one sample with 4 scan modalities: {} seconds.".format(
            MinSampleDur, MaxSampleDur))

    # save the results in text files to plot them using Utils/PlotResults
    with open(savingDiceScoresPath, 'w') as f:
        print(Dices, file=f)

    with open(savingPrecisionScoresPath, 'w') as f:
        print(Precisions, file=f)

    with open(savingRecallScoresPath, 'w') as f:
        print(Recalls, file=f)

    with open(savingDicesOfStepsPath, 'w') as f:
        for obj in DicesOfSteps:
            f.write("%s\n" % obj)

    with open(savingNamesOfStepsPath, 'w') as f:
        for str in NamesOfSteps:
            f.write("%s\n" % str)

    savingParams = []
    savingParams.append(endTime - startTime)
    savingParams.append(MinScanDur)
    savingParams.append(MaxScanDur)
    savingParams.append(MinSampleDur)
    savingParams.append(MaxSampleDur)

    # save the required additional parameters
    with open(savingOutput, 'w') as f:
        print(savingParams, file=f)

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
