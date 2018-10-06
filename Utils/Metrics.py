import numpy as np


def calculateDiceSimilarity(TP, FP, FN, noScore=1.0):
    """Calculates the dice similarity coefficient given the basic cardinalities. Returns the dice or 1 if TP+FP+FN==0 ."""
    if TP + FP + FN == 0:
        return noScore
    dice = (2 * TP) / ((2 * TP) + FP + FN)
    return dice


def calculateCardinalities(seg, gt):
    """Calculates and returns the basic cardinalities (TP, FP, TN, FN) of the automatic segmentation and the corresponding ground truth segmentation."""
    if seg.shape != gt.shape:
        raise ValueError("Seg and GT must have the same shape.")

    TP = np.logical_and(seg, gt)
    TN = np.logical_and(1 - seg, 1 - gt)
    FP = np.logical_and(seg, 1 - gt)
    FN = np.logical_and(1 - seg, gt)

    return TP.sum(), TN.sum(), FP.sum(), FN.sum()


def calculateRecall(TP, FN, noScore=0.0):
    """Calculates the recall given the basic cardinalities. Returns the calculated recall or 0 if TP+FN==0 ."""
    if TP + FN == 0:
        return noScore
    recall = TP / (TP + FN)
    return recall


def calculatePrecision(TP, FP, noScore=0.0):
    """Calculates the precision given the basic cardinalities. Returns the precision or 0 if TP+FP==0 ."""
    if (TP + FP) == 0:
        return noScore
    precision = TP / (TP + FP)
    return precision
