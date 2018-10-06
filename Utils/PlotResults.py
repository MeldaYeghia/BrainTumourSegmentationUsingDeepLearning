import matplotlib.pyplot as plt
import numpy as np


def readResults(fileName):
    """This function reads the text file passed as a parameter, which should be a list of the average scores of all the scans in the dataset. It returns the corresponding list object having the average scores."""
    with open(fileName) as f:
        stringList = f.read()

    processedString = stringList.replace("[", "").replace("]", "").replace(",", "")
    readList = [float(x) for x in processedString.split()]
    return readList


def finalBoxPlots(finalResults, metricName, finalResultsNames, finalResFolder):
    """Creates the final comparison box plots given the results, the metric and the folder in which the plot to be saved as an image, in the Plots folder."""
    plt.boxplot(finalResults, showmeans=True)
    plt.xlabel("Conducted Experiments")
    plt.xticks(np.arange(1, (len(finalResults) + 1)), finalResultsNames, rotation=-10)
    plt.ylabel(metricName)
    plt.yticks(np.arange(0.00, 1.10, step=0.10))
    plt.suptitle("Evaluating the segmentations using the {} metric".format(metricName))
    plt.savefig('../Plots/{}/FinalResults_{}.png'.format(finalResFolder, metricName), dpi=300, bbox_inches="tight")
    # plt.show()


def setMetricName(scoreShortName):
    """Returns the metric name based on the given argument, which is the first letter of the metric. """
    if scoreShortName == 'd':
        shortMetricName = "Dice"
        metricName = "Dice Similarity Coefficient"

    elif scoreShortName == 'p':
        shortMetricName = "Precision"
        metricName = shortMetricName
    else:
        shortMetricName = "Recall"
        metricName = shortMetricName
    return shortMetricName, metricName


def generateSingleBoxPlot(shortMetricName, metricName, datasetFName, method, xTickName):
    """Generates a single box plot given the results, the metric and saves the plot in the same folder, in Plots directory."""
    if "val" in datasetFName:
        datasetName = "Validation Set"
    elif "test" in datasetFName:
        datasetName = "Test Set"
    else:
        datasetName="Test Set"

    xAxisCaption = "Experiment: {}_{}".format(method, datasetName)
    yAxisCaption = metricName

    filePath = '../Plots/{}/{}.txt'.format(datasetFName, shortMetricName)
    figureName = "../Plots/{}/{}.png".format(datasetFName, shortMetricName)

    avgScore = readResults(filePath)
    singleAverage = sum(avgScore) / float(len(avgScore))
    singleAverage = round(singleAverage, 2)
    plotTitle = "{} {} Scores, \n with final average {} of {}".format(datasetName, metricName, shortMetricName,
                                                                      singleAverage)
    singleBoxPlot(avgScore, plotTitle, figureName, xAxisCaption, yAxisCaption, xTickName)


def getResultsFilePath(folderName, shortMetricName):
    """Creates the string path of the results text file, given its folder name and the metric name."""
    filePath = '../Plots/{}/{}.txt'.format(folderName, shortMetricName)
    return filePath


def generateFinalComparisonPlot(shortMetricName, metricName, finalResFolder):
    """Prepares all the required settings to plot the final comparison plot, including reading the results of all experiments given the folder names they are contained in, and plotting."""
    watershed = readResults(getResultsFilePath("test_WS", shortMetricName))
    preProcessed = readResults(getResultsFilePath("test_DL_Preprocessed", shortMetricName))
    morePreProcessed = readResults(getResultsFilePath("test_DL_MoreData", shortMetricName))
    morePreProcessedCRF = readResults(getResultsFilePath("test_DL_MoreData_CRF_3_2_8", shortMetricName))
    morePreProcessedMirroredScaled = readResults(getResultsFilePath("test_DL_MoreData_augment", shortMetricName))
    morePreProcessedMirroredScaledASPP = readResults(
        getResultsFilePath("test_DL_MoreData_augment_ASPP", shortMetricName))

    finalResultsNames = ["Watershed", "DL", "DL +Data", "DL +Data-CRF", "DL ++Data", "DL ++Data-ASPP"]
    finalResults = [watershed, preProcessed, morePreProcessed, morePreProcessedCRF, morePreProcessedMirroredScaled,
                    morePreProcessedMirroredScaledASPP]
    finalBoxPlots(finalResults, metricName, finalResultsNames, finalResFolder)


def singleBoxPlot(scores, title, figName, xAxisCaption, yAxisCaption, xTickName):
    """Plots a single box plot given all the required elements including the results, and axes information."""
    plt.figure(figsize=(4, 4))
    plt.boxplot(scores, showmeans=True)
    plt.xlabel(xAxisCaption)
    plt.ylabel(yAxisCaption)
    plt.yticks(np.arange(0.00, 1.10, step=0.10))
    plt.xticks([1], [xTickName])
    plt.suptitle(title)
    plt.savefig(figName, dpi=300, bbox_inches="tight")
    # plt.show()


def main():
    # The main function that reads the provided text files containing the average scores and plots them

    scoreShortName = 'd'
    shortMetricName, metricName = setMetricName(scoreShortName)
    finalResultsFolder = 'FinalResults'

    #generates the final comparison plot having all the input in the project directory as required.
    #generateFinalComparisonPlot(shortMetricName, metricName, finalResultsFolder)

    datasetFName = "bestModel"  # folderName

    # descMethod="Watershed"
    descMethod = "BestModel"

    method = "DL"
    # method="WS"

    generateSingleBoxPlot(shortMetricName,metricName,datasetFName,method,descMethod)





if __name__ == "__main__":
    main()
