from scipy import ndimage as ndi
from skimage import restoration, morphology
from skimage.filters import sobel
from scipy import ndimage
from Utils.Metrics import *
from Utils.utils import *

InterimDices = []
SegNames = []


class Segmentation:
    """The Segmentation class that encapsulates the pre-processing, Watershed segmentation and post-processing steps that are only required for the Watershed approach."""

    def addRow(self, imgSlice):
        """This method appends a row of zeros (black pixels) to the passed image slice. It is useful for the cropped slices that are smaller than (2x 2), to enable plotting them."""
        newrow = np.zeros((1, imgSlice.shape[1]))
        imgSlice = np.r_[imgSlice, newrow]
        reshaped = True
        rowAdded = True
        return imgSlice, reshaped, rowAdded

    def addColumn(self, imgSlice):
        """This method appends a column of zeros (black pixels) to the passed image slice. It is useful for the cropped slices that are smaller than (2x 2), to enable plotting them."""
        newcolumn = np.zeros((imgSlice.shape[0], 1))
        imgSlice = np.c_[imgSlice, newcolumn]
        reshaped = True
        columnAdded = True
        return imgSlice, reshaped, columnAdded

    def checkSizeForContouring(self, img):
        """Check the size of the cropped image, if it is less than (2x 2), reshape it by adding a row and/or a column as required, so matplotlib will be able to plot it."""
        shape = img.shape
        reshaped = False
        rowAdded = False
        columnAdded = False

        if shape[0] == 1 and shape[1] == 1:
            # add row and column
            img, reshaped, rowAdded = self.addRow(img)
            img, reshaped, columnAdded = self.addColumn(img)


        elif shape[0] == 1:
            # add row
            img, reshaped, rowAdded = self.addRow(img)

        else:
            # add column
            img, reshaped, columnAdded = self.addColumn(img)
        return img, reshaped, rowAdded, columnAdded

    def preprocessImageSlice(self, imgSlice):
        """
        This method pre-processes the image slice and returns the cropped and the preprocessed image slice, along with the box used for
        cropping the image to be utilized in cropping the ground truth. Returning also information whether the returned cropped image slice is reshaped,
        which is important for evaluation.

        The main ideas of the preprocessing and some parts of the following code is used from:
        Reference:
        “Images and words, Emmanuelle Gouillart’s blog.” [Online]. Available: http://emmanuelle.github.io/segmentation-of-3-d-tomography-images-with-python-and-scikit-image.html. [Accessed: 23-Mar-2018].
        """

        reshaped = False
        rowAdded = False
        columnAdded = False

        allZeros = not np.any(imgSlice)
        # self.dispImg(imgSlice,"Original slice")

        # don't pre-process the image if it is all black
        if (not allZeros):
            imgSlice = normalizeImageSlice(imgSlice)
            # self.dispImg(imgSlice,"After normalization")

            # apply bilateral denoising to be able to segment the brain and crop the image
            bilateral = restoration.denoise_bilateral(imgSlice, multichannel=False)
            # self.dispImg(bilateral,"After bilateral denoising")

            # segment the foreground image, as the background pixels have the value 0 (black)
            if np.any(bilateral > 0.2):
                sample = bilateral > 0.2  # segmenting the brain
                sample = ndimage.binary_fill_holes(sample)
                # self.dispImg(sample,"After segmenting the brain")

            else:
                sample = bilateral
                sample = ndimage.binary_fill_holes(sample)

            # find the segmented object, the brain and get the its boundary box
            bbox = ndimage.find_objects(sample)

            mask = sample[bbox[0]]

            # crop the image to get rid of the background pixels
            croppedImgSlice = imgSlice[bbox[0]]
            # self.dispImg(croppedImgSlice,"After cropping")

            # if the cropped image is less than (2x 2), reshape it, so matplotlib will be able to plot it. Otherwise, it won't.
            if (croppedImgSlice.shape[0] == 1 or croppedImgSlice.shape[1] == 1):
                croppedImgSlice, reshaped, rowAdded, columnAdded = self.checkSizeForContouring(croppedImgSlice)

            # apply non-local means denoising algorithm to get a better contrast
            nlm = restoration.denoise_nl_means(croppedImgSlice, patch_size=5, patch_distance=7,
                                               h=0.12, multichannel=False)
            # self.dispImg(nlm,"After applying the non-local means denoising algorithm")


        else:
            # The image slice doesn't contain anything else than black pixels, don't preprocess it. Only crop the image for consistency.
            bbox = [slice(11, 136, None), slice(11, 136, None)]
            croppedImgSlice = imgSlice[bbox[0]]
            nlm = croppedImgSlice

        return croppedImgSlice, nlm, bbox[0], reshaped, rowAdded, columnAdded

    def segmentImage_Watershed(self, preprocessed, original):
        """This method segments the preprocessed image using the marker based Watershed algorithm. It returns the resulted segmentation.
         The following code is modified but the main ideas and some parts of the code are obtained from:
         Reference:
         “Comparing edge-based and region-based segmentation — skimage v0.14dev docs.” [Online]. Available: http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_coins_segmentation.html#sphx-glr-auto-examples-xx-applications-plot-coins-segmentation-py. [Accessed: 27-Feb-2018].
        """

        # Get the gradient of the image
        elevation_map = sobel(preprocessed)

        # Set the markers, where 0.85 corresponds the lightest regions of the MRI scan, usually, corresponding to a tumour.
        markers = np.zeros_like(preprocessed)
        markers[preprocessed < 0.8] = 1
        markers[preprocessed > 0.85] = 2

        # Apply Watershed algorithm
        segmentation = morphology.watershed(elevation_map, markers)
        segmentation = segmentation - 1  # to have a range of 0-1 instead of 1-2
        # remove the very small spots from the segmentation
        segmentation = morphology.erosion(segmentation, selem=np.ones((3, 3)))
        # connect any unconnected boundaries in the resulted segmentation, that introduce a hole.
        segmentation = self.connectHolesBoundaries(segmentation)
        # fill the holes
        segmentation = ndi.binary_fill_holes(segmentation)
        # remove small undesired objects
        segmentation = self.RemoveSmallHoles(segmentation)
        return segmentation

    def removeAddedRowOrColumn(self, image, rowAdded, columnAdded):
        """This method removes any added rows or columns used for reshaping and plotting the image."""
        if (rowAdded and columnAdded):  # a row and a column are added to the image
            image = image[0:-1, 0:-1]

        elif rowAdded:  # a row is added to the image
            image = image[0:-1, :]

        elif columnAdded:  # a column is added to the image
            image = image[:, 0:-1]

        return image

    def startProcessing(self, sampleSrc, labelSrc, pathofSegs, pathofGTs, pathofContours1, pathofContours2,
                        modalityName, names, namesIndex):
        """This method starts processing the image, by reading both the image slice and the label, preprocessing them, applying segmentation and evaluating them with Dice, Precision and Recall."""
        imageData = readImage(sampleSrc)
        labelImgData = readImage(labelSrc)
        AllSlicesDices = []
        AllSlicesPrecision = []
        AllSlicesRecall = []
        Allcardinalities = []

        tumourSlices = list(range(imageData.shape[1]))  # coronal view, 240
        TPs = TNs = FPs = FNs = 0
        # for each slice of the fully visible tumour area, preprocess the slice (image+seg), segment it and calculate cardinalities.
        for i in tumourSlices:
            gtName = names[namesIndex]
            segName = gtName.replace("seg", modalityName)
            print("Scan: {}".format(segName))
            namesIndex += 1
            imageSlice = getSliceAt(imageData, i)
            labelSlice = getSliceAt(labelImgData, i)

            reshaped = reshapedPre = reshapedPreL = False
            rowAdded = rowAddedPre = rowAddedPreL = False
            columnAdded = columnAddedPre = columnAddedPreL = False

            croppedOriginalImage, preprocessedImage, bbox, reshaped, rowAdded, columnAdded = self.preprocessImageSlice(
                imageSlice)
            preprocessedLabel = preprocessLabelSlice(labelSlice, bbox)
            # self.displayImgAndHist(preprocessedImage)
            segmentedImage = self.segmentImage_Watershed(preprocessedImage, croppedOriginalImage)
            # self.dispImportantSteps(imageSlice,preprocessedImage,croppedOriginalImage,segmentedImage,preprocessedLabel)

            # reshape the image by adding additional row or/and column if it has 1x1 shape, since it can't be displayed and saved if it has this dimension.
            reshapedPre = False
            if (preprocessedImage.shape[0] == 1 or preprocessedImage.shape[1] == 1):
                preprocessedImage, reshapedPre, rowAddedPre, columnAddedPre = self.checkSizeForContouring(
                    preprocessedImage)

            reshapedPreL = False
            if (preprocessedLabel.shape[0] == 1 or preprocessedLabel.shape[1] == 1):
                preprocessedLabel, reshapedPreL, rowAddedPreL, columnAddedPreL = self.checkSizeForContouring(
                    preprocessedLabel)

            # save the output of Watershed segmentation
            saveContouredSample(preprocessedImage, segmentedImage, preprocessedLabel, segName, pathofContours1,
                                "Watershed")
            saveContouredSample(croppedOriginalImage, segmentedImage, preprocessedLabel, segName, pathofContours2,
                                "Watershed")
            matplotlib.image.imsave("{}/{}.png".format(pathofSegs, segName), segmentedImage, cmap='gray')
            matplotlib.image.imsave("{}/{}.png".format(pathofGTs, gtName), preprocessedLabel, cmap='gray')

            # reshape the image back to its original shape for evaluation
            if (reshaped):
                segmentedImage = self.removeAddedRowOrColumn(segmentedImage, rowAdded, columnAdded)

            if reshapedPreL:
                preprocessedLabel = self.removeAddedRowOrColumn(preprocessedLabel, rowAddedPreL, columnAddedPreL)

            # calculate cardinialities
            TP, TN, FP, FN = calculateCardinalities(segmentedImage, preprocessedLabel)
            diceSimilarityScore = calculateDiceSimilarity(TP, FP, FN)
            InterimDices.append(diceSimilarityScore)
            SegNames.append(segName)

            TPs += TP
            TNs += TN
            FPs += FP
            FNs += FN

        print("Dice scores of all the slices of the image:")
        # calculate metrics once per each scan
        diceScore = calculateDiceSimilarity(TPs, FPs, FNs)
        precisionScore = calculatePrecision(TPs, FPs)
        recallScore = calculateRecall(TPs, FNs, FPs)

        return namesIndex, diceScore, precisionScore, recallScore, TPs, FPs, TNs, FNs

    def dispImg(self, img, title):
        """Displays the passed image with the passed title, used for displaying the intermediate Watershed pre-processing applied on 2D images. """
        plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest', origin="lower")
        plt.axis('off')
        plt.savefig('./{}.png'.format(title))

    def displayImgAndHist(self, img):
        """This method plots the image and the histogram of its grey intensity values."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img, cmap=plt.cm.gray, origin="lower")
        axes[0].set_title("Pre-processed image slice")
        axes[0].axis('off')
        x, y, _ = axes[1].hist(img.T.ravel(), bins=256)
        axes[1].set_title('Histogram of the grey intensity values')
        axes[1].set_xlabel(xlabel="Pixel intensities")
        axes[1].set_ylabel(ylabel="Number of pixels")
        plt.suptitle("")
        plt.savefig('./FindingMarkers.png')
        # plt.show()

    def dispImportantSteps(self, imageSlice, preprocessedImage, croppedImage, segmentedSlice, labelSlice):
        """This method plots the original image slice, the preprocessed slice, the segmented slice and the ground truth of the same slice """
        fig, axes = plt.subplots(4, 2, figsize=(6, 12))
        fig.subplots_adjust(hspace=0.3, wspace=0.1)
        axes[0, 0].imshow(imageSlice, origin="lower", cmap="gray")
        axes[0, 0].set_title("Original \n Image Slice")
        axes[0, 0].axis('off')
        fig.delaxes(axes[0, 1])

        axes[1, 0].imshow(croppedImage, cmap=plt.cm.gray, interpolation='nearest', origin="lower")
        axes[1, 0].set_title("Contrast-enhanced\nCropped Slice")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(croppedImage, cmap=plt.cm.gray, interpolation='nearest', origin="lower")
        axes[1, 1].contour(segmentedSlice, [0.5], linewidths=1.2, colors='y', origin="lower")
        axes[1, 1].set_title("Segmented Contrast-enhanced\nCropped Slice")
        axes[1, 1].axis('off')

        axes[2, 0].imshow(preprocessedImage, cmap=plt.cm.gray, interpolation='nearest', origin="lower")
        axes[2, 0].set_title("Pre-processed \n Image Slice")
        axes[2, 0].axis('off')

        axes[2, 1].imshow(preprocessedImage, cmap=plt.cm.gray, interpolation='nearest', origin="lower")
        axes[2, 1].contour(segmentedSlice, [0.5], linewidths=1.2, colors='y', origin="lower")
        axes[2, 1].set_title("Segmented \n Pre-processed Slice")
        axes[2, 1].axis('off')

        axes[3, 1].imshow(segmentedSlice, origin="lower", cmap="gray")
        axes[3, 1].set_title("Segmented Slice")
        axes[3, 1].axis('off')

        axes[3, 0].imshow(labelSlice, origin="lower", cmap="gray")
        axes[3, 0].set_title("Ground Truth")
        axes[3, 0].axis('off')
        plt.savefig('WatershedOutcome.png')
        # plt.show()

    def connectHolesBoundaries(self, segmentation):
        """This method connects any unconnected segmentation boundaries."""
        copiedSeg = segmentation.copy()
        for (i, j), element in np.ndenumerate(segmentation):
            if (i == 0 or i == len(segmentation) - 1 or j == 0 or j == len(segmentation[0]) - 1):
                continue
            elif (element == 0 and ((((segmentation[i - 1][j - 1] == 1) or (segmentation[i][j - 1] == 1) or
                                    (segmentation[i + 1][j - 1] == 1)) and ((segmentation[i - 1][j + 1] == 1) or
                                    (segmentation[i][j + 1] == 1) or (segmentation[i + 1][j + 1] == 1))) or
                                    (((segmentation[i - 1][j - 1] == 1) or (segmentation[i - 1][j] == 1) or
                                    (segmentation[i - 1][j + 1] == 1)) and ((segmentation[i + 1][j - 1] == 1) or
                                    (segmentation[i + 1][j] == 1) or (segmentation[i + 1][j + 1] == 1))))):

                copiedSeg[i][j] = 1

        return copiedSeg

    def RemoveSmallHoles(self, segmentation):
        """This method removes any small objects in the segmented image, and returns the image without these objects."""
        for (i, j), element in np.ndenumerate(segmentation):
            if segmentation[i][j] == 0:
                segmentation[i][j] = 1
            else:
                segmentation[i][j] = 0
        segmentation = morphology.remove_small_holes(segmentation, 40, 2)
        for (i, j), element in np.ndenumerate(segmentation):
            if segmentation[i][j] == True:
                segmentation[i][j] = 0
            else:
                segmentation[i][j] = 1
        return segmentation
