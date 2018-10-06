from sklearn.model_selection import train_test_split
from Utils.utils import *
from time import time


def main():
    """The main function to create 2D images extracted from all the slices of the scans in 4 modalities, and their corresponding 2D labels (ground truth segmentations)."""

    #path of the final samples to be saved in
    pathOfAllData = "D:/DeepModelFullData"

    #path of the folder containing the BraTS 2015 training dataset
    path = "D:/DATASET/BRATS2015_Training"

    HGG, LGG = readFiles(path)
    HGG_LGG = mergeLists(HGG, LGG)
    HGG_LGG_np = np.array(HGG_LGG)
    X = HGG_LGG_np[:, :4] #scans
    y = HGG_LGG_np[:, 4]  #GT

    #split the dataset into training and testing sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training-validating: ", X_train_val.shape, " ", y_train_val.shape)
    print("Testing: ", X_test.shape, " ", y_test.shape)

    #further split the training set into training and validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=42)
    print("Training: ", X_train.shape, " ", y_train.shape)
    print("Validating: ", X_val.shape, " ", y_val.shape)

    #extract the training set 2D images by extracting all slices (240) of the scans in the set.
    print("\n Creating training data (all 240 slices)..")
    start2 = time()
    create2Dimgs(pathOfAllData, X_train, y_train, "TrainingData_Full", "train", 240)
    end2 = time()
    print("Time taken to create training data: {} seconds".format(end2 - start2))

    #extract the validation set 2D images by extracting all slices (240) of the scans in the set.
    print("\n Creating validation data (all 240 slices)..")
    start4 = time()
    create2Dimgs(pathOfAllData, X_val, y_val, "ValidationData_Full", "val", 240)
    end4 = time()
    print("Time taken to create validation data: {} seconds".format(end4 - start4))

    #extract the testing set 2D images by extracting all slices (240) of the scans in the set.
    print("\n Creating testing data (all 240 slices)..")
    start6 = time()
    create2Dimgs(pathOfAllData, X_test, y_test, "TestingData_Full", "test", 240)
    end6 = time()
    print("Time taken to create testing data: {} seconds".format(end6 - start6))


if __name__ == "__main__":
    main()
