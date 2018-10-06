# BrainTumourSegmentationUsingDeepLearning
Brain Tumour Segmentation Using DeepLabv2 algorithm.

The project is completed as part of an MSc dissertation in the University of St Andrews, UK to segment brain tumours from MRI scans of the brain, utilizing the BraTS 2015 dataset. 

The DeepLabv2 [1]⁠ algorithm, which uses ResNet-101 and CRF post-processing, is utilized to perform the automatic brain tumour segmentation. A classical image segmentation algorithm, Watershed, is also utilized to compare both approaches.

The code utilizes the tensorflow-deeplab-resnet library [2]⁠, as the deep learning model.

The code requires Python 3.6, and the following libraries:

numpy
scipy
matplotlib
Scikit-Image
SimpleITK
Pillow
Scikit-Learn
six
tqdm
Cython
Tensorflow-gpu OR Tensorflow if GPUs are not available
pydensecrf [3]⁠

The current version is tested on PyCharm.  

The current version is the initial version of the code, which is being updated currently. The second version, accompanied with more details, will be uploaded soon once the work is completed.


The code is not meant to be used for commercial purposes.


References:

[1]	L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, “DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs,” Jun. 2016.

[2]	V. Nekrasov, “Tensorflow DeepLabv2 ResNet+CRF.” [Online]. Available: https://github.com/DrSleep/tensorflow-deeplab-resnet/tree/crf. [Accessed: 18-Jun-2018].

[3]	L. Beyer, “PyDenseCRF.” [Online]. Available: https://github.com/lucasb-eyer/pydensecrf. [Accessed: 18-Jun-2018].
