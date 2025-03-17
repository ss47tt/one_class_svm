# One Class SVM

How to train one class support vector machine to detect outlier images?

1. I used 12000 cat images only as training images which is only one class (cat).

2. I set the one class svm hyperparameters to kernel = "rbf", gamma = 0.001, nu = 0.08.

3. Resnet50 is used to collect features of training and testing images.

4. There are 500 cat images and 500 other (dog) images for testing.

5. The accuracy is about 85 percent.
