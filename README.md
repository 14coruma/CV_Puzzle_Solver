# Project
P556 Final Project

Andrew Corum

## Deliverables:
- [x] Topic idea
- [x] Project proposal
- [x] Progress report
- [ ] Final submission
- [ ] Presentation

## CNN Trials
| Test Accuracy | Epochs | Conv Layers                                                                      | Dropout                      | Dense Layers | BatchNorm? |
|---------------|--------|----------------------------------------------------------------------------------|------------------------------|--------------|------------|
| 0.9938        | 90     | 32 (3x3), Pool (2x2)<br>64 (3x3), Pool (2x2)<br>64 (3x3)                         | N/A                          | 128<br>64    |            |
| 0.9929        | 90     | 32 (3x3), Pool (2x2)<br>64 (3x3), Pool (2x2)<br>64 (3x3)                         | 0.25<br>0.25                 | 128<br>64    |            |
| 0.9920        | 90     | 32 (3x3), Pool (2x2)<br>64 (3x3), Pool (2x2)<br>64 (3x3)                         | 0.5<br>0.5                   | 128<br>64    |            |
| 0.9932        | 90     | 32 (3x3), Pool (2x2)<br>64 (3x3), Pool (2x2)<br>64 (3x3)                         | 0.25<br>0.5                  | 128<br>64    |            |
| 0.9926        | 90     | 32 (3x3), Pool (2x2)<br>64 (3x3), Pool (2x2)<br>64 (3x3                          | 0.5<br>0.25                  | 128<br>64    |            |
| 0.9930        | 90     | 32 (3x3), Pool (2x2)<br>64 (3x3), Pool (2x2)<br>64 (3x3)                         | N/A                          | 256<br>128   |            |
| 0.9921        | 90     | 32 (3x3), Pool (2x2)<br>64 (3x3), Pool (2x2)<br>64 (3x3)                         | 0.25<br>0.25                 | 256<br>128   |            |
| 0.9932        | 90     | 32 (3x3), Pool (2x2)<br>64 (3x3), Pool (2x2)<br>64 (3x3)                         | 0.25<br>0.5                  | 256<br>128   |            |
| 0.9947        | 90     | 32 (3x3), Pool (2x2)<br>64 (3x3)<br>64 (3x3), Pool (2x2)<br>64 (3x3)             | N/A                          | 256<br>128   |            |
| 0.9937        | 90     | 32 (3x3), Pool (2x2)<br>64 (3x3)<br>64 (3x3), Pool (2x2)<br>64 (3x3)             | 0.25<br>0.25                 | 256<br>128   |            |
| 0.9950        | 90     | 32 (3x3), Pool (2x2)<br>64 (3x3)<br>64 (3x3), Pool (2x2)<br>64 (3x3)             | 0.5<br>0.5<br>0.5            | 128          |            |
| 0.9955        | 90     | 32 (3x3), Pool (2x2)<br>64 (3x3)<br>64 (3x3), Pool (2x2)<br>64 (3x3)             | 0.5<br>0.5<br>0.5            | 128          | Yes        |
| 0.9924        | 90     | 32 (3x3)<br>64 (3x3)<br>64 (3x3)<br>64 (3x3)                                     | 0.5<br>0.5<br>0.5            | 128          | Yes        |
| 0.9954        | 90     | 32 (3x3)<br>64 (3x3), Pool (2x2)<br>64 (3x3)<br>64 (3x3), Pool (2x2)<br>64 (3x3) | 0.5<br>0.5<br>0.5            | 128          | Yes        |
| 0.9937        | 90     | 32 (3x3)<br>64 (3x3), Pool (2x2)<br>64 (3x3)<br>64 (3x3), Pool (2x2)<br>64 (3x3) |                              | 128          | Yes        |
| 0.9951        | 90     | 32 (3x3)<br>64 (3x3), Pool (2x2)<br>64 (3x3)<br>64 (3x3), Pool (2x2)<br>64 (3x3) | 0.25<br>0.25<br>0.25<br>0.25 | 128          | Yes        |
| 0.9950        | 90     | 32 (3x3)<br>64 (3x3), Pool (2x2)<br>64 (3x3)<br>64 (3x3), Pool (2x2)<br>64 (3x3) | 0.4<br>0.4<br>0.4<br>0.4     | 128          | Yes        |
| 0.9960        | 90     | 32 (3x3)<br>64 (3x3), Pool (2x2)<br>64 (3x3)<br>64 (3x3), Pool (2x2)<br>64 (3x3) | 0.3<br>0.3<br>0.3<br>0.3     | 128          | Yes        |
[table generator](https://www.tablesgenerator.com/markdown_tables)


## References
* https://imagemagick.org/
* https://docs.opencv.org/
    * https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
    * https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
    * https://docs.opencv.org/master/d1/d5c/tutorial_py_kmeans_opencv.html
    * https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/
    * Adpative thresholding for various lighting conditions: https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    * Contours https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    * Hough Transform: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    * OTSU Thresholding: https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
* Webcam using OpenCV: https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python
* OpenCV crop: https://stackoverflow.com/questions/61927877/how-to-crop-opencv-image-from-center
* Bag of Visual Words:
    * https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision
    * https://medium.com/@aybukeyalcinerr/bag-of-visual-words-bovw-db9500331b2f
* Searching for array in a 2d array:
    * https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-171.php
* https://scikit-learn.org/stable/modules/classes.html
* Rubiks cube: https://towardsdatascience.com/learning-to-solve-a-rubiks-cube-from-scratch-using-reinforcement-learning-381c3bac5476
* OpenCV Sudoku solver: https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/
* Hough transform for grid detection: https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python
* TensorFlow Docs: https://www.tensorflow.org/
* Keras Docs: https://www.tensorflow.org/guide/keras
* CUDA install/setup docs: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/
* Simple CNN for MNIST dataset: https://linux-blog.anracom.com/2020/05/31/a-simple-cnn-for-the-mnist-datasets-ii-building-the-cnn-with-keras-and-a-first-test/