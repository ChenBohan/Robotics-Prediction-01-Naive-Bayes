# Robotics-Prediction-Naive Bayes
Udacity Self-Driving Car Engineer Nanodegree: Prediction.

<img src="https://github.com/ChenBohan/Robotics-Prediction-Intro-to-Prediction/blob/master/readme_img/MBvsDD.png" width = "70%" height = "70%" div align=center />

## 1.Model-Based 

Model-Based: multimodal estimation algorithm

<img src="https://github.com/ChenBohan/Robotics-Prediction-Intro-to-Prediction/blob/master/readme_img/model%20based.png" width = "70%" height = "70%" div align=center />

1. Defining Process Models(offline).
2. Using Process Models: to compare driver behavior to what would be expected for each model.
3. Classifying Intent with Multiple Model Algorithm: Probabilistically classifying driver intent by comparing the likelihoods of various behaviors with a multiple-model algorithm.
4. Trajectory Generation: Extrapolating process models to generate trajectories.


### Autonomous Multiple Model

Multiple model algorithms are responsible for maintaining beliefs for the probability of each maneuver.

[A comparative study of multiple-model algorithms for maneuvering target tracking](http://xueshu.baidu.com/s?wd=paperuri%3A%28635878b3d358a4e3afac7f9a61e8835b%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fciteseerx.ist.psu.edu%2Fviewdoc%2Fdownload%3Fdoi%3D10.1.1.61.9763%26rep%3Drep1%26type%3Dpdf&ie=utf-8&sc_us=3450841413258429113)

## 2.Data-Driven

Data-Driven: meachine learning

<img src="https://github.com/ChenBohan/Robotics-Prediction-Intro-to-Prediction/blob/master/readme_img/trajectory%20clustering.png" width = "70%" height = "70%" div align=center />

<img src="https://github.com/ChenBohan/Robotics-Prediction-Intro-to-Prediction/blob/master/readme_img/online%20prediction.png" width = "70%" height = "70%" div align=center />

### Offline Training

1. Define similarity - we first need a definition of similarity that agrees with human common-sense definition.

2. Unsupervised clustering - at this step some machine learning algorithm clusters the trajectories we've observed. 

3. Define Prototype Trajectories - for each cluster identify some small number of typical "prototype" trajectories.

### Online Prediction

1. Observe Partial Trajectory - As the target vehicle drives we can think of it leaving a "partial trajectory" behind it.

2. Compare to Prototype Trajectories - We can compare this partial trajectory to the corresponding parts of the prototype trajectories. When these partial trajectories are more similar (using the same notion of similarity defined earlier) their likelihoods should increase relative to the other trajectories.

3. Generate Predictions - For each cluster we identify the most likely prototype trajectory. We broadcast each of these trajectories along with the associated probability (see the image below).

