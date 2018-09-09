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

<img src="https://github.com/ChenBohan/Robotics-Prediction-Naive-Bayes/blob/master/readme_img/variables%20in%20AMM.png" width = "70%" height = "70%" div align=center />

<img src="https://github.com/ChenBohan/Robotics-Prediction-Naive-Bayes/blob/master/readme_img/multimodal%20estimate.png" width = "70%" height = "70%" div align=center />

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

## 3.Hybrid Approaches: Naive Bayes

1.Implement the ``train(self, data, labels)`` method in the class GNB in ``classifier.cpp``. 

Training a Gaussian Naive Bayes classifier consists of computing and storing the mean and standard deviation from the data for each label/feature pair. 

```cpp
    //For each label, compute the numerators of the means for each class
    //and the total number of data points given with that label.
	for (int i=0; i<labels.size(); i++){
	    if (labels[i] == "left"){
	        left_means += ArrayXd::Map(data[i].data(), data[i].size()); //conversion of data[i] to ArrayXd
	        left_size += 1;
	    } 
	    else if (labels[i] == "keep") {
	        keep_means += ArrayXd::Map(data[i].data(), data[i].size());
	        keep_size += 1;
	    } else if (labels[i] == "right") {
	        right_means += ArrayXd::Map(data[i].data(), data[i].size());
	        right_size += 1;
	    }
	}
	
	//Compute the means. Each result is a ArrayXd of means (4 means, one for each class)..
	left_means = left_means/left_size;
    keep_means = keep_means/keep_size;
	right_means = right_means/right_size;
	
	//Begin computation of standard deviations for each class/label combination.
	ArrayXd data_point;
	
	//Compute numerators of the standard deviations.
	for (int i=0; i<labels.size(); i++){
	    data_point = ArrayXd::Map(data[i].data(), data[i].size());
	    if (labels[i] == "left"){
	        left_sds += (data_point - left_means)*(data_point - left_means);
	    } else if (labels[i] == "keep") {
	        keep_sds += (data_point - keep_means)*(data_point - keep_means);
	    } else if (labels[i] == "right") {
	        right_sds += (data_point - right_means)*(data_point - right_means);
	    }
	}
	
	//compute standard deviations
	left_sds = (left_sds/left_size).sqrt();
    keep_sds = (keep_sds/keep_size).sqrt();
    right_sds = (right_sds/right_size).sqrt();
```

2.Implement the ``predict(self, observation)`` method in ``classifier.cpp``.

Given a new data point, prediction requires two steps: 

2.1. Compute the conditional probabilities for each feature/label combination.

For a feature xxx and label CCC with mean μ\muμ and standard deviation σ\sigmaσ (computed in training), the conditional probability can be computed using the formula:

2.2. Use the conditional probabilities in a Naive Bayes classifier. 

This can be done using the formula:

```python
string GNB::predict(vector<double> sample)
{
	//Calculate product of conditional probabilities for each label.
	double left_p = 1.0;
	double keep_p = 1.0;
	double right_p = 1.0; 
	for (int i=0; i<4; i++){
	    left_p *= (1.0/sqrt(2.0 * M_PI * pow(left_sds[i], 2))) * exp(-0.5*pow(sample[i] - left_means[i], 2)/pow(left_sds[i], 2));
	    keep_p *= (1.0/sqrt(2.0 * M_PI * pow(keep_sds[i], 2))) * exp(-0.5*pow(sample[i] - keep_means[i], 2)/pow(keep_sds[i], 2));
	    right_p *= (1.0/sqrt(2.0 * M_PI * pow(right_sds[i], 2))) * exp(-0.5*pow(sample[i] - right_means[i], 2)/pow(right_sds[i], 2));
	}
	
	//Multiply each by the prior
	left_p *= left_prior;
	keep_p *= keep_prior;
	right_p *= right_prior;
    
    	double probs[3] = {left_p, keep_p, right_p};
    	double max = left_p;
    	double max_index = 0;
    	for (int i=1; i<3; i++){
        	if (probs[i] > max) {
            		max = probs[i];
            		max_index = i;
        }
    }	
	return this -> possible_labels[max_index];
}
```


