# Finding Lemons: Traditional Algorithms vs Self Organizing Maps 

## Structure

The goal of this project is to compare an unsupervised deep learning algorithm to its more traditional classification counterpart. Self Organizing Maps are effective at group together data with large feature sets. It can differentiate and detect correlations that are normally difficult to detect. We will use this attribute in hopes to be able to detect a junk car, a lemon. We will compare the Self Organizing Maps ability to classify a junk vechile to the more traditional machine learning algorithms (link to [Final Report](report.pdf) ).

The project is divided into nine sections. The first section is focus on preparing the data to be used by future algorithms. Section 2, Logistic regression, will be our baseline model to compare the other machine learning algorithms to. We will also compare our results to human error, the total number of bad purchases out of all the total purchases. This will be a close approximation to Bayes error; an asymtopic level error rate. Since the target feature is binary, we will evalute our model performance with standard metrics such as accuracy, precision and recall. Your feedback on the notebooks is welcome!

All the notebooks below, with the exception of XgBoost and Self Organizing Maps, had their features extracted using a Linear Discriminant Analysis. This allowed all the algorithms to run faster. 


* **[Section 1: Data Exploration and Prepocessing](Section1_DataExplorationPreProcessing.ipynb)** — The goal of the Data Preprocessing is to make the dataset compatiable for various Machine Learning algorithms. This will requrie creating dummy variables for the categorical features and standizing continous features. When a feature is missing data, we will visualize the feature with histograms and boxplots in order to determine the best method of imputation.


* **[Section 2: Logistic Regression](Section2_LogisticRegression.ipynb)** — The central premise of Logistic Regression is the assumption that your input space can be separated into two ‘regions’, one for each class, by a linear boundary. Basically, its a type of model that is really good at answering binary questions such as yes or no to whether something will occur. The 'magic' of logistic regressions comes from the fact that most things do not have a perfectly linear relation and therefore a special sigmoid function is more useful. The sigmoid function allows us to predict probabilites of an event occuring.


* **[Section 3: K Nearest Neighbors](Section3_KNearestNeighbors.ipynb)** — The K-nearest Neighbor is a fairly intuitive algorithm. When checking to see how to classify a new data point, look at n number of the nearest data points. If n=1, then which ever data point is closest to the new data point, that new data point will be grouped with its nearest neighbor.


* **[Section 4: Support Vector Machines(SVM)](Section4_SupportVectorMachines.ipynb)** — Support Vector Machines try to find the best line or decision boundary that can separate out the different classes. Unlike a simple regression line that tries to minimize the distance of the line to all the points, SVM creates a line that has the highest Maximum Margin between the two closest points(support vectors) from a different class. The strength of SVM's lies in the fact they take the most extremes and compare the two instead of using the most common "traits." For example, it would take a yellow apple and reddish orange and see how they are different. It looks at the very extreme cases and uses that to construct analysis.  


* **[Section 5: Naive Bayes](Section5_NaiveBayes.ipynb)** — Naive Bayes Classifiers works on Bayesian principles. The components of Bayes theorem; posterior probability, likelihood, prior probabilty and marginal likelihood are all used. The reason this this algorithm is called Naive Bayes is due to the fact the condition of feature independence is not required. 


* **[Section 6: Decision Tree](Section6_DecisionTree.ipynb)** — Decisions trees takes each feature and makes a decision / split on a value that provides the most information gain(minimum entropy). Then it alternates to another feature and makes the same split decision. It continues to do so until no further information gain is available. When classifying a new data point, take the average values of all the data points in that decision section. 


* **[Section 7: Random Forest](Section7_RandomForest.ipynb)** — A Random forest is another version of ensemble learning;other examples include gradient boosting. Ensemble learning is when you take multiple algorithms or the same alogorithm multiple times and you put them together to get something much more powerful than the original. In this case, we are in essences running multiple decision trees but instead of getting one prediction value from a decision tree, you get prediction values from the entire forest.  

* **[Section 8: XgBoost](Section8_XGBoost.ipynb)** — XgBoost is one of the most popular models in machine learning. It is also the most powerful implementation of gradient boosting. One of the major advantages of Xgboost besides having high performance and fast execution speed is that you can keep the interpretation of the original problem.  We were unable to do a K-fold cross validation  with our limited computational power.

* **[Section 9: Self Organizing Maps](Section9_SelfOrganizingMaps.ipynb)** - Self Organizing Maps are an unsupervised deep learning algorithm used for feature detection. Specifically, SOMs are used for reducing dimensionality. They take a multidimensional dataset and reduce it to a 2 dimensional representation of your dataset.The are similar to the input / output architecture of Neural Networks except the weights are not actual weights that are multiplied but coordinate representation in 2 dimensions; characteristics of the node itself. Inputs are values between 0 and 1 so must be standardized or normalized. Whichever node has the smallest distance represents the best matching unit(BMU). After finding the BMU, the SOM is going to update the "weights" so that BMU so it is even closer to the row that was input in. Since you move the BMU, you will move the entire node set closer to that row of value.Self Organizing Maps retain the topology of the input set and often times reveal correlations that are not easily identified. Since they are unsupervised, we do not need to worry about preliminary relations.







You can also read a [Capstone Report](report.pdf) which summarizes the implementation as well as the methodology of the whole project.



## Requirements

### Dataset

The dataset consist of one files(training.csv) and needs to be downloaded separately (~14 MB). Just unzip it in the same directory with notebooks. The dataset is orginally available for download on Kaggle's competition page but has long since removed. A copy as been uploaded here. 


### Software

This project uses the following software (if version number is omitted, latest version is recommended):


* **Python stack**: python 3.5.3, numpy, scipy, sklearn, pandas, matplotlib, h5py.
* **Gradient Boost stack**: multi-threaded xgboost should be compiled, xgboost python package is also required.


## Guide to running this project

### Option 1 - Setting up Desktop to run  Nvidia's GeForce 770 (This project)

All the algorithms except the Self Organizing maps can be run on a simple desktop setup. The Self Organizing Maps however is computational intense. Below are two options for running this algorithm. 

**Step 1. Install necessary drivers to use GPU**
The desktop is running Windows 7 with the following installs:
If you need the C++ compiler, you can download it here (**[C++ Compiler](http://landinghub.visualstudio.com/visual-cpp-build-tools)**) 

* **cuda toolkit -** https://developer.nvidia.com/cuda-toolkit -  The NVIDIA® CUDA® Toolkit provides a development environment for creating high performance GPU-accelerated applications. With the CUDA Toolkit, you can develop, optimize and deploy your applications on GPU-accelerated embedded systems, desktop workstations, enterprise data centers, cloud-based platforms and HPC supercomputers. The toolkit includes GPU-accelerated libraries, debugging and optimization tools, a C/C++ compiler and a runtime library to deploy your application.

* **Nvidia Drivers -** http://www.nvidia.com/Download/index.aspx

* **cuDNN 7 -** https://developer.nvidia.com/cudnn - cuDNN is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.

**Step 2. Install DeepLearning Packages using Conda**
* **Theano -** ' `conda install -c conda-forge theano`

* **Tensorflow GPU-** 

 `conda create -n tensorflow python=3.5`
 
 `activate tensorflow`
 
 `pip install --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-win_amd64.whl`


* **Keras -** `conda install -c conda-forge keras`

### Option 2 - Using AWS instances

**Step 1. Launch EC2 instance**
The cheaper option to run the project is to use EC2 AWS instances:

* `c4.8xlarge` CPU optimized instance for Feature Selection calculations (best for Part 2).
* `EC2 g2.2xlarge` GPU optimized instance for MLP and ensemble calculations (best for Part 3). If you run an Ireland-based spot instance, the price will be about $0.65 per hour or you can use "Spot Instances" to help reduce cost.

* For more detail instructions view the following link : http://markus.com/install-theano-on-aws/

Please make sure you run Ubuntu 14.04. For Ireland region you can use this AMI: **ami-ed82e39e**. Also, add 30 GB of EBS volume 

**Step 2. Clone this project**

`sudo apt-get install git`

`cd ~; git clone https://github.com/volak4/Zcar.git`

`cd Zcar
