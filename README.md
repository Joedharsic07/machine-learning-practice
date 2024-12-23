## start learning ML:

## What is Machine Learning?
Machine learning is a branch of artificial intelligence that enables algorithms to uncover hidden patterns within datasets, allowing them to make predictions on new, similar data without explicit programming for each task.

## Types Of Machine Learning:

1) Supervised Learning:
	It uses the labeled inputs to train the model and learn the outputs. 
2) un Supervised Learning:
	It uses the unlabeled data to learn about patterns in data.
3) Reinforcement Learning:
	Agent Learning in an inactive environment based on rewards and penalties. 

## Supervised Learning:
Supervised learning is a category of machine learning that uses labeled datasets to train algorithms to predict outcomes and recognize patterns.

## Unsupervised Learning:
Unsupervised learning is a type of machine learning that analyzes data without human intervention.

## Reinforcement Learning:
Reinforcement learning (RL) is a machine learning technique that teaches software to make decisions to achieve optimal results.

## K-Nearest Neighbors:
-Most basic classification algorithm
-It belongs to supervised learning
-Find intense application in pattern recognition
- it is non-parametric
-It does not make any underlying assumptions about the distribution of data

## Distance metrics used in KNN:

## Euclidean Distance:
-It is a way to measure how far apart two points are
-It is simply the length of the straight line connecting those two points

## Naive Bayes:
It’s a method used in machine learning to classify things into categories.

## The formula is:

𝑃(𝐴∣𝐵)=𝑃(𝐵∣𝐴)⋅𝑃(𝐴)
       --------
	 𝑃(𝐵)

## Where:

-P(A∣B): Probability of the hypothesis A (email is spam) given the evidence B (contains "win").
-P(B∣A): Probability of seeing the evidence B if hypothesis A is true.
-P(A): Probability of the hypothesis A being true in general.
-P(B): Probability of the evidence B appearing in any situation.

## How Does Naive Bayes Work?
-Training: The model learns from data by calculating probabilities.
(e.g) it counts how often words like "win" or "free" appear in spam vs. non-spam emails.

-Prediction: When a new input (like an email), the model calculates probabilities for each category and picks the one with the highest probability.

## Why Use Naive Bayes?
-Simple: It’s easy to implement.
-Fast: Works well even with large datasets.
-Effective: Great for tasks like spam filtering, sentiment analysis, and document classification.

## Logistic Regression:

Logistic regression is a supervised machine learning algorithm used for classification tasks where the goal is to predict the probability that an instance belongs to a given class or not.

-Logistic regression predicts the output of a categorical dependent variable. Therefore,  the outcome must be a categorical or discrete value.
-It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1.
-In Logistic regression, instead of fitting a regression line, we fit an “S” shaped logistic function, which predicts two maximum values (0 or 1).

## Types of Logistic Regression:

-Binomial: 
	In binomial Logistic regression, there can be only two possible types of dependent variables, such as 0 or 1, Pass or Fail, etc.
-Multinomial: 
	In multinomial Logistic regression, there can be 3 or more possible unordered types of the dependent variable, such as “cat”, “dogs”, or “sheep”
-Ordinal: 
	In ordinal Logistic regression, there can be 3 or more possible ordered types of dependent variables, such as “low”, “Medium”, or “High”.

Support Vector Machine:
	A Support Vector Machine (SVM) is a supervised machine learning algorithm used for both classification and regression tasks. While it can be applied to regression problems, SVM is best suited for classification tasks.

## Neural Network:
Neural networks are capable of learning and identifying patterns directly from data without pre-defined rules.

## These networks are built from several key components:

-Neurons: The basic units that receive inputs, each neuron is governed by a threshold and an activation function.

-Connections: Links between neurons that carry information, regulated by weights and biases.

-Weights and Biases: These parameters determine the strength and influence of connections.

-Propagation Functions: Mechanisms that help process and transfer data across layers of neurons.
Learning Rule: The method that adjusts weights and biases over time to improve accuracy.

## Learning in neural networks follows a structured, three-stage process:

1)Input Computation: Data is fed into the network.

2)Output Generation: Based on the current parameters, the network generates an output.

3)Iterative Refinement: The network refines its output by adjusting weights and biases, gradually improving its performance on diverse tasks.

## Forward Propagation:

When data is input into the network, it passes through the network in the forward direction, from the input layer through the hidden layers to the output layer. This process is known as forward propagation. 

## Linear Transformation:
Each neuron in a layer receives inputs, which are multiplied by the weights associated with the connections. These products are summed together, and a bias is added to the sum.
	
## Activation:
The result of the linear transformation (denoted as z) is then passed through an activation function. The activation function is crucial because it introduces non-linearity into the system, enabling the network to learn more complex patterns. 

## Backpropagation:
After forward propagation, the network evaluates its performance using a loss function, which measures the difference between the actual output and the predicted output. The goal of training is to minimize this loss. This is where backpropagation comes into play:

1)Loss Calculation: The network calculates the loss, which provides a measure of error in the predictions. The loss function could vary; common choices are mean squared error for regression tasks or cross-entropy loss for classification.

2)Gradient Calculation: The network computes the gradients of the loss function with respect to each weight and bias in the network. This involves applying the chain rule of calculus to find out how much each part of the output error can be attributed to each weight and bias.

3)Weight Update: Once the gradients are calculated, the weights and biases are updated using an optimization algorithm like stochastic gradient descent (SGD). The weights are adjusted in the opposite direction of the gradient to minimize the loss. The size of the step taken in each update is determined by the learning rate.

## Iteration:
This process of forward propagation, loss calculation, backpropagation, and weight update is repeated for many iterations over the dataset. Over time, this iterative process reduces the loss, and the network’s predictions become more accurate.

Through these steps, neural networks can adapt their parameters to better approximate the relationships in the data, thereby improving their performance on tasks such as classification, regression, or any other predictive modeling.

## Linear regression:
   Linear regression is a type of supervised machine learning algorithm that computes the linear relationship between the dependent variable and one or more independent features by fitting a linear equation to observed data.
When there is only one independent feature, it is known as Simple Linear Regression, and when there are more than one feature, it is known as Multiple Linear Regression.
Similarly, when there is only one dependent variable, it is considered Univariate Linear Regression, while when there are more than one dependent variables, it is known as Multivariate Regression.

## Types of linear regression:

## Simple linear regression:
   This is a simple form of linear regression, and it involves only one independent variable and one dependent variable. The equation for simple linear regression is:
## y=β0+β 1X

-Y is the dependent variable
-X is the independent variable
-β0 is the intercept
-β1 is the slope

## K-mean clustering:
   Unsupervised Machine Learning is the process of teaching a computer to use unlabeled, unclassified data and enabling the algorithm to operate on that data without supervision.	
The goal of clustering is to divide the set of data points into a number of groups so that the data points within each group are more capable to one another

## Algorithm:

1. Unsupervised Machine Learning is the process of teaching a computer to use unlabeled, unclassified data and enabling the algorithm to operate on that data without supervision.
2. We categorize each item to its closest mean, and we update the mean’s coordinates, which are the averages of the items categorized in that cluster so far.
3. We repeat the process for a given number of iterations and at the end, we have our clusters.

## Bias-Variance Trade Off:
   The Bias-Variance Tradeoff is a fundamental concept in machine learning that describes the balance between two sources of error in a model: bias and variance. It helps us understand how to create a model that performs well on both the training data and unseen data 
## Bias:
-Bias refers to errors made because the model is too simple to understand the true patterns in the data.
-It often leads to underfitting, where the model performs poorly on both training and test data.
-Example: Using a straight line to model data that follows a curve.
Variance:
-Variance refers to errors made because the model is too complex and overly sensitive to small variations in the training data.
-It often leads to overfitting, where the model performs well on training data but poorly on test data.
-Example: Using a highly flexible curve that tries to go through every data point, including noise.
## The Tradeoff:
-Reducing bias makes the model more complex, which might increase variance.
-Reducing variance simplifies the model, which might increase bias.
-The goal is to find a balance where the model is just right—not too simple, not too complex.

## Overfitting:
   It occures when the model is perform well in the training while performing worse on the test dataset.

## How to fix Overfitting:
1- Reduces the complaxity of the model
2- collect more data
3- Use resampling technique
4- Early stopping
5- Ensemble methods
6- Dropout

## Regularization:
Regularization or shrinkage is a model that shrinks some of the estimated coefficients towards zero, to pentalize unimportant variables for incresing the variance of the model
- Used to solve the overfitting problem
- introduces a little bias in model to decreases its variance 

## Types of Regularization:
- Ridge Regression based on L2 norm
- lasso Regression based on L1 norm
- Dropout (in NN)

## Ridge Regression:
   Ridge Regression or L2 regularization is a srinkage technique that aims to solve Overflittingn by shrinking some of the model coefficients towarde o
   L2 norm is a mathematical term comming from Linear Algebra and it stands for Euclidean norm or Distance
pros:
- solves overfitting
- Lower model variance
- computationally cheap 
cons:
-low interpretability

## lasso Regression:
Lasso Regression or L1 regularization is a srinkage technique that aims to solve Overflittingn by shrinking some of the model coefficients towarde 0 and setting some to exactly 0
L1 norm is a mathematical term coming from Linear Algebra and it stands for a manhattan norm or distance
pros:
- solves overfitting
- easy to understand
- high interpretability
- feature selection
cons:
- higher variance than Ridge 

## Dependent Variable:
   Dependent Variable is the variable that is being measured or tested. It is called the dependent variable because it is though to depend on the independent variables

## Independent Variable:
   It is the variable that is being manipulated or controlled in the experiment, and it is belived to have an effect on the dependent variable
## Statistical Significant Effect:
   It means that this effect in unlikely to have occured by chance. In other words, a statistically significant effect in one that is likely to be real and not due to random chance
## Logistic Regression:
   It is a supervised classification technique that models conditional probablity of an event occuring or observation belong to a certain class,given of dependent variables

   - Relation between two variaables is liniear 
   - Logistic function,too large and too small values [0,1]
## Handling Missing Data Analysis
- Missing Data Analysis is the process of examining, understanding, and addressing missing or incomplete data in a dataset.
- Missing data can occur for various reasons, such as human error, data corruption, or certain responses being unavailable
- This analysis is crucial because missing data can distort results, reduce accuracy, and bias outcomes if not handled properly.
- Identifies Patterns: Helps to see if data is missing randomly or systematically.
- Prevents Bias: Ensures missing values don’t skew results.
- Improves Decision-Making: Helps choose the right strategy to handle missing values.
- Maintains Data Integrity: Preserves as much useful information as possible.

## Types of Missing Data
1. MCAR (Missing Completely at Random)
- Missing data is unrelated to any variables in the dataset.
- Example: A random technical glitch causes missing entries.
- Impact: Can be ignored or handled easily without introducing bias.
  
2. MAR (Missing at Random)
- Missing data depends on other observed variables but not the missing variable itself.
- Example: Income data missing mostly for younger individuals.
- Impact: Requires careful handling, often using other variable data.

3. MNAR (Missing Not at Random)
- Missing data depends on the missing variable itself.
- Example: People with low income don’t report their salary.
- Impact: More challenging to address; often requires domain knowledge.

## descriptive Statistics:
Descriptive statistics summarize data to make it easy to understand.
-The purpose is used to describe the main feature of dataset using numbers.
## Key Components:
## 1. Central Tendency (Finding the Center):
- Mean: The average value.
- Median: The middle value when sorted.
- Mode: The most frequent value.
## 2. Spread (How Spread Out Data Is):
-Range: Difference between the largest and smallest values.
- Variance: How far data points are from the mean (on average).
- Standard Deviation: Average distance of values from the mean.

## 3. Other Metrics:
-Minimum/Maximum: Smallest and largest values.
-Quartiles: Divide data into 4 parts.
-Interquartile Range (IQR): Middle 50% of data (between Q1 and Q3).

##  Data Visualization:
Data visualization shows data as graphs or charts to make it easier to see patterns.
The purpose is to make large datasets simple to understand, Help spot trends, patterns, or outliers, and Communicate findings clearly.
## Types of Visualizations:
1. For Single Variables:
- Histogram: Shows the frequency of data in ranges (bins).
- Box Plot: Displays data spread, median, quartiles, and outliers.
2. For Two Variables:
- Scatter Plot: Shows the relationship between two numbers.
- Bar Chart: Compares categories (e.g., average sales by region).
- Line Chart: Displays trends over time.
3. For Multiple Variables:
- Heatmap: Shows correlation or relationships between variables.
- Pair Plot: Combines multiple scatter plots for comparison.
