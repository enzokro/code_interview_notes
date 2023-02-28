
# General notes about role of ML Engineer
```
Responsibilities of a research engineer can differ quite a lot by company, and even by team within the same company. But generally, doing effective research means taking advantage of the work that others have done, be it algorithm implementations, model checkpoints, or evaluation tools. In order for this to happen at scale, a level of software engineering discipline is necessary. I define the primary goal of a research engineer as enabling, contributing to, and accelerating ML research by bringing engineering expertise to the projects. Some examples of a research engineer’s responsibilities are:

Implementing algorithms and related baselines under a common API to allow for rapid experimentation.
Setting up distributed training.
Creating evaluation tools in Jupyter notebooks.
Often, a research engineer also makes contributions to the research itself, especially using insights and intuitions derived from implementing and iterating on experiments.

If you want a specific example, I’ve found this talk from AllenNLP to closely represent the kinds of things a research engineer thinks about.
```

## Good questions to feel out the details of a team:
```
What questions should I ask that is specific to the research engineer role if I’m considering joining a team?
I think the overaching message of this FAQ has been that there is a lot of variance in the RE role. Therefore, it’s important to figure out exactly what it means to be an RE on each team. Some important questions are:

If you distinguish between research RS/RE/SWE, can you explain the distinction?
If these distinctions exist, what’s the typical team composition for a project? What’s the collaboration pattern? IMHO, the most productive and healthy pattern is where research scientists and engineers are both invested and bring their unique expertise in an equal partnership. Other patterns, such as research engineers “working for” research scientists or research engineers leading projects with research scientists in a “consulting” role for multiple projects, are OK but less ideal.
How many projects does a research engineer typically contribute to? More than 3 is a signal that the work is probably more platform focused. When it comes to research, you cannot really meaningfully work on more than one main project and one side project.
What will be my balance between research vs. production? What’s the relative priority between publishing and launching? (While this is a subpar metric for many reasons, I find it’s still the best one to gauge this balance).
Will my manager be a research scientist or research engineer? Will I have a mentor who is the other kind my manager is not?
What is the intended career track for a research engineer on the team? Become a research expert? Become more of a SWE TL? Become a manager? Some combination of the above?
```



## Generic example of ML questions
What is batch normalisation? What are its drawbacks? Describe normalisations that rectify them.
Explain bias-variance trade-off.
List common generative models and their pros and cons.
Why is it difficult to train vanilla GANs? How can you improve the training.
How would you prevent a neural network from overfitting?
Explain how to apply drop-out. Does it differ for train and test?
Give examples of neural-network optimisers, how do they work? What is gradient explosion? How to rectify it with activation functions?
Why neural networks might not be suitable for a given task? (check out this article)
Explain attention mechanism in transformers (these are now a general framework that can be applied to a wide range of problems).
Describe a dimensionality reduction technique of your choice.

# MORE CV TAILORED
Why CNNs are so good when applied to images?
How to apply deconvolution?
Why transformers are challenging to apply to images?
How would you map emotions from one face to another?
What is temporal consistency and how to achieve it?
What loss functions can you use for a generative model?
Why FID score might is not the best realism metric?
Describe convolution types and the motivation behind them.
What is instance normalisation?
Describe loss functions for image reconstruction and their pros and cons.


# Questions from Chip Hyuen's ML Interview and ML System books

## Math

### Vectors



- Dot Product
1. What is the geometric interpretation of the dot product?  
2. Given a vector, find unit-length vector such that the dot product is maximum.

- Outer Product
1. Calculate the outer product of two vectors
2. How can the outer product be useful in ML?

- What does it mean for vectors to be linearly independent?
- How can you check two sets of vectors share the same basis?
- Given vector of a given dimension, what's the dimension of its span?

- Norms and metrics
1. What is a norm?
2. How do norms and metric differ? Given a metric, can we make it a norm?
   

### Matrices

- Why do we say matrices are linear transformations?
- What's the inverse of a matrix? Do all matrices have an inverse? Is an inverse always unique?
- What does the determinant of a matrix represent?
- What happens to the determinant of a matrix if we multiply a row by scalar t x R?
- A 4x4 matrix has four eigenvalues: 3, 3, 2, -1. What can we say about the trace and determinant?
- Given matrix [[1, 4, -2], [-1, 3, 2], [3, 5, -6]]: what can we say about this matrix's determinant? Hint: rely on a property of this matrix, look at first/last columns.
- What's the difference between covariance matrix and Gram matrix? A_T * A vs. A * A_T
- Give A \eps R^(n x m) and b \eps R^n,
  - Find x such that Ax = b
  - When does this have a unique solution?
  - Why does Ax = b have multiple solutions when A has more columns than rows?
  - How can we solve Ax = b when A has no inverse? What is the pseudo inverse?

- What does the derivative represent?
- What's the difference between derivative, gradient, and Jacobian?

- Given w \eps R^(d, m) and mini-batch x of "N" elements, each element is shape (1 x d) -> x \eps R(n, d). We have output y = F(x, w) = xw. What's the dimension of the Jacobian dy/dx?

- Given a large symmetrix matrix that doesn't fit in memory, A \eps R(1M, 1M) and a function f, that we can compute f(x) = Ax for x \eps R^(1M). Find the unit vector x so that (xT * A * x) is minimal.
  - Hint: can you frame it as optimization problem, can we use gradient descent to find a solution? Iterate through rows.


### Dimensionality Reduction

- Why do we need it?
- Is the Eigendecomp of a matrix always unique?
- Name some application of eigenvalues and eigenvectors.
- Will PCA work on a dataset of multiple feature with different ranges?
- Under what conditions can we use PCA? What about SVD?
  - What is the relationship between SVD and eigendecomp?
  - What about relationship between PCA and SVD?
- How does t-SNE (t-distributed Stochastic Neighbor Embedding) work? Why do we need it?

```PCA NOTES:
Assume that your grandma likes wine and would like to find characteristics that best describe wine bottles sitting in her cellar. There are many characteristics we can use to describe a bottle of wine including age, price, color, alcoholic content, sweetness, acidity, etc. Many of these characteristics are related and therefore redundant. Is there a way we can choose fewer characteristics to describe our wine and answer questions such as: which two bottles of wine differ the most?

PCA is a technique to construct new characteristics out of the existing characteristics. For example, a new characteristic might be computed as age - acidity + price or something like that, which we call a linear combination.

To differentiate our wines, we'd like to find characteristics that strongly differ across wines. If we find a new characteristic that is the same for most of the wines, then it wouldn't be very useful. PCA looks for characteristics that show as much variation across wines as possible, out of all linear combinations of existing characteristics. These constructed characteristics are principal components of our wines.

If you want to see a more detailed, intuitive explanation of PCA with visualization, check out amoeba's answer on StackOverflow. This is possibly the best PCA explanation I've ever read.
```

### Calculus and Convexity

Differentiable funcs:
- What does it mean for function f to be differentiable?
- When does function f "not" have a derivative at a point?
- Give example of non-differentiable functions used in ML. How do we take the gradients?


# Convexity
- What does concave / convex mean? Draw them
- Why is convexity desirable for optimization?
- Show the cross-entropy is convex

- Given logistic discriminant classifier, with sigmoid activation function:
- p(y = 1 | x) = sigma(wT * x), sigma = sigmoid function 1/(1 + e^-x), and loss = -log(p(y|x))
  - Show that p(y=-1|x) = sigma(-wT * x)
  - Show that grad( L(x,y | w) ) = -y * (1 - p(y|x)) * x 
  - Show that grad(Loss) is convex

Most ML algos use 1st order derivatives.
- How can we use 2nd order derivatives for optimization?
- Pros and cons of second order derivatives?
- Why isn't 2nd order used more in practice?

- How can use the Hessian matrix to test for critical points?

- Explain Jensen's inequality

- Explain the chain rule

- Given function f(x,y) = 4x^2 - y with constraint that x^2 + y^2 = 1, find the functions max and minimum values.
  - Note: turn constraint into lagrangian

```General notes:
On convex optimization:

Convex optimization is important because it's the only type of optimization that we more or less understand. Some might argue that since many of the common objective functions in deep learning aren't convex, we don't need to know about convex optimization. However, even when the functions aren't convex, analyzing them as if they were convex often gives us meaningful bounds. If an algorithm doesn't work assuming that a loss function is convex, it definitely doesn't work when the loss function is non-convex.

Convexity is the exception, not the rule. If you're asked whether a function is convex and it isn't already in the list of commonly known convex functions, there's a good chance that it isn't convex. If you want to learn about convex optimization, check out Stephen Boyd's textbook.


On Hessian matrix:

The Hessian matrix or Hessian is a square matrix of second-order partial derivatives of a scalar-valued function.

Given a function . If all second partial derivatives of f exist and are continuous over the domain of the function, then the Hessian matrix H of f is a square nn matrix such that: .

Hessian matrix
The Hessian is used for large-scale optimization problems within Newton-type methods and quasi-Newton methods. It is also commonly used for expressing image processing operators in image processing and computer vision for tasks such as blob detection and multi-scale signal representation.
```

### Stats and Probability

```General questions, need to dive in
[E] Given a uniform random variable  X  in the range of  [0,1]  inclusively. What’s the probability that  X=0.5 ?
[E] Can the values of PDF be greater than 1? If so, how do we interpret PDF?
[E] What’s the difference between multivariate distribution and multimodal distribution?
[E] What does it mean for two variables to be independent?
[E] It’s a common practice to assume an unknown variable to be of the normal distribution. Why is that?
[E] How would you turn a probabilistic model into a deterministic model?
[H] Is it possible to transform non-normal variables into normal variables? How?
[M] When is the t-distribution useful?
Assume you manage an unreliable file storage system that crashed 5 times in the last year, each crash happens independently.
[M] What's the probability that it will crash in the next month?
[M] What's the probability that it will crash at any given moment?
[M] Say you built a classifier to predict the outcome of football matches. In the past, it's made 10 wrong predictions out of 100. Assume all predictions are made independently., what's the probability that the next 20 predictions are all correct?
[M] Given two random variables  X  and  Y . We have the values  P(X|Y)  and  P(Y)  for all values of  X  and  Y . How would you calculate  P(X) ?
[M] You know that your colleague Jason has two children and one of them is a boy. What’s the probability that Jason has two sons? Hint: it’s not  12 .
There are only two electronic chip manufacturers: A and B, both manufacture the same amount of chips. A makes defective chips with a probability of 30%, while B makes defective chips with a probability of 70%.
[E] If you randomly pick a chip from the store, what is the probability that it is defective?
[M] Suppose you now get two chips coming from the same company, but you don’t know which one. When you test the first chip, it appears to be functioning. What is the probability that the second electronic chip is also good?
There’s a rare disease that only 1 in 10000 people get. Scientists have developed a test to diagnose the disease with the false positive rate and false negative rate of 1%.
[E] Given a person is diagnosed positive, what’s the probability that this person actually has the disease?
[M] What’s the probability that a person has the disease if two independent tests both come back positive?
[M] A dating site allows users to select 10 out of 50 adjectives to describe themselves. Two users are said to match if they share at least 5 adjectives. If Jack and Jin randomly pick adjectives, what is the probability that they match?
[M] Consider a person A whose sex we don’t know. We know that for the general human height, there are two distributions: the height of males follows  hm=N(μm,σ2m)  and the height of females follows  hj=N(μj,σ2j)  . Derive a probability density function to describe A’s height.
[H] There are three weather apps, each the probability of being wrong ⅓ of the time. What’s the probability that it will be foggy in San Francisco tomorrow if all the apps predict that it’s going to be foggy in San Francisco tomorrow and during this time of the year, San Francisco is foggy 50% of the time?

Hint: you’d need to consider both the cases where all the apps are independent and where they are dependent.

[M] Given  n  samples from a uniform distribution  [0,d] . How do you estimate  d ? (Also known as the German tank problem)
[M] You’re drawing from a random variable that is normally distributed,  X∼N(0,1) , once per day. What is the expected number of days that it takes to draw a value that’s higher than 0.5?
[M] You’re part of a class. How big the class has to be for the probability of at least a person sharing the same birthday with you is greater than 50%?
[H] You decide to fly to Vegas for a weekend. You pick a table that doesn’t have a bet limit, and for each game, you have the probability  p  of winning, which doubles your bet, and  1−p  of losing your bet. Assume that you have unlimited money (e.g. you bought Bitcoin when it was 10 cents), is there a betting strategy that has a guaranteed positive payout, regardless of the value of  p ?
[H] Given a fair coin, what’s the number of flips you have to do to get two consecutive heads?
[H] In national health research in the US, the results show that the top 3 cities with the lowest rate of kidney failure are cities with populations under 5,000. Doctors originally thought that there must be something special about small town diets, but when they looked at the top 3 cities with the highest rate of kidney failure, they are also very small cities. What might be a probabilistic explanation for this phenomenon?

Hint: The law of small numbers.

[M] Derive the maximum likelihood estimator of an exponential distribution.
```

```Stats notes and questions:
[E] Explain frequentist vs. Bayesian statistics.
[E] Given the array  [1,5,3,2,4,4] , find its mean, median, variance, and standard deviation.
[M] When should we use median instead of mean? When should we use mean instead of median?
[M] What is a moment of function? Explain the meanings of the zeroth to fourth moments.
[M] Are independence and zero covariance the same? Give a counterexample if not.
[E] Suppose that you take 100 random newborn puppies and determine that the average weight is 1 pound with the population standard deviation of 0.12 pounds. Assuming the weight of newborn puppies follows a normal distribution, calculate the 95% confidence interval for the average weight of all newborn puppies.
[M] Suppose that we examine 100 newborn puppies and the 95% confidence interval for their average weight is  [0.9,1.1]  pounds. Which of the following statements is true?

Given a random newborn puppy, its weight has a 95% chance of being between 0.9 and 1.1 pounds.
If we examine another 100 newborn puppies, their mean has a 95% chance of being in that interval.
We're 95% confident that this interval captured the true mean weight.

Hint: This is a subtle point that many people misunderstand. If you struggle with the answer, Khan Academy has a great article on it.

[H] Suppose we have a random variable  X  supported on  [0,1]  from which we can draw samples. How can we come up with an unbiased estimate of the median of  X ?
[H] Can correlation be greater than 1? Why or why not? How to interpret a correlation value of 0.3?
The weight of newborn puppies is roughly symmetric with a mean of 1 pound and a standard deviation of 0.12. Your favorite newborn puppy weighs 1.1 pounds.
[E] Calculate your puppy’s z-score (standard score).
[E] How much does your newborn puppy have to weigh to be in the top 10% in terms of weight?
[M] Suppose the weight of newborn puppies followed a skew distribution. Would it still make sense to calculate z-scores?
[H] Tossing a coin ten times resulted in 10 heads and 5 tails. How would you analyze whether a coin is fair?
Statistical significance.
[E] How do you assess the statistical significance of a pattern whether it is a meaningful pattern or just by chance?
[E] What’s the distribution of p-values?
[H] Recently, a lot of scientists started a war against statistical significance. What do we need to keep in mind when using p-value and statistical significance?
Variable correlation.
[M] What happens to a regression model if two of their supposedly independent variables are strongly correlated?
[M] How do we test for independence between two categorical variables?
[H] How do we test for independence between two continuous variables?
[E] A/B testing is a method of comparing two versions of a solution against each other to determine which one performs better. What are some of the pros and cons of A/B testing?
[M] You want to test which of the two ad placements on your website is better. How many visitors and/or how many times each ad is clicked do we need so that we can be 95% sure that one placement is better?
[M] Your company runs a social network whose revenue comes from showing ads in newsfeed. To double revenue, your coworker suggests that you should just double the number of ads shown. Is that a good idea? How do you find out?
Imagine that you have the prices of 10,000 stocks over the last 24 month period and you only have the price at the end of each month, which means you have 24 price points for each stock. After calculating the correlations of 10,000 * 9,9992 pairs of stock, you found a pair that has the correlation to be above 0.8.

[E] What’s the probability that this happens by chance?
[M] How to avoid this kind of accidental patterns?
Hint: Check out the curse of big data.

[H] How are sufficient statistics and Information Bottleneck Principle used in machine learning?
```


### Data structures and algos

ALGORITHMS
```Code a bit every day. Mastery requires practice and dedication
Write a Python function to recursively read a JSON file.
Implement an  O(NlogN)  sorting algorithm, preferably quick sort or merge sort.
Find the longest increasing subsequence in a string.
Find the longest common subsequence between two strings.
Traverse a tree in pre-order, in-order, and post-order.
Given an array of integers and an integer k, find the total number of continuous subarrays whose sum equals  k . The solution should have  O(N)  runtime.
There are two sorted arrays  nums1  and  nums2  with  m  and  n  elements respectively. Find the median of the two sorted arrays. The solution should have  O(log(m+n))  runtime.
Write a program to solve a Sudoku puzzle by filling the empty cells. The board is of the size  9×9 . It contains only 1-9 numbers. Empty cells are denoted with *. Each board has one unique solution.
Given a memory block represented by an empty array, write a program to manage the dynamic allocation of that memory block. The program should support two methods: malloc() to allocate memory and free() to free a memory block.
Given a string of mathematical expression, such as 10 * 4 + (4 + 3) / (2 - 1), calculate it. It should support four operators +, -, :, /, and the brackets ().
Given a directory path, descend into that directory and find all the files with duplicated content.
In Google Docs, you have the Justify alignment option that spaces your text to align with both left and right margins. Write a function to print out a given text line-by-line (except the last line) in Justify alignment format. The length of a line should be configurable.
You have 1 million text files, each is a news article scraped from various news sites. Since news sites often report the same news, even the same articles, many of the files have content very similar to each other. Write a program to filter out these files so that the end result contains only files that are sufficiently different from each other in the language of your choice. You’re free to choose a metric to define the “similarity” of content between files.
```

COMPLEXITY AND NUMERICAL ANALYSIS
```
Given that most of the recent breakthroughs in machine learning come from bigger models that require massive memory and computational power, it’s important to not only know how to implement a model but also how to scale it. To scale a model, we’d need to be able to estimate memory requirement and computational cost, as well as mitigate numerical instability when training and serving machine learning models. Here are some of the questions that can be asked to evaluate your understanding of numerical stability and scalability.

Matrix multiplication
[E] You have three matrices:  A∈R100×5,B∈R5×200,C∈R200×20  and you need to calculate the product  ABC . In what order would you perform your multiplication and why?
[M] Now you need to calculate the product of  N  matrices  A1A2...An . How would you determine the order in which to perform the multiplication?
[E] What are some of the causes for numerical instability in deep learning?
[E] In many machine learning techniques (e.g. batch norm), we often see a small term  ϵ  added to the calculation. What’s the purpose of that term?
[E] What made GPUs popular for deep learning? How are they compared to TPUs?
[M] What does it mean when we say a problem is intractable?
[H] What are the time and space complexity for doing backpropagation on a recurrent neural network?
[H] Is knowing a model’s architecture and its hyperparameters enough to calculate the memory requirements for that model?
[H] Your model works fine on a single GPU but gives poor results when you train it on 8 GPUs. What might be the cause of this? What would you do to address it?
[H] What benefits do we get from reducing the precision of our model? What problems might we run into? How to solve these problems?
[H] How to calculate the average of 1M floating-point numbers with minimal loss of precision?
[H] How should we implement batch normalization if a batch is spread out over multiple GPUs?
[M] Given the following code snippet. What might be a problem with it? How would you improve it? Hint: this is an actual question asked on StackOverflow.
import numpy as np

def within_radius(a, b, radius):
    if np.linalg.norm(a - b) < radius:
        return 1
    return 0

def make_mask(volume, roi, radius):
    mask = np.zeros(volume.shape)
    for x in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            for z in range(volume.shape[2]):
                mask[x, y, z] = within_radius((x, y, z), roi, radius)
    return mask
```


### ML WORKFLOWS

 BASICS

[E] Explain supervised, unsupervised, weakly supervised, semi-supervised, and active learning.
Empirical risk minimization.
[E] What’s the risk in empirical risk minimization?
[E] Why is it empirical?
[E] How do we minimize that risk?
[E] Occam's razor states that when the simple explanation and complex explanation both work equally well, the simple explanation is usually correct. How do we apply this principle in ML?
[E] What are the conditions that allowed deep learning to gain popularity in the last decade?
[M] If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why?
[H] The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range. Then why can’t a simple neural network reach an arbitrarily small positive error?
[E] What are saddle points and local minima? Which are thought to cause more problems for training large NNs?
Hyperparameters.
[E] What are the differences between parameters and hyperparameters?
[E] Why is hyperparameter tuning important?
[M] Explain algorithm for tuning hyperparameters.
Classification vs. regression.
[E] What makes a classification problem different from a regression problem?
[E] Can a classification problem be turned into a regression problem and vice versa?
Parametric vs. non-parametric methods.
[E] What’s the difference between parametric methods and non-parametric methods? Give an example of each method.
[H] When should we use one and when should we use the other?
[M] Why does ensembling independently trained models generally improve performance?
[M] Why does L1 regularization tend to lead to sparsity while L2 regularization pushes weights closer to 0?
[E] Why does an ML model’s performance degrade in production?
[M] What problems might we run into when deploying large machine learning models?
Your model performs really well on the test set but poorly in production.
[M] What are your hypotheses about the causes?
[H] How do you validate whether your hypotheses are correct?
[M] Imagine your hypotheses about the causes are correct. What would you do to address them?

 SAMPLING AND TRAINING DATA
[E] If you have 6 shirts and 4 pairs of pants, how many ways are there to choose 2 shirts and 1 pair of pants?
[M] What is the difference between sampling with vs. without replacement? Name an example of when you would use one rather than the other?
[M] Explain Markov chain Monte Carlo sampling.
[M] If you need to sample from high-dimensional data, which sampling method would you choose?
[H] Suppose we have a classification task with many classes. An example is when you have to predict the next word in a sentence -- the next word can be one of many, many possible words. If we have to calculate the probabilities for all classes, it’ll be prohibitively expensive. Instead, we can calculate the probabilities for a small set of candidate classes. This method is called candidate sampling. Name and explain some of the candidate sampling algorithms.

Hint: check out this great article on candidate sampling by the TensorFlow team.

Suppose you want to build a model to classify whether a Reddit comment violates the website’s rule. You have 10 million unlabeled comments from 10K users over the last 24 months and you want to label 100K of them.

[M] How would you sample 100K comments to label?
[M] Suppose you get back 100K labeled comments from 20 annotators and you want to look at some labels to estimate the quality of the labels. How many labels would you look at? How would you sample them?

Hint: This article on different sampling methods and their use cases might help.

[M] Suppose you work for a news site that historically has translated only 1% of all its articles. Your coworker argues that we should translate more articles into Chinese because translations help with the readership. On average, your translated articles have twice as many views as your non-translated articles. What might be wrong with this argument?

Hint: think about selection bias.

[M] How to determine whether two sets of samples (e.g. train and test splits) come from the same distribution?

[H] How do you know you’ve collected enough samples to train your ML model?
[M] How to determine outliers in your data samples? What to do with them?
Sample duplication
[M] When should you remove duplicate training samples? When shouldn’t you?
[M] What happens if we accidentally duplicate every data point in your train set or in your test set?
Missing data
[H] In your dataset, two out of 20 variables have more than 30% missing values. What would you do?
[M] How might techniques that handle missing data make selection bias worse? How do you handle this bias?
[M] Why is randomization important when designing experiments (experimental design)?
Class imbalance.
[E] How would class imbalance affect your model?
[E] Why is it hard for ML models to perform well on data with class imbalance?
[M] Imagine you want to build a model to detect skin legions from images. In your training dataset, only 1% of your images shows signs of legions. After training, your model seems to make a lot more false negatives than false positives. What are some of the techniques you'd use to improve your model?
Training data leakage.

[M] Imagine you're working with a binary task where the positive class accounts for only 1% of your data. You decide to oversample the rare class then split your data into train and test splits. Your model performs well on the test split but poorly in production. What might have happened?
[M] You want to build a model to classify whether a comment is spam or not spam. You have a dataset of a million comments over the period of 7 days. You decide to randomly split all your data into the train and test splits. Your co-worker points out that this can lead to data leakage. How?
Hint: You might want to clarify what oversampling here means. Oversampling can be as simple as dupplicating samples from the rare class.

[M] How does data sparsity affect your models?

Hint: Sparse data is different from missing data.

Feature leakage

[E] What are some causes of feature leakage?
[E] Why does normalization help prevent feature leakage?
[M] How do you detect feature leakage?
[M] Suppose you want to build a model to classify whether a tweet spreads misinformation. You have 100K labeled tweets over the last 24 months. You decide to randomly shuffle on your data and pick 80% to be the train split, 10% to be the valid split, and 10% to be the test split. What might be the problem with this way of partitioning?
[M] You’re building a neural network and you want to use both numerical and textual features. How would you process those different features?
[H] Your model has been performing fairly well using just a subset of features available in your data. Your boss decided that you should use all the features available instead. What might happen to the training error? What might happen to the test error?

Hint: Think about the curse of dimensionality: as we use more dimensions to describe our data, the more sparse space becomes, and the further are data points from each other.

 OBJECTIVE FUNCTIONS, METRICS, EVALUATIONS
 Convergence.
[E] When we say an algorithm converges, what does convergence mean?
[E] How do we know when a model has converged?
[E] Draw the loss curves for overfitting and underfitting.
Bias-variance trade-off
[E] What’s the bias-variance trade-off?
[M] How’s this tradeoff related to overfitting and underfitting?
[M] How do you know that your model is high variance, low bias? What would you do in this case?
[M] How do you know that your model is low variance, high bias? What would you do in this case?
Cross-validation.
[E] Explain different methods for cross-validation.
[M] Why don’t we see more cross-validation in deep learning?
Train, valid, test splits.

[E] What’s wrong with training and testing a model on the same data?
[E] Why do we need a validation set on top of a train set and a test set?
[M] Your model’s loss curves on the train, valid, and test sets look like this. What might have been the cause of this? What would you do?
Problematic loss curves
[E] Your team is building a system to aid doctors in predicting whether a patient has cancer or not from their X-ray scan. Your colleague announces that the problem is solved now that they’ve built a system that can predict with 99.99% accuracy. How would you respond to that claim?

F1 score.
[E] What’s the benefit of F1 over the accuracy?
[M] Can we still use F1 for a problem with more than two classes. How?
Given a binary classifier that outputs the following confusion matrix.

Predicted True	Predicted False
Actual True	30	20
Actual False	5	40
[E] Calculate the model’s precision, recall, and F1.
[M] What can we do to improve the model’s performance?
Consider a classification where 99% of data belongs to class A and 1% of data belongs to class B.
[M] If your model predicts A 100% of the time, what would the F1 score be? Hint: The F1 score when A is mapped to 0 and B to 1 is different from the F1 score when A is mapped to 1 and B to 0.
[M] If we have a model that predicts A and B at a random (uniformly), what would the expected F1 be?
[M] For logistic regression, why is log loss recommended over MSE (mean squared error)?
[M] When should we use RMSE (Root Mean Squared Error) over MAE (Mean Absolute Error) and vice versa?
[M] Show that the negative log-likelihood and cross-entropy are the same for binary classification tasks.
[M] For classification tasks with more than two labels (e.g. MNIST with 10 labels), why is cross-entropy a better loss function than MSE?
[E] Consider a language with an alphabet of 27 characters. What would be the maximal entropy of this language?
[E] A lot of machine learning models aim to approximate probability distributions. Let’s say P is the distribution of the data and Q is the distribution learned by our model. How do measure how close Q is to P?
MPE (Most Probable Explanation) vs. MAP (Maximum A Posteriori)
[E] How do MPE and MAP differ?
[H] Give an example of when they would produce different results.
[E] Suppose you want to build a model to predict the price of a stock in the next 8 hours and that the predicted price should never be off more than 10% from the actual price. Which metric would you use?

Hint: check out MAPE.

```NOTES ON INFO THEORY AND ENTROPY
In case you need a refresh on information entropy, here's an explanation without any math.

Your parents are finally letting you adopt a pet! They spend the entire weekend taking you to various pet shelters to find a pet.

The first shelter has only dogs. Your mom covers your eyes when your dad picks out an animal for you. You don't need to open your eyes to know that this animal is a dog. It isn't hard to guess.

The second shelter has both dogs and cats. Again your mom covers your eyes and your dad picks out an email. This time, you have to think harder to guess which animal is that. You make a guess that it's a dog, and your dad says no. So you guess it's a cat and you're right. It takes you two guesses to know for sure what animal it is.

The next shelter is the biggest one of them all. They have so many different kinds of animals: dogs, cats, hamsters, fish, parrots, cute little pigs, bunnies, ferrets, hedgehogs, chickens, even the exotic bearded dragons! There must be close to a hundred different types of pets. Now it's really hard for you to guess which one your dad brings you. It takes you a dozen guesses to guess the right animal.

Entropy is a measure of the "spread out" in diversity. The more spread out the diversity, the header it is to guess an item correctly. The first shelter has very low entropy. The second shelter has a little bit higher entropy. The third shelter has the highest entropy.
```


### ML Algorithms


 NLP
 RNNs
[E] What’s the motivation for RNN?
[E] What’s the motivation for LSTM?
[M] How would you do dropouts in an RNN?
[E] What’s density estimation? Why do we say a language model is a density estimator?
[M] Language models are often referred to as unsupervised learning, but some say its mechanism isn’t that different from supervised learning. What are your thoughts?
Word embeddings.
[M] Why do we need word embeddings?
[M] What’s the difference between count-based and prediction-based word embeddings?
[H] Most word embedding algorithms are based on the assumption that words that appear in similar contexts have similar meanings. What are some of the problems with context-based word embeddings?
Given 5 documents:
 D1: The duck loves to eat the worm
 D2: The worm doesn’t like the early bird
 D3: The bird loves to get up early to get the worm
 D4: The bird gets the worm from the early duck
 D5: The duck and the birds are so different from each other but one thing they have in common is that they both get the worm
[M] Given a query Q: “The early bird gets the worm”, find the two top-ranked documents according to the TF/IDF rank using the cosine similarity measure and the term set {bird, duck, worm, early, get, love}. Are the top-ranked documents relevant to the query?
[M] Assume that document D5 goes on to tell more about the duck and the bird and mentions “bird” three times, instead of just once. What happens to the rank of D5? Is this change in the ranking of D5 a desirable property of TF/IDF? Why?
[E] Your client wants you to train a language model on their dataset but their dataset is very small with only about 10,000 tokens. Would you use an n-gram or a neural language model?
[E] For n-gram language models, does increasing the context length (n) improve the model’s performance? Why or why not?
[M] What problems might we encounter when using softmax as the last layer for word-level language models? How do we fix it?
[E] What's the Levenshtein distance of the two words “doctor” and “bottle”?
[M] BLEU is a popular metric for machine translation. What are the pros and cons of BLEU?
[H] On the same test set, LM model A has a character-level entropy of 2 while LM model A has a word-level entropy of 6. Which model would you choose to deploy?
[M] Imagine you have to train a NER model on the text corpus A. Would you make A case-sensitive or case-insensitive?
[M] Why does removing stop words sometimes hurt a sentiment analysis model?
[M] Many models use relative position embedding instead of absolute position embedding. Why is that?
[H] Some NLP models use the same weights for both the embedding layer and the layer just before softmax. What’s the purpose of this?


 # COMPUTER VISION
[M] For neural networks that work with images like VGG-19, InceptionNet, you often see a visualization of what type of features each filter captures. How are these visualizations created?

Hint: check out this Distill post on Feature Visualization.

Filter size.
[M] How are your model’s accuracy and computational efficiency affected when you decrease or increase its filter size?
[E] How do you choose the ideal filter size?
[M] Convolutional layers are also known as “locally connected.” Explain what it means.
[M] When we use CNNs for text data, what would the number of channels be for the first conv layer?
[E] What is the role of zero padding?
[E] Why do we need upsampling? How to do it?
[M] What does a 1x1 convolutional layer do?
Pooling.
[E] What happens when you use max-pooling instead of average pooling?
[E] When should we use one instead of the other?
[E] What happens when pooling is removed completely?
[M] What happens if we replace a 2 x 2 max pool layer with a conv layer of stride 2?
[M] When we replace a normal convolutional layer with a depthwise separable convolutional layer, the number of parameters can go down. How does this happen? Give an example to illustrate this.
[M] Can you use a base model trained on ImageNet (image size 256 x 256) for an object classification task on images of size 320 x 360? How?
[H] How can a fully-connected layer be converted to a convolutional layer?
[H] Pros and cons of FFT-based convolution and Winograd-based convolution.

Hint: Read Fast Algorithms for Convolutional Neural Networks (Andrew Lavin and Scott Gray, 2015)


 OTHER
[M] An autoencoder is a neural network that learns to copy its input to its output. When would this be useful?
Self-attention.
[E] What’s the motivation for self-attention?
[E] Why would you choose a self-attention architecture over RNNs or CNNs?
[M] Why would you need multi-headed attention instead of just one head for attention?
[M] How would changing the number of heads in multi-headed attention affect the model’s performance?
Transfer learning
[E] You want to build a classifier to predict sentiment in tweets but you have very little labeled data (say 1000). What do you do?
[M] What’s gradual unfreezing? How might it help with transfer learning?
Bayesian methods.
[M] How do Bayesian methods differ from the mainstream deep learning approach?
[M] How are the pros and cons of Bayesian neural networks compared to the mainstream neural networks?
[M] Why do we say that Bayesian neural networks are natural ensembles?
GANs.
[E] What do GANs converge to?
[M] Why are GANs so hard to train?

 ***TRAINING
[E] When building a neural network, should you overfit or underfit it first?
[E] Write the vanilla gradient update.
Neural network in simple Numpy.
[E] Write in plain NumPy the forward and backward pass for a two-layer feed-forward neural network with a ReLU layer in between.
[M] Implement vanilla dropout for the forward and backward pass in NumPy.
Activation functions.
[E] Draw the graphs for sigmoid, tanh, ReLU, and leaky ReLU.
[E] Pros and cons of each activation function.
[E] Is ReLU differentiable? What to do when it’s not differentiable?
[M] Derive derivatives for sigmoid function when is a vector.
[E] What’s the motivation for skip connection in neural works?
Vanishing and exploding gradients.
[E] How do we know that gradients are exploding? How do we prevent it?
[E] Why are RNNs especially susceptible to vanishing and exploding gradients?
[M] Weight normalization separates a weight vector’s norm from its gradient. How would it help with training?
[M] When training a large neural network, say a language model with a billion parameters, you evaluate your model on a validation set at the end of every epoch. You realize that your validation loss is often lower than your train loss. What might be happening?
[E] What criteria would you use for early stopping?
[E] Gradient descent vs SGD vs mini-batch SGD.
[H] It’s a common practice to train deep learning models using epochs: we sample batches from data without replacement. Why would we use epochs instead of just sampling data with replacement?
[M] Your model’ weights fluctuate a lot during training. How does that affect your model’s performance? What to do about it?
Learning rate.
[E] Draw a graph number of training epochs vs training error for when the learning rate is:
too high
too low
acceptable.
[E] What’s learning rate warmup? Why do we need it?
[E] Compare batch norm and layer norm.
[M] Why is squared L2 norm sometimes preferred to L2 norm for regularizing neural networks?
[E] Some models use weight decay: after each gradient update, the weights are multiplied by a factor slightly less than 1. What is this useful for?
It’s a common practice for the learning rate to be reduced throughout the training.
[E] What’s the motivation?
[M] What might be the exceptions?
Batch size.
[E] What happens to your model training when you decrease the batch size to 1?
[E] What happens when you use the entire training data in a batch?
[M] How should we adjust the learning rate as we increase or decrease the batch size?
[M] Why is Adagrad sometimes favored in problems with sparse gradients?
Adam vs. SGD.
[M] What can you say about the ability to converge and generalize of Adam vs. SGD?
[M] What else can you say about the difference between these two optimizers?
[M] With model parallelism, you might update your model weights using the gradients from each machine asynchronously or synchronously. What are the pros and cons of asynchronous SGD vs. synchronous SGD?
[M] Why shouldn’t we have two consecutive linear layers in a neural network?
[M] Can a neural network with only RELU (non-linearity) act as a linear classifier?
[M] Design the smallest neural network that can function as an XOR gate.
[E] Why don’t we just initialize all weights in a neural network to zero?
Stochasticity.
[M] What are some sources of randomness in a neural network?
[M] Sometimes stochasticity is desirable when training neural networks. Why is that?
Dead neuron.
[E] What’s a dead neuron?
[E] How do we detect them in our neural network?
[M] How to prevent them?
Pruning.
[M] Pruning is a popular technique where certain weights of a neural network are set to 0. Why is it desirable?
[M] How do you choose what to prune from a neural network?
[H] Under what conditions would it be possible to recover training data from the weight checkpoints?
[H] Why do we try to reduce the size of a big trained model through techniques such as knowledge distillation instead of just training a small model from the beginning?