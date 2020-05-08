# Chapter 1

## The machine learning landscape

*Tom Mitchell's* definition - 

```latex
A computer program is said to learn from experience E wrt to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E. 
```

### Types of machine learning - 

>1. Supervised vs Unsupervised learning
>2. Batch vs Online learning
>3. Instance-Based vc Model-Based learning

#### Supervised vs Unsupervised

This classification is primarily based on the fact whether systems are trained with any kind supervision (human or machine) or not. We have four sub categories for such models - 

> 1. ##### Supervised learning - 
>    The training set fed to the algorithm contains the desired solutions (labels). **Classification** and **Prediction (Regression)** are typical supervised learning tasks - Spam filters, predicting housing prices.
>
>    Some common supervised learning algorithms are - 
>
>    * k-Nearest Neighbour (KNN)
>    * Linear Regression
>    * Logistic Regression
>    * SVMs
>    * Decision Trees and Random Forests
>    * Neural Networks*
>
>    
>    
> 2. ##### Unsupervised learning - 
>    The training data in unlabelled and the system tries to learn without a teacher. Important Unsupervised learning algorithms include - 
>
>    * **Clustering** - segment users, group similar tastes in music. 
>      * *K-Means*
>      * *DBCSCAN*
>      * *Hierarchical Clustering Analysis (HCA)*
>    * **Anomaly detection and novelty detection** - credit card frauds, manufacturing defects
>      * *One class SVM*
>      * *Isolation Forest*
>    * **Visualization and dimensionality reduction** - get 2D/3D representation of your data and identify patterns
>      * *PCA*
>      * *Kernel PCA*
>      * *Locally Linear Embeddings (LLE)*
>      * *t-Distributed Stochastic Neighbor Embeddings (t-SNE)*
>    * **Association Rule learning** ([link](https://medium.com/machine-learning-researcher/association-rule-apriori-and-eclat-algorithm-4e963fa972a4)) - dig into large datasets and discover relations. Eg, keeping items close in supermarkets
>      * *Apriori*
>      * *Eclat*
>    
>    ***Dimensionality reduction*** helps to reduce the number of features required to train the model without losing too much information by mergin several correlated features. The process of finding this correlation and merging features together is called feature extraction. Often a good idea to reduce the dimension of your dataset before feeding it to any other ML model - speed, less memory intensive.
>    
>    ***Anomaly Detection*** - The system is shown mostly normal instances during training so it learns to recognize them. When given a new instance it can tell whether it is normal behaviour or an anomaly (or something novel). Novelty detection requires an extremely clean dataset devoid of any instances that we want our algorithm to flag as novel. 
>    
> 3. ##### Semi-Supervised learning -
>
>    Labeling data is costly. Algos that deal with partially labeled data. Combination of unsupervised and supervised algorithms. Used by Google Photos. Once you upload photos, it groups the photos of the same person in a cluster, you then manually enter the label(name) for that person. Examples include - ***Deep Belief networks (DBNs)*** based on restricted ***Boltzmann machines(RBMs)***
>    
>4. ##### Reinforcement Learning
> 
>   The learning system, called an *agent* observes the environment and selects and performs the *actions* and gets *rewards*/*penalties* in return. The agent then learns by itself what the best strategy is, called a *policy*
>    Examples include robot walking, Deepmind's Alpha Go
> 

#### Batch vs Online learning

The criterion is based on whether or not the system can learn incrementally from a stream of incoming data or not.

> 1. ##### Batch Learning
>
>    System in incapable of learning incrementally. System is trained using all available data offline. Then the model is put into production and runs without learning anymore. To improve the model, it's needs to be trained again on the **new** + **old** data and then put into production again(replace the old model). Needs a lot of IO, memory, computing resources. Cannot adapt to rapidly changing data - stock market, need something more reactive
>    
>2. ##### Online learning
> 
>   Should have been named ***incremental learning***. System can be trained incrementally by feeding data instances sequentially (in form of mini batches). Learning steps are fast and cheap. Good option if we have 
> 
>   * Fast chaning data - stock market
>    * Less computing resources - don't need to save old data
> 
>   **Out-of-core learning** - Training systems on huge datasets that can't fit into the main memory at once. The algorithm runs part of data, runs a training step on that data, and repeats unitl all the data is used.
> 
>   **Learning rate** - The rate at which the system adapts to new data. High learning rate means that system adapts to new data but forgets old data quickly - won't want the spam filter just to classify latest spams as spam, shouldn't forget how old spams looked like. A lower learning rate means high inertia and the system is less sensitive to new data (or outlier points).

#### Instance based vs Model based

Categorization is done based on the fact how they generalise predictions. 

> 1. ##### Instance based
>
>    Flag emails identical to the ones already labeled as spam. Define some similarity measure between mails - counting same words. Many words is common, email likely to be spam
>
> 2. ##### Model based
>
>    Build a model - regressor or classifier, with appropriate features and try to learn the weights associated with those features. 
>
>    * Study the data
>    * Select a model
>    * Train on training set - Find Thetas that minimize the cost function
>    * Infer the prediction on unseen data

#### Challenges for ML

There are two things that can go wrong - bad data or bad algorithm

> 1. ##### Bad data
>
>    1. Insufficient training data - [The unreasonable effectiveness of Data](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/35179.pdf)
>    2. Non-representative training data - Sampling bias
>    3. Irrelevant features - focus on feature selection, feature extraction
>       
>
> 2. ##### Bad Algorithm
>
>    1. Overfitting the training data
>       1. Simplify the model by selecting fewer params
>       2. Use regularization
>       3. Gather more data
>       4. Reduce noise in training data
>    2. Underfitting the training data
>       1. Select a powerful model with more params
>       2. Find better features
>       3. Reduce constraints on the model (less regularization)

#### Hyperparameter tuning and Model selection

Training the model on the training set and tuning the hyperparameters using the test set can lead to overfitting on the test set and hence the model fails to generalize. 
One way to solve this problem is to divide the dataset into three chunks - **training, validation and test**

* Training - Use it to train multiple models( eg. LR, SVM, KNN) with various hyperparameters on the training set
* Validation - Run those multiple models on the validation set and select the one with the best performance. Now retrain that model on the Training + Validation data (with the same hyperparameters)
* Test - Once you get the best model, run it on the test set to check for generalization error. 

##### [Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw)

Take a small chunk of validation data from the training data at any given time. Each model is evaluated once per validation set after it is trained on the rest of the data. By averaging out the evaluation we get a more accurate measure of the model. Drawback is that the training time is multiplied by the number of validation sets. 

