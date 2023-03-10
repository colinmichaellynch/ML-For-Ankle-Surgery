# ML-For-Ankle-Surgery
Finding comorbidities that can result in adverse reactions to ankle surgery

## Table of Contents

* [Data compiled from National Surgical Quality Improvement Program (NSQIP) registry](https://github.com/colinmichaellynch/ML-For-Ankle-Surgery/blob/main/CombinedDatasetAnkle.csv) 

* [Machine learning code to predict patient outcomes](https://github.com/colinmichaellynch/ML-For-Ankle-Surgery/blob/main/mlModlesAnkle2.py)

## Background

Surgery can often result in seemingly random complications. I use supervised machine learning to predict whether a patient who has undergone ankle surgery will experience an adverse event within 30 days of the surgery. The goal of the algorithm is to determine which patient features are the most associated with adverse events so that these features can be controlled (if possible). For instance, if a patient underwent a blood transfusion 72 hours before a surgery, and this fact is closely tied with adverse events, then surgeries could be postponed for the sake of patient health. 

## Methods

* Patient data was collected from National Surgical Quality Improvement Program (NSQIP) registry 
  - Patient features include BMI, age, history of COPD, race, etc. 
  - Adverse events included stroke, sepsis, wound disruption, etc. If a pateint experienced at least one of these events, he/she was considered to have experienced an adverse event. 
  
* Data cleaning 
  - Predictor variables were removed if 80% or more of the values in that column were identical. 
  - Continuous variables were normalized
  - Categorical variables were tansformed with one-hot-encoding. 
  
* Algorithm development
  - Data was used to develop 6 machine learning models: a random forest, naive Bayes, neural network, support vector machine, k-nearest neighbors, and stochastic gradient boosting.
  - These models were tested against the performance of a multiple logistic regression model. 
  - Patients were randomly sorted into a training set (80%) and a test set (20%).
  - As the response variable is unbalanced (only 16.96% of patients experienced an adverse event), the training dataset was then upsampled so that there were an equal number of patients that both experienced and did not experience an adverse event.
  - Each model was trained with stratified 5-fold cross-validation and a grid search over all combinations of hyperparameters to maximize the average precision of each model.
  - Process was boostraped 100 times, where each bootstrap only had 50% of the validation data. 
  - I use permutation feature importance to measure the contribution of each patient attribute on the final ML model. 
  
## Results 
  
* Performance metrics for each model. Parentheses give 95% confidence intervals. 

<p align="center">
  <img src=/Images/performanceTable.png>
</p>

* Random forest performes the best, having the highest ROC AUC. It also outperforms the logistic regression for most metrics. 

<p align="center">
  <img src=/Images/rocauc.png>
</p>

* We also found the 5 most important variables for the random forest model and the logistic regression. The rank order of each variable differs between the logistic regression and the random forest, however, they both have the same top 4 variables, indicating that these are the most important predictors of an adverse event.
  - In vs outpatient, age, race, ASA Classification 
  
 * Older, non-white outpatients with more severe injuries are the individuals who are most likely to experience adverse events
  - Patients within these categories require more care than others. They should be treated as inpatients when possible, more advice should be given to them on how to treat their surgical wounds, and potentially racist practices need to be addressed within hospitals. For instance, clinical studies tend to preferentially focus on white demographics, so patient outcomes for minorities are not as well studied. 

## Acknowledgements

I would like to thank my collaborator Puneet Gupta for collecting this data and for developing the project idea. 
