from IPython import get_ipython;   
get_ipython().magic('reset -sf')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression
from scipy.stats import sem, t
from sklearn.inspection import permutation_importance
from sklearn import metrics
import matplotlib.pyplot as plt

# Set seed for reproducibility
SEED = 1
warnings.filterwarnings("ignore")

### Modeling/data decisions
scoringMethod = 'average_precision' #balanced accuracy, roc_auc, accuracy, average_precision
removalThreshold = .9
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

#k fold vs stratified kfold, regular vs stratified
kFold = 'stratified'
n_iterations = 5

scores = ['accuracy', 'brier_score', 'calibration_slope', 'calibration_intercept', 'verification_accuracy', 'verification_balanced_accuracy', 'auc_roc']

# Define functions

def confidence_interval(data, confidence=0.95):
    sem_val = sem(data)
    t_val = t.ppf(confidence, len(data)-1)
    interval = [np.mean(data) - t_val * sem_val, np.mean(data) , np.mean(data) + t_val * sem_val]
    
    return interval

def upsample_dataset(X_train, y_train): 
    X_train['y'] = y_train
    X1 = X_train[X_train['y'] == 1]
    X0 = X_train[X_train['y'] == 0]
    X1Resample = resample(X1, n_samples=len(X0),replace=True)
    X1Resample = pd.concat([X1Resample,X0])
    X_train = X1Resample.drop('y',axis=1)
    y_train = X1Resample.y
    
    return X_train, y_train

def run_simulations(X, y, SEED, classifier):

    validation_accuracy = list()
    validation_accuracy_std = list()
    verification_accuracy = list()
    verification_ba = list()
    roc_auc = list()
    calibration_slope = list()
    calibration_intercept = list()
    brier_score = list()
    
    for i in range(n_iterations):
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=SEED)
        X_train, y_train = upsample_dataset(X_train, y_train)
        X_train['y'] = y_train
        X_test['y'] = y_test
        n_size = int(len(X_train) * 0.50)
        train = resample(X_train, n_samples = n_size) 
    
        classifier.fit(train.iloc[:,:-1], train.iloc[:,-1]) #model.fit(X_train,y_train) 
        all_accuracies = cross_val_score(estimator=classifier, X=train.iloc[:,:-1], y=train.iloc[:,-1], cv=cv)
        
        validation_accuracy.append(np.mean(all_accuracies))
        validation_accuracy_std.append(np.std(all_accuracies))
        
        y_pred = classifier.predict(X_test.iloc[:,:-1]) 
        verification_accuracy.append(accuracy_score(X_test.iloc[:,-1], y_pred))
        verification_ba.append(balanced_accuracy_score(X_test.iloc[:,-1], y_pred))
        roc_auc.append(roc_auc_score(y_test, classifier.predict_proba(X_test.iloc[:,:-1])[:,1]))
        
        probs = classifier.predict_proba(X_test.iloc[:,:-1])[:,1]
        calY, calX = calibration_curve(y_test, probs, n_bins=10)
        calX = calX.reshape((-1, 1))
        model = LinearRegression()
        model.fit(calX, calY)
        calibration_slope.append(model.coef_[0])
        calibration_intercept.append(model.intercept_)
    
        brier_score.append(brier_score_loss(X_test.iloc[:,-1], classifier.predict_proba(X_test.iloc[:,:-1])[:,1]))
     
    validation_accuracy_vec = np.around(confidence_interval(validation_accuracy, confidence=0.95), decimals=4)
    validation_accuracy_std_vec = np.around(confidence_interval(validation_accuracy_std, confidence=0.95), decimals=4)
    verification_accuracy_vec = np.around(confidence_interval(verification_accuracy, confidence=0.95), decimals=4)
    verification_ba_vec = np.around(confidence_interval(verification_ba, confidence=0.95), decimals=4)
    roc_auc_vec = np.around(confidence_interval(roc_auc, confidence=0.95), decimals=4)
    calibration_slope_vec = np.around(confidence_interval(calibration_slope, confidence=0.95), decimals=4)
    calibration_intercept_vec = np.around(confidence_interval(calibration_intercept, confidence=0.95), decimals=4)
    brier_score_vec = np.around(confidence_interval(brier_score, confidence=0.95), decimals=4)
    
    return validation_accuracy_vec, validation_accuracy_std_vec, verification_accuracy_vec, verification_ba_vec, roc_auc_vec, calibration_slope_vec,  calibration_intercept_vec, brier_score_vec

# Load data 

df = pd.read_csv('C:/Users/user/Documents/Fiverr projects/Elbow Surgery/Final Folder/CombinedDatasetAnkle.csv')
df = df.drop('Unnamed: 0', axis=1)
res = df
for col in res.columns:
    sum = res[col].value_counts()
    if(any(sum > len(res)*removalThreshold)):
        res = res.drop(col,axis=1)

res = res.replace(-99, np.NaN)
res = res.replace('NULL', np.NaN)
res = res.fillna(0)
res = res.drop('AdverseEvents', axis=1)

### transforming variables 

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
dfNumeric  = res.select_dtypes(include=numerics)

X = dfNumeric
X = (X-X.min())/(X.max()-X.min()) 

dfCategorical = res.select_dtypes(include=["object"])
categorical_cols = list(dfCategorical.columns)
dfCategorical = pd.get_dummies(dfCategorical, columns = categorical_cols)

X = pd.concat([dfCategorical, X], axis=1, ignore_index=False)
y = df.AdverseEvents

### Random forest

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=SEED)
X_train, y_train = upsample_dataset(X_train, y_train)

grid_param = {
    'n_estimators': [300, 500, 700, 900, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False],
    'max_depth': [1, 2, 3, 4], 
    'min_samples_leaf': [1, 2, 5, 10, 100, 1000]
}

gd_sr = GridSearchCV(estimator=RandomForestClassifier(), param_grid=grid_param,scoring=scoringMethod, cv=cv, n_jobs=-1)
gd_sr.fit(X_train, y_train)
best_params_ = gd_sr.best_params_
print('Random Forest')
print(best_params_)

classifier = RandomForestClassifier(n_estimators=best_params_['n_estimators'], criterion = best_params_['criterion'], bootstrap = best_params_['bootstrap'], max_depth = best_params_['max_depth'], min_samples_leaf = best_params_['min_samples_leaf'], random_state=SEED)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

rf_validation_accuracy_vec, rf_validation_accuracy_std_vec, rf_verification_accuracy_vec, rf_verification_ba_vec, rf_roc_auc_vec, rf_calibration_slope_vec, rf_calibration_intercept_vec, rf_brier_score_vec = run_simulations(X, y, SEED, classifier)

plt.figure(0).clf()

y_pred = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Random Forest")
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 

rfModel = classifier

### Naive Bayes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=SEED)
X_train, y_train = upsample_dataset(X_train, y_train)

classifier = GaussianNB()
print('Naive Bayes')

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

nb_validation_accuracy_vec, nb_validation_accuracy_std_vec, nb_verification_accuracy_vec, nb_verification_ba_vec, nb_roc_auc_vec, nb_calibration_slope_vec, nb_calibration_intercept_vec, nb_brier_score_vec = run_simulations(X, y, SEED, classifier)

y_pred = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Naive Bayes")

### Neural Network

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=SEED)
X_train, y_train = upsample_dataset(X_train, y_train)

grid_param = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [
             (1,),(5,),(10,),(15,),(20,), (1,2),(5,2),(10,2),(15,2),(20,2)
             ],
    'alpha': [0.0001, .001, .01, 0.05],
    'learning_rate': ['constant','adaptive']
        }
       ]

gd_sr = GridSearchCV(estimator=MLPClassifier(), param_grid=grid_param,scoring=scoringMethod, cv=cv, n_jobs=-1)
gd_sr.fit(X_train, y_train)
best_params_ = gd_sr.best_params_

classifier = MLPClassifier(activation=best_params_['activation'], hidden_layer_sizes = best_params_['hidden_layer_sizes'], solver = best_params_['solver'], random_state=SEED)

print('Neural Network')
print(best_params_)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

nn_validation_accuracy_vec, nn_validation_accuracy_std_vec, nn_verification_accuracy_vec, nn_verification_ba_vec, nn_roc_auc_vec, nn_calibration_slope_vec, nn_calibration_intercept_vec, nn_brier_score_vec = run_simulations(X, y, SEED, classifier)

y_pred = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Neural Network")

### SVM 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=SEED)
X_train, y_train = upsample_dataset(X_train, y_train)

grid_param = {'C': [0.1, 100],
              'gamma': [1, 0.001],
              'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

gd_sr = GridSearchCV(estimator=SVC(), param_grid=grid_param,scoring=scoringMethod, cv=cv, n_jobs=-1)
gd_sr.fit(X_train, y_train)
best_params_ = gd_sr.best_params_
print('SVM')
print(best_params_)

classifier = SVC(C=best_params_['C'], gamma = best_params_['gamma'], kernel = best_params_['kernel'], random_state=SEED, probability = True)
all_accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=cv)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

svm_validation_accuracy_vec, svm_validation_accuracy_std_vec, svm_verification_accuracy_vec, svm_verification_ba_vec, svm_roc_auc_vec, svm_calibration_slope_vec, svm_calibration_intercept_vec, svm_brier_score_vec = run_simulations(X, y, SEED, classifier)

y_pred = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="SVM")

### k nearest neighbors

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=SEED)
X_train, y_train = upsample_dataset(X_train, y_train)

grid_param = {'n_neighbors': [3, 5, 11, 19],
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan']}

gd_sr = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=grid_param,scoring=scoringMethod, cv=cv, n_jobs=-1)
gd_sr.fit(X_train, y_train)
best_params_ = gd_sr.best_params_
print('knearest')
print(best_params_)

classifier = KNeighborsClassifier(n_neighbors=best_params_['n_neighbors'], weights = best_params_['weights'], metric = best_params_['metric'])

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

knn_validation_accuracy_vec, knn_validation_accuracy_std_vec, knn_verification_accuracy_vec, knn_verification_ba_vec, knn_roc_auc_vec, knn_calibration_slope_vec, knn_calibration_intercept_vec, knn_brier_score_vec = run_simulations(X, y, SEED, classifier)

y_pred = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="KNN")

### Gradient boosting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=SEED)
X_train, y_train = upsample_dataset(X_train, y_train)

grid_param  = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 3),
    "min_samples_leaf": np.linspace(0.1, 0.5, 3),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.75, 1.0],
    "n_estimators":[1, 2, 5, 10, 20, 100]
    }

gd_sr = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=grid_param,scoring=scoringMethod, cv=cv, n_jobs=-1)
gd_sr.fit(X_train, y_train)
best_params_ = gd_sr.best_params_
print('Gradient Boosting')
print(best_params_)

classifier = GradientBoostingClassifier(criterion = best_params_['criterion'], learning_rate =  best_params_['learning_rate'], loss = best_params_['loss'], max_depth = best_params_['max_depth'], max_features = best_params_['max_features'], min_samples_leaf = best_params_['min_samples_leaf'], min_samples_split = best_params_['min_samples_split'], n_estimators = best_params_['n_estimators'], subsample = best_params_['subsample'], random_state=SEED)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

gbt_validation_accuracy_vec, gbt_validation_accuracy_std_vec, gbt_verification_accuracy_vec, gbt_verification_ba_vec, gbt_roc_auc_vec, gbt_calibration_slope_vec, gbt_calibration_intercept_vec, gbt_brier_score_vec = run_simulations(X, y, SEED, classifier)

y_pred = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Stochastic Gradient Boosting")

### logistic regression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=SEED)
X_train, y_train = upsample_dataset(X_train, y_train)

classifier = LogisticRegression(solver='liblinear', random_state=0)
all_accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=cv)
print('Logistic Regression')

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

lr_validation_accuracy_vec, lr_validation_accuracy_std_vec, lr_verification_accuracy_vec, lr_verification_ba_vec, lr_roc_auc_vec, lr_calibration_slope_vec, lr_calibration_intercept_vec, lr_brier_score_vec = run_simulations(X, y, SEED, classifier)

y_pred = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Logistic Regression")

plt.legend()

### compile data

final_scores = {'Model': ['Random Forest', 'Naive Bayes', 'Neural Network', 'SVM', 'KNN', 'Gradient Boosted Trees', 'Logistic Regression'], 'Validation Accuracy': [str(rf_validation_accuracy_vec[1]) + ' (' + str(rf_validation_accuracy_vec[0]) + ', ' + str(rf_validation_accuracy_vec[2]) + ')', str(nb_validation_accuracy_vec[1]) + ' (' + str(nb_validation_accuracy_vec[0]) + ', ' + str(nb_validation_accuracy_vec[2]) + ')', str(nn_validation_accuracy_vec[1]) + ' (' + str(nn_validation_accuracy_vec[0]) + ', ' + str(nn_validation_accuracy_vec[2]) + ')', str(svm_validation_accuracy_vec[1]) + ' (' + str(svm_validation_accuracy_vec[0]) + ', ' + str(svm_validation_accuracy_vec[2]) + ')', str(knn_validation_accuracy_vec[1]) + ' (' + str(knn_validation_accuracy_vec[0]) + ', ' + str(knn_validation_accuracy_vec[2]) + ')', str(gbt_validation_accuracy_vec[1]) + ' (' + str(gbt_validation_accuracy_vec[0]) + ', ' + str(gbt_validation_accuracy_vec[2]) + ')', str(lr_validation_accuracy_vec[1]) + ' (' + str(lr_validation_accuracy_vec[0]) + ', ' + str(lr_validation_accuracy_vec[2]) + ')'], 'Validation Accuracy STD': [str(rf_validation_accuracy_std_vec[1]) + ' (' + str(rf_validation_accuracy_std_vec[0]) + ', ' + str(rf_validation_accuracy_std_vec[2]) + ')', str(nb_validation_accuracy_std_vec[1]) + ' (' + str(nb_validation_accuracy_std_vec[0]) + ', ' + str(nb_validation_accuracy_std_vec[2]) + ')', str(nn_validation_accuracy_std_vec[1]) + ' (' + str(nn_validation_accuracy_std_vec[0]) + ', ' + str(nn_validation_accuracy_std_vec[2]) + ')', str(svm_validation_accuracy_std_vec[1]) + ' (' + str(svm_validation_accuracy_std_vec[0]) + ', ' + str(svm_validation_accuracy_std_vec[2]) + ')', str(knn_validation_accuracy_std_vec[1]) + ' (' + str(knn_validation_accuracy_std_vec[0]) + ', ' + str(knn_validation_accuracy_std_vec[2]) + ')', str(gbt_validation_accuracy_std_vec[1]) + ' (' + str(gbt_validation_accuracy_std_vec[0]) + ', ' + str(gbt_validation_accuracy_std_vec[2]) + ')', str(lr_validation_accuracy_std_vec[1]) + ' (' + str(lr_validation_accuracy_std_vec[0]) + ', ' + str(lr_validation_accuracy_std_vec[2]) + ')'], 'Verification Accuracy': [str(rf_verification_accuracy_vec[1]) + ' (' + str(rf_verification_accuracy_vec[0]) + ', ' + str(rf_verification_accuracy_vec[2]) + ')', str(nb_verification_accuracy_vec[1]) + ' (' + str(nb_verification_accuracy_vec[0]) + ', ' + str(nb_verification_accuracy_vec[2]) + ')', str(nn_verification_accuracy_vec[1]) + ' (' + str(nn_verification_accuracy_vec[0]) + ', ' + str(nn_verification_accuracy_vec[2]) + ')', str(svm_verification_accuracy_vec[1]) + ' (' + str(svm_verification_accuracy_vec[0]) + ', ' + str(svm_verification_accuracy_vec[2]) + ')', str(knn_verification_accuracy_vec[1]) + ' (' + str(knn_verification_accuracy_vec[0]) + ', ' + str(knn_verification_accuracy_vec[2]) + ')', str(gbt_verification_accuracy_vec[1]) + ' (' + str(gbt_verification_accuracy_vec[0]) + ', ' + str(gbt_verification_accuracy_vec[2]) + ')', str(lr_verification_accuracy_vec[1]) + ' (' + str(lr_verification_accuracy_vec[0]) + ', ' + str(lr_verification_accuracy_vec[2]) + ')'], 'Verification BA': [str(rf_verification_ba_vec[1]) + ' (' + str(rf_verification_ba_vec[0]) + ', ' + str(rf_verification_ba_vec[2]) + ')', str(nb_verification_ba_vec[1]) + ' (' + str(nb_verification_ba_vec[0]) + ', ' + str(nb_verification_ba_vec[2]) + ')', str(nn_verification_ba_vec[1]) + ' (' + str(nn_verification_ba_vec[0]) + ', ' + str(nn_verification_ba_vec[2]) + ')', str(svm_verification_ba_vec[1]) + ' (' + str(svm_verification_ba_vec[0]) + ', ' + str(svm_verification_ba_vec[2]) + ')', str(knn_verification_ba_vec[1]) + ' (' + str(knn_verification_ba_vec[0]) + ', ' + str(knn_verification_ba_vec[2]) + ')', str(gbt_verification_ba_vec[1]) + ' (' + str(gbt_verification_ba_vec[0]) + ', ' + str(gbt_verification_ba_vec[2]) + ')', str(lr_verification_ba_vec[1]) + ' (' + str(lr_verification_ba_vec[0]) + ', ' + str(lr_verification_ba_vec[2]) + ')'], 'ROC AUC': [str(rf_roc_auc_vec[1]) + ' (' + str(rf_roc_auc_vec[0]) + ', ' + str(rf_roc_auc_vec[2]) + ')', str(nb_roc_auc_vec[1]) + ' (' + str(nb_roc_auc_vec[0]) + ', ' + str(nb_roc_auc_vec[2]) + ')', str(nn_roc_auc_vec[1]) + ' (' + str(nn_roc_auc_vec[0]) + ', ' + str(nn_roc_auc_vec[2]) + ')', str(svm_roc_auc_vec[1]) + ' (' + str(svm_roc_auc_vec[0]) + ', ' + str(svm_roc_auc_vec[2]) + ')', str(knn_roc_auc_vec[1]) + ' (' + str(knn_roc_auc_vec[0]) + ', ' + str(knn_roc_auc_vec[2]) + ')', str(gbt_roc_auc_vec[1]) + ' (' + str(gbt_roc_auc_vec[0]) + ', ' + str(gbt_roc_auc_vec[2]) + ')', str(lr_roc_auc_vec[1]) + ' (' + str(lr_roc_auc_vec[0]) + ', ' + str(lr_roc_auc_vec[2]) + ')'], 'Calibration Slope': [str(rf_calibration_slope_vec[1]) + ' (' + str(rf_calibration_slope_vec[0]) + ', ' + str(rf_calibration_slope_vec[2]) + ')', str(nb_calibration_slope_vec[1]) + ' (' + str(nb_calibration_slope_vec[0]) + ', ' + str(nb_calibration_slope_vec[2]) + ')', str(nn_calibration_slope_vec[1]) + ' (' + str(nn_calibration_slope_vec[0]) + ', ' + str(nn_calibration_slope_vec[2]) + ')', str(svm_calibration_slope_vec[1]) + ' (' + str(svm_calibration_slope_vec[0]) + ', ' + str(svm_calibration_slope_vec[2]) + ')', str(knn_calibration_slope_vec[1]) + ' (' + str(knn_calibration_slope_vec[0]) + ', ' + str(knn_calibration_slope_vec[2]) + ')', str(gbt_calibration_slope_vec[1]) + ' (' + str(gbt_calibration_slope_vec[0]) + ', ' + str(gbt_calibration_slope_vec[2]) + ')', str(lr_calibration_slope_vec[1]) + ' (' + str(lr_calibration_slope_vec[0]) + ', ' + str(lr_calibration_slope_vec[2]) + ')'], 'Calibration Intercept': [str(rf_calibration_intercept_vec[1]) + ' (' + str(rf_calibration_intercept_vec[0]) + ', ' + str(rf_calibration_intercept_vec[2]) + ')', str(nb_calibration_intercept_vec[1]) + ' (' + str(nb_calibration_intercept_vec[0]) + ', ' + str(nb_calibration_intercept_vec[2]) + ')', str(nn_calibration_intercept_vec[1]) + ' (' + str(nn_calibration_intercept_vec[0]) + ', ' + str(nn_calibration_intercept_vec[2]) + ')', str(svm_calibration_intercept_vec[1]) + ' (' + str(svm_calibration_intercept_vec[0]) + ', ' + str(svm_calibration_intercept_vec[2]) + ')', str(knn_calibration_intercept_vec[1]) + ' (' + str(knn_calibration_intercept_vec[0]) + ', ' + str(knn_calibration_intercept_vec[2]) + ')', str(gbt_calibration_intercept_vec[1]) + ' (' + str(gbt_calibration_intercept_vec[0]) + ', ' + str(gbt_calibration_intercept_vec[2]) + ')', str(lr_calibration_intercept_vec[1]) + ' (' + str(lr_calibration_intercept_vec[0]) + ', ' + str(lr_calibration_intercept_vec[2]) + ')'], 'Brier Score': [str(rf_brier_score_vec[1]) + ' (' + str(rf_brier_score_vec[0]) + ', ' + str(rf_brier_score_vec[2]) + ')', str(nb_brier_score_vec[1]) + ' (' + str(nb_brier_score_vec[0]) + ', ' + str(nb_brier_score_vec[2]) + ')', str(nn_brier_score_vec[1]) + ' (' + str(nn_brier_score_vec[0]) + ', ' + str(nn_brier_score_vec[2]) + ')', str(svm_brier_score_vec[1]) + ' (' + str(svm_brier_score_vec[0]) + ', ' + str(svm_brier_score_vec[2]) + ')', str(knn_brier_score_vec[1]) + ' (' + str(knn_brier_score_vec[0]) + ', ' + str(knn_brier_score_vec[2]) + ')', str(gbt_brier_score_vec[1]) + ' (' + str(gbt_brier_score_vec[0]) + ', ' + str(gbt_brier_score_vec[2]) + ')', str(lr_brier_score_vec[1]) + ' (' + str(lr_brier_score_vec[0]) + ', ' + str(lr_brier_score_vec[2]) + ')']}
final_scores = pd.DataFrame.from_dict(final_scores)

# sensitivity scores for lr and rf

feature_list = list(X.columns)
importances = np.absolute(classifier.coef_[0])
importances = list(importances)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

classifier = rfModel
feature_list = list(X.columns)
results = permutation_importance(classifier, X, y, scoring='accuracy')
importance = results.importances_mean
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))