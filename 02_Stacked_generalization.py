"""
Download data from here
https://www.kaggle.com/c/bioresponse/data
"""

from __future__ import division
import numpy as np
#import load_data loading load_data.py code
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':

    np.random.seed(0) # seed to shuffle the train set

    n_folds = 3
    verbose = True

#Calling load() function from load_data.py to import the data
    X, y belong to &  X_submission is test data
    X, y, X_submission = load()

#Provides indices to split data into k folds
#StratifiedKFold() will return 2 arrays for each iteration of cross-validation. 1st array represent all k-1 number of folds 
#& 2nd array is the kth fold
# We will build model in 1st array & make predictions using this model on the 2nd array

    skf = list(StratifiedKFold(y, n_folds,shuffle=True))
    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
           # RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            #ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate =0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    print ("Creating train and test sets for blending.")
#Creating empty arrays & initialise it with 0s
#This will have the probabilities for each classifier
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

#This outer loop is for iterating through each of the classifier
    for j, clf in enumerate(clfs):
        print (j, clf)
# This holds the probabilities of the test data predicted from each of the k models from the K fold
# We will avg to get a single probability in the end        

        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (rest, ithfold) in enumerate(skf):
            
            print ("Fold", i)
            print ("Length of rest",len(rest))
            print ("Length of ithfold",len(ithfold))
            print ()

            X_rest = X[rest]
            y_rest = y[rest]
            X_ithfold = X[ithfold]
            y_ithfold = y[ithfold]
#Building model on rest (k-1) folds
            clf.fit(X_rest, y_rest)
#predicting probability on ith fold
            y_submission = clf.predict_proba(X_ithfold)[:,1]
            dataset_blend_train[ithfold, j] = y_submission
#Using the same model predicting probabilities on test data
# Note: For the same classifier for each of the k folds we will get different models since data for building is 
# different each time. We will score test data based on each of the k models mentioned above, so for each classifier 
# we will get k probabilities. In the last step we will take avg of probabilities to get 1 probability for each classifier
            
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
#Averaging the probability from each k folds after the iterations is over to get 1 probability per classifier
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
        
    print ()
    print ("Blending.")
#Note we are using logistic regression here for blending but 1 drawback of logistic regression is it doesn't support
#different loss functions like AUC or logloss in scikit-learn. So to directly optimise the loss we may have to use 
# a different classifier
# Note: If we are using a different classifier we can again use CV to cross validate & tune the model on this new train & test data
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    clf.coef_ 
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

#    print ("Linear stretch of predictions to [0,1]")
#    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print ("Saving Results.")
    np.savetxt(fname='test.csv', X=y_submission, fmt='%0.9f')
