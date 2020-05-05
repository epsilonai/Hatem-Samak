Dealing with Imbalanced Data
https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18


# Fraud model_Balanced
#you will need to extract the source file (../clsify/data/crditcard.zip)befor fitting the modul again
## Pre-processing Data



```python
#Import Libararies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

```


```python
#Read dataset
url="/mnt/1A746F7B746F5891/L U N I X/Machine learning/Final Project/Creditcardfraud/Dataset/creditcardfraud/creditcard.csv"
dataset=pd.read_csv(url)

#Display head of dataset
dataset.head()

#split into dataset_X and dataset_y
dataset_X=dataset.drop(["Time","Class"],axis=1)# Must Write axis=1
dataset_y=dataset.Class
```


```python

```


```python
#check imbalancing data
unfruad_count=dataset.Class[dataset.Class==0].count()
fruad_count=dataset.Class[dataset.Class==1].count()
percent_unfruad=unfruad_count/(unfruad_count+fruad_count)
print("The percentag of unfruad is : %3f" % percent_unfruad,\
      "\nCount of fruad is :%i"%fruad_count,"\nCount of unfruad is : %i"%unfruad_count)
```

    The percentag of unfruad is : 0.998273 
    Count of fruad is :492 
    Count of unfruad is : 284315



```python
#check distribution of fraud instance after splitting

count_trn=0
count_tst=0
all_trn=0
all_tst=0
for i in y_train :
    all_trn+=1
    if i==1:
        count_trn+=1
for j in y_test:
    all_tst+=1
    if j==1:
        count_tst+=1
print(all_trn,count_trn,all_tst,count_tst,count_trn+count_tst)
```

    227845 394 56962 98 492


# Balancing Data Using SMOT


```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
# Separate input features and target
# y = dataset.Class
# X = dataset.drop(["Class","Time"], axis=1)

# setting up testing and training sets
Xb_train, Xb_test, yb_train, yb_test = train_test_split(dataset_X, dataset_y, test_size=0.2, random_state=0)

sm = SMOTE(random_state=0)
Xb_train, yb_train = sm.fit_sample(Xb_train, yb_train)
```

## Algorithms


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Logistic Regression                  
lgst_bdata = LogisticRegression(solver='liblinear')
lgst_bdata.fit(Xb_train, yb_train)


#Random Forest Classifier
clf_RFCls=RandomForestClassifier()
clf_RFCls.fit(Xb_train,yb_train)

```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)



# Prediction


```python
pred_y_RFCls=clf_RFCls.predict(Xb_test)# Rondom Forest Classifier
pred_y_lgst=lgst_bdata.predict(Xb_test)#Logistic Regression
```


```python

```

# Metrices


```python
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

#precision score
print(precision_score(yb_test, pred_y_RFCls, average='micro'))
print(precision_score(yb_test, pred_y_lgst, average='micro'))


#recall score
print(recall_score(yb_test, pred_y_RFCls, average='micro'))
print(recall_score(yb_test, pred_y_lgst, average='micro'))


# #f_score
print(f1_score(yb_test, pred_y_RFCls, average='micro'))
print(f1_score(yb_test, pred_y_lgst, average='micro'))


#roc_auc_score
print(lgst_bdata.predict_proba(Xb_test))
print(clf_RFCls.predict_proba(Xb_test))





```

    0.9994733330992591
    0.9815139917839963
    0.9994733330992591
    0.9815139917839963
    0.9994733330992591
    0.9815139917839963
    [[0.91125927 0.08874073]
     [0.92604305 0.07395695]
     [0.93749888 0.06250112]
     ...
     [0.98490025 0.01509975]
     [0.88492465 0.11507535]
     [0.99380035 0.00619965]]
    [[1.   0.  ]
     [1.   0.  ]
     [0.98 0.02]
     ...
     [1.   0.  ]
     [0.99 0.01]
     [1.   0.  ]]


# Classification Report


```python
from sklearn.metrics import classification_report
```


```python
for clf, label in zip([clf_RFCls, lgst_bdata], ["Rondom Forest", 'Logistic Regression']):
    print("{} report:".format(label))
    y_pred = clf.predict(Xb_test)
    print(classification_report(yb_test, y_pred))
    print("\n---------------------\n")
```

    Rondom Forest report:
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00     56861
               1       0.85      0.85      0.85       101
    
        accuracy                           1.00     56962
       macro avg       0.93      0.93      0.93     56962
    weighted avg       1.00      1.00      1.00     56962
    
    
    ---------------------
    
    Logistic Regression report:
                  precision    recall  f1-score   support
    
               0       1.00      0.98      0.99     56861
               1       0.08      0.92      0.15       101
    
        accuracy                           0.98     56962
       macro avg       0.54      0.95      0.57     56962
    weighted avg       1.00      0.98      0.99     56962
    
    
    ---------------------
    


# Get Single Sample 


```python
#Read dataset
url="/mnt/1A746F7B746F5891/L U N I X/Machine learning/Final Project/Creditcardfraud/Dataset/creditcardfraud/creditcard.csv"
dataset=pd.read_csv(url)


fruad_s=dataset[dataset.Class==0].iloc[0].drop(["Time","Class"])
fruad_s_shap=np.reshape(fruad_s.values,(1,-1))
fruad_s_shap

unfruad_s=dataset[dataset.Class==0].iloc[0].drop(["Time","Class"])
unfruad_s_shap=np.reshape(unfruad_s.values,(1,-1))
unfruad_s_shap

# unfruad=dataset[dataset.Class==0].iloc[0].drop(["Time","Class"])
# # print(fruad_ar)
# # frd_shp=array(fruad_l).reshape(1,-1)
# # print(fruad_l)
# pred_y_RFCls=clf_RFCls.predict(fruad_l)# Rondom Forest Classifier
# pred_y_lgst=lgst_bdata.predict(fruad_l)#Logistic Regression
# pred_y_RFCls[0],pred_y_lgst[0]

# print(lgst_bdata.predict_proba(fruad_l)[:,0])
# proba=(clf_RFCls.predict_proba(fruad_l)[:,1][0])*100
# print("Confidence  is :%i precentage"%proba)
```




    array([[-1.35980713e+00, -7.27811733e-02,  2.53634674e+00,
             1.37815522e+00, -3.38320770e-01,  4.62387778e-01,
             2.39598554e-01,  9.86979013e-02,  3.63786970e-01,
             9.07941720e-02, -5.51599533e-01, -6.17800856e-01,
            -9.91389847e-01, -3.11169354e-01,  1.46817697e+00,
            -4.70400525e-01,  2.07971242e-01,  2.57905802e-02,
             4.03992960e-01,  2.51412098e-01, -1.83067779e-02,
             2.77837576e-01, -1.10473910e-01,  6.69280749e-02,
             1.28539358e-01, -1.89114844e-01,  1.33558377e-01,
            -2.10530535e-02,  1.49620000e+02]])


