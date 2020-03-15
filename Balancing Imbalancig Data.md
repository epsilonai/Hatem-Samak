# The Effect of Impbalancing Data

When only about 8% of ther observation were balanced, we will achieve an accoracy of 92% if we predict 0 (0:92% observation , 1: 8% observation)

Example:
if we train model by logistic Regression and get accuracy , we can apporve the above paragraph.

use 
print(np.unique(pred_data)

# Technique of Balancing Data

## 1- Up sample Minority class

### another tactics is Create Synthetic Samples (Data Augmentation)
SMOTE algorithm (creating "new"samples)

###Combine Minority Calsses
for multi classes, we can transfer it to binary by combine them into single classe

###Reframe as Anomaly Detection
Anomaly detection, a.k.a. outlier detection, is for detecting outliers and rare events
instead of building a classification model , we'd a "profile" of normal observation . if a new observation strays too far from that "noraml profile" it flagged as anamoly.


```python
from sklearn.utils import resample
```


```python
#calculate the value_counts of each class
dataset["class"].value_counts()

#Separate observation from each class into different DataFrame
df_majority=dataset[dataset.class==0]
df_minority=dataset[dataset.class==1]

#upsample minority class:
df_minority_upsample=resample(df_minority,
                             replace=True # sample with replacement
                             n_samples= majorty_counts # to match majority class
                             random_state= 125) # reproducible results)

#Combine majority class with upsample minority class
df_upsample=pd.concat([df_majority, df_minority_upsample])

#Dispaly new class counts
df_upsample.class.value_counts()
```

## 2- Down sample of Majority Class 


```python
#Separate observation from each class into different DataFrame
df_majority=dataset[dataset.class==0]
df_minority=dataset[dataset.class==1]

#upsample minority class:
df_minority_upsample=resample(df_minority,
                             replace=False # sample without replacement
                             n_samples= minorty_counts # to match majority class
                             random_state= 125) # reproducible results)

#Combine majority class with upsample minority class
dataset_balanced=pd.concat([df_majority, df_minority_upsample])

#Dispaly new class counts
df_upsample.class.value_counts()
```


```python
#Train model on both downsampled and upsampled dataset

#Separate input feature (X) and target variable (y)
y=dataset_balanced.class
X=dataset_balanced.drop("class",axis=1)

#Train model
clf_2=logisticRegression().fit(X,y)

#Predict on training set
pred_y=clf_2.predict(X)

#Check if it predicting one or two classes
print(np.unique(pred_y))

# Accuracy

print(accuracy_score(y, pred_y))
```

## 3- Changing Your Performance Metric

we use this metrics after using the 2 above technique 


```python
#use Area Under ROC Curve (AUROC)
#it calculate probability that the model will be able to "rank" 
#correctly i we rondomly select one observation from each class

# we will need to predict class probalilityes instead of just the predicted classes using
# .predict_proba()
from sklearn.metrics import roc_auc_score

prob_y=clf_2.predict_proba(X)

#Keep only the positive class
prob_y=[p[1] for p in prob_y]

print(prob_y[:5])

#AUROC of model trained on downsampled dataset
print(roc_auc_score(y,prob_y))

#we can do thin on original model befor apply the technique and compare the results

datase_y=clf_0.prdict_proba(X)

#keep only the positive class
prob_y_0=clf_0.predict_proba()
prob_y_0=[p[1] for p in prob_y_0]

# get the scaore
print(roc_auc_score(y, prob_y_0))

#Notes if AUROC <= .5 we need to invert the prediction of positive class

```

## 4-  Penalize Algorithms (Cost-Sensitive Training)


```python
#the popular algorithm is the Penalized-SVM
```


```python
from sklearn.svm import SVC
```


```python
# during training we can use the argument class_weitht="balanced" to penalize mistake on the 
# minority class by an amount proportional to how under-represented is
# we also will inclode the argument probability="True"

#Separate input features (X) and target (y)


#Train model
clf_3=SVC(kernel="linear", class_weight="balanced", # penalize
         probability=True)

clf_3.fit(X,y)

#predict on training set
pred_y_3=clf_3.predict(X)

#check if predict one class of two

# Calculate accuracy
print(accuracy_score(y,pred_y_3))

#Calculate AUROC metric
prob_y_3=clf_3.predict_proba(X)

prob_y_3=[p[1] for p in prob_y_3]
print (roc_auc_score(y,prob_y_3))
```

## 5 - Use Tree-Based Algorithms


```python
#Descision trees oftem perform well on imbalnced dataset because their hierarchical structre
# allows them to learn signlas from both classes
#tree ensembles (Random Forests, Grandient Boosted Trees,etc.)
```


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
#Train Random Forest on imbalance dataset

# separate input features (X) target(y)
y=dataset.class
X=dataset.drop("balance",axis=1)

# Train model
clf_4=RandomForestClassifier()
clf_4.fit(X,y)

#Predict on training set
pred_y_4=clf_4.predic(X)

#check if we predict from the two classes or not
print(np.unique(pred_y_4))

#get accurcy
print(accuracy_score(y,pred_y_4))

#calcualte the AUROC
prob_y_4=clf_4.predict_proba(X)

prob_y_4=[p[1] for p in prob_y_4]
print (roc_auc_score(y,prob_y_4))
```
