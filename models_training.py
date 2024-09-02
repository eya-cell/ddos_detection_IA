import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt

#import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.cluster import KMeans,SpectralClustering,MiniBatchKMeans
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


#data processing
df= pd.read_csv('/kaggle/input/dataseet/DDos_dataset.csv',low_memory=False)
df.columns= df.columns.str.strip()
df.shape
df.Label.value_counts()


#Z-score normalisation:
features = ['Flow Duration','Total Length of Fwd Packets','Total Length of Bwd Packets','Total Fwd Packets','Total Backward Packets','Max Packet Length',
 'Min Packet Length','Packet Length Mean','Fwd Packet Length Mean','Fwd Packet Length Max','Fwd Packet Length Min','Bwd Packet Length Mean',
 'Bwd Packet Length Max','Bwd Packet Length Min']
df[features] = df[features].apply(
   lambda x: (x - x.mean()) / (x.std()))
  # Fill empty values by 0
df = df.fillna(0)

is_attack = df.Label.map(lambda a: 0 if a == 'BENIGN' else 1)
df['Label'] = is_attack
df.Label.value_counts()


X = df.loc[:,features]
y = df.loc[:, 'Label']

#class_balance
from imblearn.over_sampling import SMOTE
smote=SMOTE(n_jobs=-1,random_state=0)
X, y = smote.fit_resample(X, y)

#splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#THE proposed Signature-based system

#XGBoost Model
xg = xgb.XGBClassifier(n_estimators = 250,random_state=0)
xg.fit(X_train,y_train)
xg_score=xg.score(X_test,y_test)
y_predict=xg.predict(X_test)
y_true=y_test
#Evaluation
print('Accuracy of XGBoost: '+ str(xg_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of XGBoost: '+(str(precision)))
print('Recall of XGBoost: '+(str(recall)))
print('F1-score of XGBoost: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


#Decision Tree Model
dt = DecisionTreeClassifier(random_state = 0)
dt.fit(X_train,y_train)
dt_score=dt.score(X_test,y_test)
y_predict=dt.predict(X_test)
y_true=y_test

#Evaluation
print('Accuracy of DT: '+ str(dt_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of DT: '+(str(precision)))
print('Recall of DT: '+(str(recall)))
print('F1-score of DT: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

dt_pred=dt.predict(X_test)


#Random Forest Model
rf = RandomForestClassifier(n_estimators = 250,random_state = 0)
rf.fit(X_train,y_train)
rf_score=rf.score(X_test,y_test)
y_predict=rf.predict(X_test)
y_true=y_test
#Evaluation
print('Accuracy of RF: '+ str(rf_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of RF: '+(str(precision)))
print('Recall of RF: '+(str(recall)))
print('F1-score of RF: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


#Extra Tree Model

#ET
et = ExtraTreesClassifier(n_estimators = 250,random_state = 0)
et.fit(X_train,y_train)
et_score=et.score(X_test,y_test)
y_predict=et.predict(X_test)
y_true=y_test

#Evaluation
print('Accuracy of ET: '+ str(et_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of ET: '+(str(precision)))
print('Recall of ET: '+(str(recall)))
print('F1-score of ET: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()



#Stacking model
# Define estimators
from sklearn.ensemble import StackingClassifier

estimators = [
    ('xg' ,xg),
    ('rf', rf),
    ('dt', dt),
    ('et', et)
]
# Build stack model
stk = StackingClassifier(
    estimators=estimators, final_estimator=xgb.XGBClassifier(random_state=0)
)
#train stacked model
stk.fit(X_train, y_train)
y_predict=stk.predict(X_test)
y_true=y_test
stk_score=accuracy_score(y_true,y_predict)
#Evaluation
print('Accuracy of Stacking: '+ str(stk_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted')
print('Precision of Stacking: '+(str(precision)))
print('Recall of Stacking: '+(str(recall)))
print('F1-score of Stacking: '+(str(fscore)))
print(classification_report(y_true,y_predict))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()



#anomaly-based system

#elbow method 
wcss = []
for i in range(1, 17):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 17), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#Clustering
# Initialization
km_cluster = MiniBatchKMeans(n_clusters=10, batch_size=100, random_state=0)

# Clustering
train_clusters = km_cluster.fit_predict(X_train)
test_clusters = km_cluster.predict(X_test)

# Label Assignment
cluster_labels = {}
for cluster in np.unique(train_clusters):
        cluster_indices = np.where(train_clusters == cluster)[0]
        cluster_labels[cluster] = np.argmax(np.bincount(y_train.iloc[cluster_indices]))

# Majority Vote
test_predictions = [cluster_labels[cluster] for cluster in test_clusters]


# Evaluation
accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy Score:", accuracy)
precision,recall,fscore,none= precision_recall_fscore_support(y_test, test_predictions, average='weighted')
print('Precision of Clustering: '+(str(precision)))
print('Recall of Clustering: '+(str(recall)))
print('F1-score of Clustering: '+(str(fscore)))
print(classification_report(y_test, test_predictions))
cm=confusion_matrix(y_test, test_predictions)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

#Cluster Mapping
print("Cluster-to-Label Mapping:")
for cluster_id, label in cluster_labels.items():
        print(f"Cluster {cluster_id}: Label {label}")



# Save the dictionary containing all models to a single file
import joblib

tree_based= {
    'xg': xg,
    'rf': rf,
    'dt': dt,
    'et': et
}

joblib.dump(tree_based, 'tree.pkl')

joblib.dump(stk, 'sids.pkl')

joblib.dump(km_cluster, 'cluster_final.pkl')

feature_stats = {}
for feature in features:
    mean_value = df[feature].mean()
    std_value = df[feature].std()
    feature_stats[feature] = {'mean': mean_value, 'std': std_value}

# Save feature statistics to a JSON file
with open('feature_stats_final.json', 'w') as json_file:
    json.dump(feature_stats, json_file)


