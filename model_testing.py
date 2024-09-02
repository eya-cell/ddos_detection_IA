#import necessary  Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score,roc_auc_score
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# Load the models
stk= joblib.load('sids.pkl')
km_cluster= joblib.load('cluster_final.pkl')



#loading data
test_data=pd.read_csv('Portmap.csv', low_memory=False)
test_data.columns = test_data.columns.str.strip()
test_data.Label.value_counts()


features = ['Flow Duration','Total Length of Fwd Packets','Total Length of Bwd Packets','Total Fwd Packets','Total Backward Packets','Max Packet Length',
 'Min Packet Length','Packet Length Mean','Fwd Packet Length Mean','Fwd Packet Length Max','Fwd Packet Length Min','Bwd Packet Length Mean',
 'Bwd Packet Length Max','Bwd Packet Length Min']


# Load feature statistics from the JSON file
with open('feature_stats_2_final.json', 'r') as json_file:
    feature_stats = json.load(json_file)

# Apply normalization to test or real-life data
def normalize_features(data):
    for feature, stats in feature_stats.items():
        mean_value = stats['mean']
        std_value = stats['std']
        data[feature] = (data[feature] - mean_value) / std_value
    return data

# Apply normalization
test_data[features] = normalize_features(test_data[features])


#label_encoding
is_attack = test_data.Label.map(lambda a: 0 if a == 'BENIGN' else 1)
test_data['Label'] = is_attack

x_test = test_data.loc[:,features]
Y_test = test_data.loc[:, 'Label']


#Stacking performance evaluation
def Stacking(stk_pred, y_test):
      y_true=y_test
      stk_score=accuracy_score(y_true,stk_pred)
      print('Accuracy of Stacking: '+ str(stk_score))
      precision,recall,fscore,none= precision_recall_fscore_support(y_true, stk_pred, average='weighted')
      print('Precision of Stacking: '+(str(precision)))
      print('Recall of Stacking: '+(str(recall)))
      print('F1-score of Stacking: '+(str(fscore)))
      print(classification_report(y_true,stk_pred))
      cm=confusion_matrix(y_true,stk_pred)
      f,ax=plt.subplots(figsize=(5,5))
      sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
      plt.xlabel("y_pred")
      plt.ylabel("y_true")
      plt.show()

# Stacking predictions
stk_pred = stk.predict(x_test)



print("Stacking:")
Stacking(stk_pred, y_test)






#clustering performance evaluation


def Clustering(km_pred, y_test):

    # Assign labels to clusters based on majority vote
    cluster_labels = {}
    for cluster in np.unique(km_pred):
        cluster_indices = np.where(km_pred == cluster)[0]
        cluster_labels[cluster] = np.argmax(np.bincount(y_test[cluster_indices]))

    # Assign labels to the new data clusters
    new_data_labels = [cluster_labels[cluster] for cluster in km_pred]

    # Evaluation
    accuracy = accuracy_score(y_test, new_data_labels)
    print("Accuracy Score:", accuracy)
    precision,recall,fscore,none= precision_recall_fscore_support(y_test, new_data_labels, average='weighted')
    print('Precision of Clustering: '+(str(precision)))
    print('Recall of Clustering: '+(str(recall)))
    print('F1-score of Clustering: '+(str(fscore)))
    print(classification_report(y_test,new_data_labels))
    cm=confusion_matrix(y_test,new_data_labels)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    plt.show()

# Make predictions using each model
km_pred =model.predict(x_test)

# Print the evaluation metrics
print("\nClustering:")
Clustering(km_pred, Y_test)
