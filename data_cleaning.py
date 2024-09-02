import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE


df=pd.read_csv(path of csv file ,low_memory=False)
df.columns = df.columns.str.strip()
df.Label.value_counts()


drop_columns = [ # this list includes all spellings across CIC NIDS datasets
    "Protocol,""Flow ID",
    'Fwd Header Length.1',
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Timestamp",
    "Unnamed: 0","Unnamed: 0.1","Unnamed: 0.2","Unnamed: 0.3", "Inbound", "SimillarHTTP", # CIC-DDoS other undocumented columns
    "Bwd PSH Flags","Fwd URG Flags", "Bwd URG Flags","FIN Flag Count","PSH Flag Count",
    "ECE Flag Count", "Fwd Avg Bytes/Bulk","Fwd Avg Packets/Bulk","Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate"
]
df.drop(columns=drop_columns, inplace=True, errors='ignore')
df.drop_duplicates(inplace=True)
df.reset_index(inplace=True, drop=True)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# retain the minority class instances and sample the majority class instances
df_minor = df2[(df2['Label']=='BENIGN')]
df_major = df2.drop(df_minor.index)

X = df_major.drop(['Label'],axis=1) 
y = df_major.iloc[:, -1].values.reshape(-1,1)
y=np.ravel(y)


# use k-means to cluster the data samples and select a proportion of data from each cluster
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=100, random_state=0).fit(X)

cols = list(df_major)
cols.insert(65, cols.pop(cols.index('Label')))
df_major = df_major.loc[:, cols]


def typicalSampling(group):
    name = group.name
    frac = 0.55
    return group.sample(frac=frac)

result = df_major.groupby(
    'klabel', group_keys=False
).apply(typicalSampling, include_groups=False)
result.shape

df =pd.concat([df_minor,result])
df.Label.value_counts()

df.to_csv('DDos_dataset.csv')
