import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde
from numpy import percentile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn import metrics
from statistics import mean
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import warnings 
warnings.filterwarnings("ignore")
anamolyDs =pd.read_csv("Anamoly.csv")
anamolyDs2 = anamolyDs.dropna(how='any',axis=0) 
anamolyDs2 = anamolyDs2.drop_duplicates()
anamolyDs2['dst_host_count']=anamolyDs2['dst_host_count'].fillna(0)
anamolyDs2['dst_host_srv_count']=anamolyDs2['dst_host_srv_count'].fillna(0)

cols = anamolyDs2.drop(columns=['class','protocol_type', 'service' , 'flag']).columns
for col in cols:
    q25, q75 = percentile(anamolyDs2[col], 25), percentile(anamolyDs2[col], 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    
    temp = anamolyDs2[(anamolyDs2[col] <= lower) | (anamolyDs2[col] >= upper)]
    for cls in temp['class'].unique():
        if temp[temp['class'] == cls].count()[0] < 20:
            #Removing outliers that are less in number breaking the IQR cutoff for the given feature
            anamolyDs2 = anamolyDs2[~anamolyDs2.isin(temp[temp['Class'] == cls])].dropna(how = 'all')

df = anamolyDs2.drop(columns=['class','protocol_type', 'service' , 'flag', 'num_outbound_cmds' , 'num_shells' , 'num_file_creations', 'num_root', 'su_attempted'  ])
cols = anamolyDs2.drop(columns=['class','protocol_type', 'service' , 'flag', 'num_outbound_cmds' , 'num_shells' , 'num_file_creations', 'num_root', 'su_attempted' ]).columns
standardized = StandardScaler().fit_transform(df)
final_df = pd.DataFrame(standardized, columns = cols,index=anamolyDs2.index)
final_df['class'] = anamolyDs2['class']

le = preprocessing.LabelEncoder()
final_df['class'] = le.fit_transform(final_df['class'])
train_data2, test_data2,train_label2,test_label2 = train_test_split(final_df.iloc[:,:-1],final_df['class'], test_size=0.9, random_state=313)

    
# Read data into pandas dataframe
df=final_df
#Define Feature Matrix (X) and Label Array (y)
X=df.drop(['class'],axis=1)
y=df['class']
lr=RandomForestClassifier(n_estimators=100, n_jobs=1,random_state=8)
lr.fit(X,y)
print("=====================")
print(X)
print(y)
print("=====================")
#Serialize the model and save
import joblib
joblib.dump(lr, 'randomfs.pkl')
print("Random Forest Model Saved")
#Load the model
lr = joblib.load('randomfs.pkl')
# Save features from training
rnd_columns = list(train_data2.columns)
print(rnd_columns)
print("==>>>>>>>>>>>>")
joblib.dump(rnd_columns, 'rnd_columns.pkl')
print("Random Forest Model Colums Saved")