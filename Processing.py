import pandas as pd   #qtwayland5 install
import numpy as np
from sklearn.preprocessing import StandardScaler   #scikit-learn==1.3.2
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
import pickle

matplotlib.use('SVG')         

def iqr(col):
    return np.where(col <= (col.quantile(.5)-1.5*(col.quantile(.95)-col.quantile(.5))),(col.quantile(.5)-1.5*(col.quantile(.95)-col.quantile(.5))),
           np.where((col >= col.quantile(.95)+1.5*(col.quantile(.95)-col.quantile(.5))),(col.quantile(.95)+1.5*(col.quantile(.95)-col.quantile(.5))),col)
                    )    

def Columns(df):
    return df.columns

def process(df):
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns=['index'],inplace=True)
    df['Customer ID']=df['Customer ID'].astype('str')

# new attribute : Monetary
    
    df['Monetary']=df['Quantity']*df['Price']
    rfm_m=df.groupby("Customer ID")['Monetary'].sum()
    rfm_m=rfm_m.reset_index()


# New Attribute : Frequency
    rfm_f=df.groupby('Customer ID')['Invoice'].count()
    rfm_f=rfm_f.reset_index()


    rfm=pd.merge(rfm_f,rfm_m,on='Customer ID',how='inner')

# New Attribute : Recency

    df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'],format='%Y-%m-%d %H:%M:%S') 
    df['Recency']=df['InvoiceDate'].max()-df["InvoiceDate"]   
    rfm_r=df.groupby('Customer ID')['Recency'].min().dt.days
    rfm_r=rfm_r.reset_index()
    rfm=pd.merge(rfm,rfm_r,on='Customer ID',how='inner')
    rfm.rename(columns={"Invoice":"Frequency"},inplace=True)
    rfm=rfm[['Customer ID',	'Recency','Frequency','Monetary']]
# Removing Outliers using iqr method 
    for col in rfm.columns[1:]:
        rfm[col]=iqr(rfm[col])

    scaler=pickle.load(open("/home/aryan/Development/Data_Science/ML_DL_project/Customer_segmentation/Model/scaler.pkl",'rb'))
    for col in rfm.columns[1:]:
        rfm[col]=scaler.fit_transform(rfm[col].values.reshape(-1,1))

  # assign label
    kms=pickle.load(open("Model/model.pkl",'rb'))
    rfm["Customer_label"]=kms.predict(rfm)

# Saving csv file
    with open("Data/processed_df.csv",'w') as f:
        rfm.to_csv(f)

    silo(rfm)
    
    return rfm



#Using silhouette tecnique to get right number of clusters
def silo(df):

  fig,ax=plt.subplots(2,2,figsize=(15,10))
  
 
  for i in range(2,6):
         
         kms=KMeans(n_clusters=i,max_iter=100,init='k-means++',n_init=10)
         q,mod=divmod(i,2)
         vlz=SilhouetteVisualizer(kms,colors='yellowbrick',ax=ax[q-1][mod])
         vlz.fit(df)
 
  
  fig.savefig("assets/silo.png")
  
  

  
# training model
# def train_model(df):
