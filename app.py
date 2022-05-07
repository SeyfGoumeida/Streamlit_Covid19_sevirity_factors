from os import sep
import streamlit as st
from datetime import datetime
import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

#--------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_icon="microbe", page_title="Covid19 Severity app")
st.title("🦠 Covid-19 severity factors")
st.sidebar.title(" 🦠 Covid-19 severity factors")

with st.expander("ℹ️ - About this app", expanded=False):

    st.write(
        """     
Ce projet vise tout d’abord à identifier les profils de patients susceptibles de développer une forme grave de la maladie. Parallèlement à ce travail, il est également demander d’identifier les comorbidités principales en lien avec la COVID19. Plus précisémment, il s’agit d’identifier des marqueurs de prédictions d’intérêt pour la sévérité de la maladies
-   (1) Ce PPD a pour premier objectif d’analyser une corpus biomédical autour de la COVID19, et plus particulièrement des comorbidités et des facteurs de sévérité de la maladie.
-   (2) Le second objectif de ce projet est l’exploitation d’approche de fouille de texte et de NLP pour l’identification de ces facteurs et de leurs interactions..
	    """
    )
    
st.markdown("")
st.markdown("## 📄 Articles : ")
#--------------------------------------------------------------------------------
#--------------------------------Functions---------------------------------------
#--------------------------------------------------------------------------------

def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")

#---------------------------------------------------------------------------------
# Key words
option = st.sidebar.selectbox(
     'Choose words to creat dataset :',
     #('Covid19 & Severity',
      ('Covid19 & Severity & Obesity',
      'Covid19 & Severity & Asthma',
      'Covid19 & Severity & Cancer',
      'Covid19 & Severity & Pneumonia',
      'Covid19 & Severity & Diabetes',
      'Covid19 & Severity & Hypertension')
     )
st.sidebar.write('You selected   :', option)

#---------------------------------------------------------------------------------
# number of articles to fetch
nb = st.sidebar.slider('choose the number of articles to fetch', 0,10000,500)
st.sidebar.write("You choosed    :", nb, ' articles')
space(1)

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------"Covid19 & Severity & Asthma"-----------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
  
if (option=="Covid19 & Severity & Asthma"):
  
  df = pd.read_csv("./text_cleaned_asthma_alldf.csv") 
  #df = pd.read_csv("./PPD 2022/Datasets/Covid19 & Severity & Asthma/text_cleaned_asthma_alldf.csv") 
  df.drop(columns="Unnamed: 0",inplace=True) 
  linkTopWords = "../PPD 2022/Datasets/Covid19 & Severity & Asthma/top_words_asthma_cocluster_"
  nbClusters = 9
  expander = st.expander("See all articles :", expanded=True)
  with expander:
    st.dataframe(df.head(nb))
  #--------------------------------------------------------------------------------------------------------------------------------
  st.header("")
  st.markdown("## 📊 Clusters : ")
  expander = st.expander("Clusters size :", expanded=False)
  with expander:
    image = st.image("../PPD 2022/Datasets/Covid19 & Severity & Asthma/clusters_size_asthma.png")
  selectedCluster = st.selectbox('Select the cluster number  :',('1', '2', '3','4','5','6','7','8','9'))
  linkSimilarity = "../PPD 2022/Datasets/Covid19 & Severity & Asthma/clusters_size_asthma.png"
  linkNER = "../PPD 2022/Datasets/Covid19 & Severity & Asthma/clusters_size_asthma.png"
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------"Covid19 & Severity & Cancer"-----------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------  
elif (option=="Covid19 & Severity & Cancer"):
  df = pd.read_csv("../PPD 2022/Datasets/Covid19 & Severity & Cancer/text_cleaned_cancer_alldf.csv")  
  linkTopWords = "../PPD 2022/Datasets/Covid19 & Severity & Cancer/top_words_cancer_cocluster_"
  nbClusters = 3
  df.drop(columns="Unnamed: 0",inplace=True)
  expander = st.expander("See all articles :", expanded=True)
  with expander:
    st.dataframe(df.head(nb))
  #--------------------------------------------------------------------------------------------------------------------------------
  st.header("")
  st.markdown("## 📊 Clusters : ")
  expander = st.expander("Cluster sizes :", expanded=False)
  with expander:
    image = st.image("../PPD 2022/Datasets/Covid19 & Severity & Cancer/clusters_size_cancer.png")
  selectedCluster = st.selectbox('Select the cluster number  :',('1', '2', '3'))
  linkSimilarity = "../PPD 2022/Datasets/Covid19 & Severity & Cancer/clusters_size_cancer.png"
  linkNER = "../PPD 2022/Datasets/Covid19 & Severity & Cancer/clusters_size_cancer.png"

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------"Covid19 & Severity & Diabetes"-----------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------- 
elif (option=="Covid19 & Severity & Diabetes"):
  df = pd.read_csv("../PPD 2022/Datasets/Covid19 & Severity & Diabetes/text_cleaned_diabetes_alldf.csv")
  linkTopWords = "../PPD 2022/Datasets/Covid19 & Severity & Diabetes/top_words_diabetes_cocluster_"
  nbClusters = 5  
  df.drop(columns="Unnamed: 0",inplace=True)
  expander = st.expander("See all articles :", expanded=True)
  with expander:
    st.dataframe(df.head(nb))
  #--------------------------------------------------------------------------------------------------------------------------------
  st.header("")
  st.markdown("## 📊 Clusters : ")
  expander = st.expander("Cluster sizes :", expanded=False)
  with expander:
    image = st.image("../PPD 2022/Datasets/Covid19 & Severity & Diabetes/clusters_size_diabetes.png")
  selectedCluster = st.selectbox('Select the cluster number  :',('1', '2', '3','4','5'))
  linkSimilarity = "../PPD 2022/Datasets/Covid19 & Severity & Diabetes/clusters_size_diabetes.png"
  linkNER =  "../PPD 2022/Datasets/Covid19 & Severity & Diabetes/clusters_size_diabetes.png"

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------"Covid19 & Severity & Hypertension"-----------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
elif (option=="Covid19 & Severity & Hypertension"):
  df = pd.read_csv("../PPD 2022/Datasets/Covid19 & Severity & Hypertension/text_cleaned_hypertension_alldf.csv")
  linkTopWords = "../PPD 2022/Datasets/Covid19 & Severity & Hypertension/top_words_hypertension_cocluster_"
  nbClusters = 5
  df.drop(columns="Unnamed: 0",inplace=True)
  expander = st.expander("See all articles :", expanded=True)
  with expander:
    st.dataframe(df.head(nb))
  #--------------------------------------------------------------------------------------------------------------------------------
  st.header("")
  st.markdown("## 📊 Clusters : ")
  expander = st.expander("Cluster sizes :", expanded=False)
  with expander:
    image = st.image("../PPD 2022/Datasets/Covid19 & Severity & Hypertension/clusters_size_hypertension.png")
  selectedCluster = st.selectbox('Select the cluster number  :',('1', '2', '3','4','5'))
  linkSimilarity = "../PPD 2022/Datasets/Covid19 & Severity & Hypertension/clusters_size_hypertension.png"
  linkNER = "../PPD 2022/Datasets/Covid19 & Severity & Hypertension/clusters_size_hypertension.png"

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------"Covid19 & Severity & Obesity"-----------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------  
elif (option=="Covid19 & Severity & Obesity"):
  df = pd.read_csv("../PPD 2022/Datasets/Covid19 & Severity & Obesity/text_cleaned_obesity_alldf.csv")  
  linkTopWords = "../PPD 2022/Datasets/Covid19 & Severity & Obesity/top_words_obesity_cocluster_"
  nbClusters = 4
  df.drop(columns="Unnamed: 0",inplace=True)
  expander = st.expander("See all articles :", expanded=True)
  with expander:
    st.dataframe(df.head(nb))
  #--------------------------------------------------------------------------------------------------------------------------------
  st.header("")
  st.markdown("## 📊 Clusters : ")
  expander = st.expander("Cluster sizes :", expanded=False)
  with expander:
    image = st.image("../PPD 2022/Datasets/Covid19 & Severity & Obesity/clusters_size_obesity.png")
  selectedCluster = st.selectbox('Select the cluster number  :',('1', '2', '3','4'))
  linkSimilarity = "../PPD 2022/Datasets/Covid19 & Severity & Obesity/clusters_size_obesity.png"
  linkNER = "../PPD 2022/Datasets/Covid19 & Severity & Obesity/clusters_size_obesity.png"

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------"Covid19 & Severity & Pneumonia"-----------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------- 
elif (option=="Covid19 & Severity & Pneumonia"):
  df = pd.read_csv("../PPD 2022/Datasets/Covid19 & Severity & Pneumonia/text_cleaned_pneumonia_alldf.csv")
  linkTopWords = "../PPD 2022/Datasets/Covid19 & Severity & Pneumonia/top_words_pneumonia_cocluster_"
  nbClusters = 5  
  df.drop(columns="Unnamed: 0",inplace=True)
  expander = st.expander("See all articles :", expanded=True)
  with expander:
    st.dataframe(df.head(nb))
  #--------------------------------------------------------------------------------------------------------------------------------
  st.header("")
  st.markdown("## 📊 Clusters : ")
  expander = st.expander("Cluster sizes :", expanded=False)
  with expander:
    image = st.image("../PPD 2022/Datasets/Covid19 & Severity & Pneumonia/clusters_size_pneumonia.png")
  selectedCluster = st.selectbox('Select the cluster number  :',('1', '2', '3','4','5'))
  linkSimilarity = "../PPD 2022/Datasets/Covid19 & Severity & Pneumonia/clusters_size_pneumonia.png"
  linkNER = "../PPD 2022/Datasets/Covid19 & Severity & Pneumonia/clusters_size_pneumonia.png"
#----------------------------------------------------------------------------------------------------------------------------
st.markdown("")
st.sidebar.write('You selected   :', option)
#----------------------------------------------------------------------------------------------------------------------------
c1, c2= st.columns([4,6])
expander = st.expander("See plot :")
with expander:
  link = linkTopWords+str(selectedCluster)+".csv"
  cluster = pd.read_csv(link)
  cluster = cluster.sort_values(by=['count'],ascending=False)
  #---------------------------------------------------------
  fig = plt.figure(figsize=(8, 2))
  ax = sns.barplot(x="words", y="count", data=cluster.head(20), alpha=0.9)
  ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
  with expander:
    c1.write(cluster)
  with expander:
    c2.pyplot(fig)
#----------------------------------------------------------------------------------------------------------------------------
st.header("")
st.markdown("## 🧬 Similarities : ")
expander2 = st.expander("Similarities  :", expanded=False)
with expander2:
  image2 = st.image(linkSimilarity)
#----------------------------------------------------------------------------------------------------------------------------
st.header("")
st.markdown("## 📌 NER Named Entity Recognition : ")
expander3 = st.expander("Similarities  :", expanded=False)
with expander3:
  image3 = st.image(linkNER)

#----------------------------------------------------------------------------------------------------------------------------
st.markdown("## 📥 Download Datasets and results :")
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv(sep=";").encode('utf-8')
dfcsv = convert_df(df)
clusterdf = convert_df(cluster)
c1, c2= st.columns([6, 6])
with c1:
  st.header("Dataset :")
  result = st.download_button(
      label="📥 Download (.csv)",
      data=dfcsv,
      file_name=option+'_df.csv',
      mime='text/csv',
  )
with c2:
  st.header("Cluster :")
  dataset = st.download_button(
      label="📥 Download(.csv)",
      data=clusterdf,
      file_name=option+selectedCluster+'_df.csv',
      mime='text_/csv',
  )
