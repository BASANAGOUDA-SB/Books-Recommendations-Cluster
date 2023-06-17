import os
import docx2txt
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import warnings
warnings.filterwarnings('ignore')

import re
import string 
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import stopwords 
wnl = WordNetLemmatizer()

from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.probability import FreqDist



file_path = r"C:\Users\rocket elecrtonics\Desktop\project 2"
extracted_data = []
software_names = []
def extract_data(file_path):
    for file in os.listdir(file_path):
        if file == 'developer':
            final = os.path.join(file_path, file)
            for data in os.listdir(final):
                if data.endswith('.docx') :
                    final_path = os.path.join(final, data)
                    extracted_data.append(docx2txt.process(final_path))
                    software_names.append(file)
        elif file == 'Peoplesoft resumes':
            final = os.path.join(file_path, file)
            for data in os.listdir(final) :
                if data.endswith('.docx') :
                    final_path = os.path.join(final, data)
                    extracted_data.append(docx2txt.process(final_path))
                    software_names.append(file)
        elif file == 'SQL Developer Lightning insight' :
            final = os.path.join(file_path, file)
            for data in os.listdir(final) :
                if data.endswith('.docx'):
                     final_path = os.path.join(final, data)
                extracted_data.append(docx2txt.process(final_path))
                software_names.append(file)
        elif file == 'workday resumes':
            final = os.path.join(file_path, file)
            for data in os.listdir(final) :
                if data.endswith('.docx'):
                    final_path = os.path.join(final, data)
                extracted_data.append(docx2txt.process(final_path))
                software_names.append(file)
                
                    
extract_data(file_path)
extracted_data


resume_data=pd.DataFrame()
resume_data['content']=extracted_data
resume_data['result']=software_names



resume_data.to_csv('resume_data.csv',encoding='utf=8',index=True)

resume_data.head()

resume_content = pd.read_csv('resume_data.csv',encoding = 'utf-8')
resume_content





def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)  


def mostcommon_words(cleaned,i):
    tokenizer = RegexpTokenizer(r'\w+')
    words=tokenizer.tokenize(cleaned)
    mostcommon=FreqDist(cleaned.split()).most_common(i)
    return mostcommon


def display_wordcloud(mostcommon):
    wordcloud=WordCloud(width=1000, height=600, background_color='black').generate(str(mostcommon))
    a=px.imshow(wordcloud)    
    st.plotly_chart(a)



    
   




 
    
    



tfidf_vect=TfidfVectorizer()
X_train_tfidf=tfidf_vect.fit(resume_content['content'])

X_train_tfiresume_data_transform=X_train_tfidf.transform(resume_content['content'])

X_train, X_test, y_train, y_test = train_test_split(X_train_tfiresume_data_transform,resume_content['result'],test_size=0.5, random_state=30,shuffle=True)

from sklearn.metrics import accuracy_score as accuracy

gradientboosting = GradientBoostingClassifier()
gradientboosting.fit(X_train,y_train)
y_train_prediction = gradientboosting.predict(X_train)
y_test_pred = gradientboosting.predict(X_test)

accuracy(y_train,y_train_prediction)

import pickle
pickle_out = open("deploy.pkl", "wb")
pickle.dump(gradientboosting, pickle_out)
pickle_out.close()
