import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
data['Review'][0]
data['Review'][10]

#Remove all the punctuations,symbols,emojis and unwanted characters from the data
temp=re.sub('[^a-zA-Z]',' ',data['Review'][0])
temp             #returns copy

#Get the data in similar case. 
temp=temp.lower()
temp             #returns copy

#Remove all the unwanted words like prepositions,conjunctions,pronouns,determiners,etc
temp=temp.split()
temp

t=[word for word in temp if not word in set(stopwords.words('english'))]
t

#Stemming or Lemmatization
t1=[ps.stem(word) for word in temp if not word in set(stopwords.words('english'))]
del (t,t1)
temp=[ps.stem(word) for word in temp if not word in set(stopwords.words('english'))]

#Choose a ML model to represent the data. We'll use 'Bag of Words' model.
temp=' '.join(temp)

#Applying the same steps for the whole data
clean_reviews=[]
for i in range(1000):
    temp=re.sub('[^a-zA-Z]',' ',data['Review'][i])
    temp=temp.lower()       
    temp=temp.split()
    temp=[ps.stem(word) for word in temp if not word in set(stopwords.words('english'))]
    temp=' '.join(temp)
    clean_reviews.append(temp)
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=500)
X=cv.fit_transform(clean_reviews)
X=X.toarray()

y=data['Liked'].values

from sklearn.linear_model import LinearRegression
log_reg=LinearRegression()

from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()

from sklearn.tree import DecisionTreeClassifier
dtf=DecisionTreeClassifier()

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()

log_reg.fit(X,y)
knn.fit(X,y)
dtf.fit(X,y)
nb.fit(X,y)

log_reg.score(X,y)
knn.score(X,y)
dtf.score(X,y)
nb.score(X,y)

print(cv.get_feature_names())











 
