# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 18:07:04 2024

@author: Mary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
os.getcwd()


data = pd.read_csv("Restaurant_Reviews.tsv" , sep = "\t" , quoting = 3)

# CLEANING THE TEXT

import re

# print(re.sub("[^a-zA-Z]",":",data["Review"][0]))

for i in range(0,1000):
    cleanText = re.sub("[^a-zA-Z]" , " " , data["Review"][i])
    cleanText = cleanText.lower()
    cleanText = cleanText.split()
    data["Review"][i] = cleanText

  
# STOPWORDS
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

"""
print(stopwords.words("english"))
print(type(stopwords.words("english")))

"""

all_stopwords = stopwords.words("english")
all_stopwords.remove("not")

for i in range(0,1000):
    liste = []
    for word in data["Review"][i]:
        if word in all_stopwords:
            continue
        else:
            liste.append(word)
        
    data["Review"][i] = liste
    
    
# STEMMING -> kelimelerin köklerini bulmak için
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

"""
print(ps.stem("waited"))

liste = data["Review"][0]
j = 0
for word in liste:
    stemmingWord = ps.stem(word)
    liste[j] = stemmingWord
    j+=1
    
print(liste)

"""

for i in range(0,1000):
    j = 0
    for word in data["Review"][i]:
        stemmingWord = ps.stem(word)
        data["Review"][i][j] = stemmingWord
        j +=1
        
        
# print(" ".join(data["Review"][0]))

for i in range(0,1000):
    joined = " ".join(data["Review"][i])
    data["Review"][i] = joined
    



#GORSELLESTIRME

from wordcloud import WordCloud

# pozıtıf ve negatıf incelemeleri ayırır.
positive_reviews = data[data['Liked'] == 1]['Review'].tolist()
negative_reviews = data[data['Liked'] == 0]['Review'].tolist()

# incelemeleri her kategori için tek bir dizede birleştirir.
positive_text = " ".join(positive_reviews)
negative_text = " ".join(negative_reviews)

# kelime bulutu oluşturma
plt.figure(figsize=(12, 6))

# olumlu incelemeler için kelime bulutu
plt.subplot(1, 2, 1)
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Positive Reviews')

# olumsuz incelemeler için
plt.subplot(1, 2, 2)
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Negative Reviews')

# grafiği göster
plt.tight_layout()
plt.show()





#BAG OF WORDS
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)

X = cv.fit_transform(data["Review"]).toarray()
y = data.iloc[:,-1].values


# TRAIN VE TEST SETLERINE BOLMEK
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,
                                                 random_state = 42)


# NAIVE BAYES MODELININ KURULMASI
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

# MODELI EGITMEK
classifier.fit(X_train, y_train)

# MODELIN TEST SET UZERINDE DENENMESI
y_pred = classifier.predict(X_test)

# CONFUSION MATRIX VE ACCURACY SCORE
from sklearn.metrics import confusion_matrix,accuracy_score

cm = confusion_matrix(y_test, y_pred)
AccuracyScoreNB = accuracy_score(y_test, y_pred) # 0.67 doğruluk oranı.



# SVM MODELININ KURULMASI
from sklearn.svm import SVC
classifierSVM = SVC(probability=True,kernel="rbf")

# MODELI EGITMEK
classifierSVM.fit(X_train, y_train)

# TEST SET UZERINDE DENEMESI
y_predSVM = classifierSVM.predict(X_test)

# CONFUSION MATRIX VE ACCURACY SCORE
cmSVM = confusion_matrix(y_test, y_predSVM)
AccuracyScoreSVM= accuracy_score(y_test, y_predSVM) # 0.75 doğruluk oranı










    