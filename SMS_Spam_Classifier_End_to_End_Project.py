#!/usr/bin/env python
# coding: utf-8

# In[265]:


import numpy as np
import pandas as pd
import csv


# In[266]:


df = pd.read_csv('spam.csv', sep='|',encoding='utf-8', errors='ignore')


# In[267]:


import chardet
with open('spam.csv', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result


# In[268]:


df = pd.read_csv('spam.csv',encoding='ISO-8859-1')
df


# In[269]:


df.head()


# In[270]:


df.shape


# In[271]:


# 1. Data Cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model Building
# 5. Evaluation
# 6. Improvement
# 7. Website
# 8. Deploy


# ## 1. Data Cleaning

# In[272]:


df.info()


# In[273]:


# drop last 3 coloumns


# In[274]:


df.drop(columns = ['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace = True)


# In[275]:


df.sample(5)


# In[276]:


# renaming the columns


# In[277]:


df.rename(columns = {'v1':'target','v2':'text'}, inplace =True)
df.sample(5)


# In[278]:


from sklearn.preprocessing  import LabelEncoder
encoder = LabelEncoder()


# In[279]:


df['target'] = encoder.fit_transform(df['target'])


# In[280]:


df.head()


# In[281]:


# missing values
df.isnull().sum()


# In[282]:


# cheak for duplicated values
df.duplicated().sum()


# In[283]:


# remove the duplicated
df = df.drop_duplicates(keep = 'first')


# In[284]:


df.duplicated().sum()


# In[285]:


df.shape


# ## 2. EDA

# In[286]:


df.head()


# In[287]:


df['target'].value_counts()


# In[288]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[289]:


# data is inbalanceed


# In[290]:


import nltk


# In[291]:


nltk.download('punkt')


# In[292]:


df['num_char'] = df['text'].apply(len)


# In[293]:


df.head()


# In[294]:


# number of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[295]:


df.head()


# In[296]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[297]:


df.head()


# In[298]:


df[['num_char','num_words','num_sentences']].describe()


# In[299]:


#ham
df[df['target'] ==0][['num_char','num_words','num_sentences']].describe()


# In[300]:


#spam
df[df['target'] ==1][['num_char','num_words','num_sentences']].describe()


# In[301]:


import seaborn as sns


# In[302]:


plt.figure(figsize = (12,6))
sns.histplot(df[df['target'] ==0]['num_char'])
sns.histplot(df[df['target'] ==1]['num_char'],color='red')


# In[303]:


plt.figure(figsize = (12,6))
sns.histplot(df[df['target'] ==0]['num_words'])
sns.histplot(df[df['target'] ==1]['num_words'],color='red')


# In[304]:


sns.pairplot(df,hue='target')


# In[305]:


sns.heatmap(df.corr(),annot=True)


# # 3. Data Preprocessing
# #### Lower case
# #### Tokenization
# #### Removing special characters
# #### Removing stop words and punctution
# #### Stemming

# In[306]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[307]:


transform_text('I LOVED THE YOU-TUBE LECTURES ON MACHINE LEARNING. HOW ABOUT YOU')


# In[308]:


nltk.download('stopwords')


# In[309]:


from nltk.corpus import stopwords
stopwords.words("english")


# In[310]:


import string
string.punctuation


# In[311]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('Fucking')


# In[312]:


df['transfromed_test'] = df['text'].apply(transform_text)


# In[313]:


df.head()


# In[314]:


# Creating the word cloud
get_ipython().system('pip install wordcloud')


# In[315]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[316]:


spam_wc = wc.generate(df[df['target'] ==1]['transfromed_test'].str.cat(sep = " "))


# In[317]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[318]:


ham_wc = wc.generate(df[df['target'] ==0]['transfromed_test'].str.cat(sep = " "))


# In[319]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[320]:


spam_corpus = []
for msg in df[df['target'] ==1]['transfromed_test'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[321]:


len(spam_corpus)


# In[322]:


from collections import Counter


# In[323]:


sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[324]:


ham_corpus = []
for msg in df[df['target'] ==0]['transfromed_test'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[325]:


len(ham_corpus)


# In[326]:


sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# ## 4. Model Building

# In[327]:


df.head()


# In[328]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features = 3000)


# In[329]:


X = tfidf.fit_transform(df['transfromed_test']).toarray()


# In[330]:


X.shape


# In[331]:


y = df['target'].values


# In[332]:


y


# In[333]:


y.shape


# In[334]:


from sklearn.model_selection import train_test_split


# In[335]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[336]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix


# In[337]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[338]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[339]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[340]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[341]:


# tfidf --->mnb


# In[342]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[343]:


svc = SVC(kernel='sigmoid',gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty = 'l1')
rfc = RandomForestClassifier(n_estimators=50, random_state = 2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc  = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)


# In[222]:


clfs = {
    'SVC': svc,
    'KN' : knc,
    'NB' : mnb,
    'DT' : dtc,
    'LR' : lrc,
    'RF' : rfc,
    'Adaboost' : abc,
    'BgC' : bc,
    'ETC' : etc,
    'GBDT': gbdt,
    'xgb' : xgb,    
}


# In[223]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[224]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[225]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    current_accuracy,current_precision = train_classifier(clf,X_train,y_train,X_test,y_test)
    
    print("For",name)
    print("Accuracy",current_accuracy)
    print("Precision", current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[226]:


performance_df = pd.DataFrame({'Algorithm': clfs.keys(), 'Accuracy':accuracy_scores, 'Precision' :precision_scores}).sort_values('Precision', ascending=False)


# In[227]:


performance_df


# In[228]:


performance_df1 = pd.melt(performance_df,id_vars = "Algorithm")


# In[229]:


performance_df1


# In[230]:


sns.catplot(x = "Algorithm", y = 'value',
            hue = 'variable',data=performance_df1, kind ='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation = 'vertical')
plt.show()


# In[231]:


# model improvement
# 1. change the max_featues paramets of TfIdf


# In[235]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


# In[236]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)


# In[237]:


new_df = performance_df.merge(temp_df,on='Algorithm')


# In[238]:


new_df_scaled = new_df.merge(temp_df,on='Algorithm')


# In[239]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)


# In[240]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[241]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0, probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[243]:


voting = VotingClassifier(estimators = [('svm', svc), ('nb', mnb), ('et', etc)],voting = 'soft')


# In[244]:


voting.fit(X_train,y_train)


# In[245]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[246]:


# Applying stacking
estimetors = [('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator = RandomForestClassifier()


# In[247]:


from sklearn.ensemble import StackingClassifier


# In[248]:


clf = StackingClassifier(estimators=estimetors, final_estimator=final_estimator)


# In[249]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[344]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




