import pandas as pd
import numpy as np
import sklearn
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, roc_auc_score

# load in data files
fake_df=pd.read_csv('input/Fake.csv')
true_df=pd.read_csv('input/True.csv')

# fake coded to 1
fake_df["Label"]=1
true_df["Label"]=0

# combine df into a single df
df=pd.concat([fake_df,true_df],ignore_index=True)

# remove the hold out set
# identical seed to rnn set
df, hold = train_test_split(df, test_size = .1, random_state= 46, shuffle= True)
df = df.reset_index(drop = True)

# seperate label from features
X=df.title.copy()
y=df.Label.copy()

# clean text features
# for the NB model a line will be added to stem words
stop_words = stopwords.words('english')
porter = PorterStemmer()

# this single function will take care of all the pre-model text cleaning

def clean(text): # argument is a title or or any string
    p1 = text
    def stemSentence(p1):  # stage 1 is to stem words
        token_words=word_tokenize(p1)
        token_words
        stem_sentence=[]
        for word in token_words:
            stem_sentence.append(porter.stem(word))
            stem_sentence.append(" ")
        stage_1 = "".join(stem_sentence)
        return(stage_1)

    p2 = stemSentence(p1) # process cleaning stage 1

    def refine(p2):  # stage 2 is to make all lower case and remove unessesary elements
        p2 = re.sub("U.S.|US|U.S|US.", "united states", p2)
        p2 =re.sub("[^a-zA-z]"," ",p2) # removing expressions that are not word
        p2=p2.lower()
        p2 = p2.split()
        p2=" ".join([word for word in p2 if not word in stop_words])
        return(p2)
    cleaned = refine(p2) # feed stage 1 results into secound stage of cleaning
    return(cleaned) # return clean text string


df["Cleaned"]= list(map(clean, X))

# write file for visualization purposes
# see plotting.R
df.to_csv('data/Visual.csv')

# replit data by fake and true

true = df[df['Label'] == 0][['Label', 'Cleaned']]
fake = df[df['Label'] == 1][['Label', 'Cleaned']]

# coutn frequency within each word
cv = CountVectorizer()
cv.fit(df['Cleaned'])
X = cv.transform(df['Cleaned'])
y = np.array(df['Label'])

# build classifier
clf = MultinomialNB()
clf.fit(X, y)

# get trainset predictions
train_preds = clf.predict(X)


# model pipeline
def NB_model(unseen_data):
    unseen_data = pd.Series(unseen_data)
    X = list(map(clean, unseen_data))
    X = cv.transform(X)
    predictions = clf.predict_proba(X)[::,1]
    return(predictions)
