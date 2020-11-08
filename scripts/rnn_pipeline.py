import pandas as pd
from nltk.corpus import stopwords
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


fake_df=pd.read_csv('input/Fake.csv')
true_df=pd.read_csv('input/True.csv')

# fake coded to 1
fake_df["Label"]=1
true_df["Label"]=0

df=pd.concat([fake_df,true_df],ignore_index=True)

# create held out final
# this set will not be used in any section of training or validation
df, hold = train_test_split(df, test_size = .1, random_state= 46, shuffle= True)
df = df.reset_index(drop = True)
hold = hold.reset_index(drop = True)
# write held out to data directory
hold.to_csv('data/held_out.csv')

# seperate label from features
X=df.title.copy()
y=df.Label.copy()

stop_words = stopwords.words('english')
def clean(title):
    title = re.sub("U.S.|US|U.S|US.", "united states", title)
    title =re.sub("[^a-zA-z]"," ", title) # removing expressions that are not word
    title=title.lower()
    title = title.split()
    title=" ".join([word for word in title if not word in stop_words])
    return(title)


df["Cleaned"]= list(map(clean, X))

X=df.Cleaned

# split train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=46,shuffle=True)

# define embedings
max_lenght=100
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index # creating word dict for words in training
sequences = tokenizer.texts_to_sequences(X_train)  # replacing words with the number corresponding to them in the dictionary(word_index)
X_train_padded = pad_sequences(sequences, padding='post',maxlen=max_lenght) # padding words
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_sequences,padding="post",maxlen=max_lenght)
vocab_size = len(tokenizer.word_index)+1
embedding_dim=16
