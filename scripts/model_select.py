import pandas as pd
import numpy as np
import sklearn
from tensorflow import keras
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# load keras model
model = keras.models.load_model('models/rnn')

# load nb model
import train_nb

# load held out data
test_set = pd.read_csv('data/held_out.csv')

# create rounding function
# this wil be used to conver prs to 0/1 decisions
def rounder(vec, threshold):
    preds = np.where(vec > threshold, 1, 0)
    return(preds)

########## fit nb ############
nb_predictions = rounder(NB_model(test_set.title), .5)

sklearn.metrics.accuracy_score(test_set['Label'], nb_predictions) # accuray of NB
### NB test set accuracy is 0.9436525612472161

# get probabiliteis
nb_prob = NB_model(test_set.title)
nb_fpr, nb_tpr, _ = roc_curve(test_set['Label'],  nb_prob)
nb_auc = roc_auc_score(test_set['Label'], nb_prob)



########## fit rnn ############

# run rnn pipeline sequence
import rnn_pipeline
X_test = list(map(clean, test_set.title))
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_sequences,padding="post",maxlen=max_lenght)

# predict
model.predict(X_test_padded)



rnn_predicitons = np.ndarray.flatten(rounder(model.predict(X_test_padded),.5))
sklearn.metrics.accuracy_score(test_set['Label'], rnn_predicitons) # accuray of rnn
### RNN accuracy 0.9628062360801781



############## AUC CURVE ##################
# get probabiliteis
rnn_prob = model.predict(X_test_padded)
rnn_fpr, rnn_tpr, _ = roc_curve(test_set['Label'],  rnn_prob)
rnn_auc = roc_auc_score(test_set['Label'], rnn_prob)


results = {'classifiers':['Naive Bayes', 'RNN'],
                                        'fpr':[nb_fpr, rnn_fpr],
                                        'tpr':[nb_tpr, rnn_tpr],
                                        'auc':[nb_auc, rnn_auc]}

result_table = pd.DataFrame(results)
result_table.set_index('classifiers', inplace=True)

fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'],
             result_table.loc[i]['tpr'],
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

plt.plot([0,1], [0,1], color='black', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()


############## Find mis-classifications #################
rnn_miss = test_set[rnn_predicitons!=test_set['Label']]
rnn_miss.to_csv('data/rnn_miss.csv')
nb_miss = test_set[nb_predictions!=test_set['Label']]
nb_miss.to_csv('data/nb_miss.csv')
