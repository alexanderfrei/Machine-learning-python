import numpy as np
import pandas as pd
import gc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from scipy.sparse import hstack

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
sample_submission = pd.read_csv('../input/sample_submission.csv')
train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

print("Job 1 done")
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

del word_vectorizer
gc.collect()

print("Job 2 done")
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

del char_vectorizer
gc.collect()

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

print("Job 3 done")

submission = pd.DataFrame.from_dict({'id': test['id']})

num_folds = 10
seed = 42
np.random.seed(42)
train_oof = pd.DataFrame.from_dict({'id': train['id']})
test_oof = pd.DataFrame.from_dict({'id': test['id']})

# OOF
def oof(X_train, X_test, y, model, predict_func, num_folds, seed):
    
    scores = []
    train_predict = np.zeros((X_train.shape[0],1))
    test_predict = np.zeros((X_test.shape[0],1))
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    i = 1
    for train_idx, val_idx in kf.split(X_train):
        
        print("Fold {}".format(i))
        
        # fit model 
        x_train = X_train[train_idx]
        x_val = X_train[val_idx]
        y_train = y[train_idx].values
        y_val = y[val_idx].values
        
        model.fit(x_train, y_train)
        
        # predict train and test oof 
        
        if predict_func=="predict_proba":
            train_predict[val_idx] = model.predict_proba(x_val)[:, 1].reshape((-1,1))
            test_predict += np_rank(model.predict_proba(X_test)[:, 1].reshape((-1,1)))
        
        # save scores 
        cv_score = roc_auc_score(y_val, train_predict[val_idx])
        scores.append(cv_score)
        print("ROC AUC score = {}".format(cv_score))
        
        i = i + 1

        del model
        gc.collect()
        
    test_predict /= (num_folds*test_predict.shape[0])
    return scores, train_predict, test_predict

classifier = LogisticRegression(C=0.1, solver='sag')
for j, class_name in enumerate(class_names):
    print("Class {}".format(class_name))
    _, train_oof[class_name], test_oof[class_name] = oof(train_features, 
                                                              test_features, 
                                                              train[class_name],
                                                              classifier, 
                                                              "predict_proba", 
                                                              num_folds, 
                                                              seed
                                                             )
    
train_oof.to_csv("../output/test/test_oof_lr_nopreprocessing.csv", index=False)
train_oof.to_csv("../output/test/test_oof_lr_nopreprocessing.csv", index=False)