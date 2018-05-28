import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')

def np_rank(array):
    ranks = np.empty_like(array)
    for i in np.arange(array.shape[1]):
        temp = array[:, i].argsort()
        ranks[temp, i] = np.arange(len(array))
    return ranks

def save_oof(train_oof, test_oof, sample_submission):
    test_oof.to_csv("../output/test/test_oof_nb.csv", index=False)
    train_oof.to_csv("../output/train/train_oof_nb.csv", index=False)
    
lens = train.comment_text.str.len()
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)

COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])

def pr(x_train, y_i, y):
    p = x_train[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)
    
x_train = trn_term_doc
x_test = test_term_doc

def nb_model_fit(X, y):
    r = np.log(pr(X,1,y) / pr(X,0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = X.multiply(r)
    return m.fit(x_nb, y), r
    
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
def oof(X_train, X_test, y, num_folds, seed):
    
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
        
        m,r = nb_model_fit(x_train, y_train)
        
        # predict train and test oof 
        train_predict[val_idx] = m.predict_proba(x_val.multiply(r))[:,1].reshape(-1,1)
        test_predict += np_rank(m.predict_proba(X_test.multiply(r))[:,1].reshape(-1,1))
        
        # save scores 
        cv_score = roc_auc_score(y_val, train_predict[val_idx])
        scores.append(cv_score)
        print("ROC AUC score = {}".format(cv_score))
        
        i = i + 1
        
    test_predict /= (num_folds*test_predict.shape[0])
    return scores, train_predict, test_predict
    
num_folds = 10
seed = 42
np.random.seed(42)
train_oof = pd.DataFrame.from_dict({'id': train['id']})
test_oof = pd.DataFrame.from_dict({'id': test['id']})

for j, class_name in enumerate(label_cols):
    print("Class {}".format(class_name))
    _, train_oof[class_name], test_oof[class_name] = oof(x_train, x_test, train[class_name], num_folds, seed)

save_oof(train_oof, test_oof, sample_submission)   
 
# preds = np.zeros((len(test), len(label_cols)))

# for i, j in enumerate(label_cols):
#     print('fit', j)
#     m,r = get_mdl(train[j])
#     preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]