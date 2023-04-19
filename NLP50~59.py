#50.データの入手・整形
import pandas as pd
from sklearn.model_selection import train_test_split

filepath = './newsCorpora.csv'
df = pd.read_csv(filepath, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df = df.loc[df['PUBLISHER'].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']), ['TITLE', 'CATEGORY']]

train, valid = train_test_split(df, test_size=0.2, shuffle=True, random_state=42, stratify=df['CATEGORY'])
valid, test = train_test_split(valid, test_size=0.5, shuffle=True, random_state=42, stratify=valid['CATEGORY'])

"""
train.to_csv('./train.txt', sep='\t', index=False)
valid.to_csv('./valid.txt', sep='\t', index=False)
test.to_csv('./test.txt', sep='\t', index=False)
print('【学習データ】')
print(train['CATEGORY'].value_counts())
print('【検証データ】')
print(valid['CATEGORY'].value_counts())
print('【評価データ】')
print(test['CATEGORY'].value_counts())
"""

#51.特徴量の抽出
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
def preprocessing(text):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(table)
    text = text.lower()
    text = re.sub('[0-9]+', '0', text)

    return text

df = pd.concat([train, valid, test], axis=0)
df.reset_index(drop=True, inplace=True)
df['TITLE'] = df['TITLE'].map(lambda x: preprocessing(x))
train_valid = df[:len(train) + len(valid)]
test = df[len(train) + len(valid):]
vec_tfidf = TfidfVectorizer(min_df=10, ngram_range=(1, 2))
X_train_valid = vec_tfidf.fit_transform(train_valid['TITLE'])
X_test = vec_tfidf.transform(test['TITLE'])
X_train_valid = pd.DataFrame(X_train_valid.toarray(), columns=vec_tfidf.get_feature_names_out())
X_test = pd.DataFrame(X_test.toarray(), columns=vec_tfidf.get_feature_names_out())
X_train = X_train_valid[:len(train)]
X_valid = X_train_valid[len(train):]
"""
X_train.to_csv('./X_train.txt', sep='\t', index=False)
X_valid.to_csv('./X_valid.txt', sep='\t', index=False)
X_test.to_csv('./X_test.txt', sep='\t', index=False)
"""

#52.学習
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression(random_state=42, max_iter=10000)
lg.fit(X_train, train['CATEGORY'])

#53.予測
import numpy as np

def score_lg(lg, X):
    return np.max(lg.predict_proba(X), axis=1), lg.predict(X)
proba_train, pred_train = score_lg(lg, X_train)
proba_test, pred_test = score_lg(lg, X_test)

#54.正解率の計測
from sklearn.metrics import accuracy_score

train_accuracy = accuracy_score(train['CATEGORY'], pred_train)
test_accuracy = accuracy_score(test['CATEGORY'], pred_test)
"""
print(f'正解率（学習データ）：{train_accuracy:.3f}')
print(f'正解率（評価データ）：{test_accuracy:.3f}')
"""

#55.混同行列の作成
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

train_cm = confusion_matrix(train['CATEGORY'], pred_train)
"""
print(train_cm)
sns.heatmap(train_cm, annot=True, cmap='Blues')
plt.show()
test_cm = confusion_matrix(test['CATEGORY'], pred_test)
print(test_cm)
sns.heatmap(test_cm, annot=True, cmap='Blues')
plt.show()
"""

#56. 適合率，再現率，F1スコアの計測
from sklearn.metrics import precision_score, recall_score, f1_score
def calculate_scores(y_true, y_pred):
    precision = precision_score(test['CATEGORY'], y_pred, average=None, labels=['b', 'e', 't', 'm'])  # Noneを指定するとクラスごとの精度をndarrayで返す
    precision = np.append(precision, precision_score(y_true, y_pred, average='micro'))  # 末尾にマイクロ平均を追加
    precision = np.append(precision, precision_score(y_true, y_pred, average='macro'))  # 末尾にマクロ平均を追加
    recall = recall_score(test['CATEGORY'], y_pred, average=None, labels=['b', 'e', 't', 'm'])
    recall = np.append(recall, recall_score(y_true, y_pred, average='micro'))
    recall = np.append(recall, recall_score(y_true, y_pred, average='macro'))
    f1 = f1_score(test['CATEGORY'], y_pred, average=None, labels=['b', 'e', 't', 'm'])
    f1 = np.append(f1, f1_score(y_true, y_pred, average='micro'))
    f1 = np.append(f1, f1_score(y_true, y_pred, average='macro'))
    scores = pd.DataFrame({'適合率': precision, '再現率': recall, 'F1スコア': f1},
    index=['b', 'e', 't', 'm', 'マイクロ平均', 'マクロ平均'])
    return scores
"""
print(calculate_scores(test['CATEGORY'], pred_test))
"""

#57. 特徴量の重みの確認
features = X_train.columns.values
index = [i for i in range(1, 11)]
"""
for c, coef in zip(lg.classes_, lg.coef_):
    print(f'【カテゴリ】{c}')
    best10 = pd.DataFrame(features[np.argsort(coef)[::-1][:10]], columns=['重要度上位'], index=index).T
    worst10 = pd.DataFrame(features[np.argsort(coef)[:10]], columns=['重要度下位'], index=index).T
    print(pd.concat([best10, worst10], axis=0))
    print('\n')
"""

#58. 正則化パラメータの変更
from tqdm import tqdm

result = []
"""
for C in tqdm(np.logspace(-5, 4, 10, base=10)):
    lg = LogisticRegression(random_state=42, max_iter=10000, C=C)
    lg.fit(X_train, train['CATEGORY'])
    train_pred = score_lg(lg, X_train)
    valid_pred = score_lg(lg, X_valid)
    test_pred = score_lg(lg, X_test)
    train_accuracy = accuracy_score(train['CATEGORY'], train_pred[1])
    valid_accuracy = accuracy_score(valid['CATEGORY'], valid_pred[1])
    test_accuracy = accuracy_score(test['CATEGORY'], test_pred[1])
    result.append([C, train_accuracy, valid_accuracy, test_accuracy])
result = np.array(result).T
plt.plot(result[0], result[1], label='train')
plt.plot(result[0], result[2], label='valid')
plt.plot(result[0], result[3], label='test')
plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
plt.xscale ('log')
plt.xlabel('C')
plt.legend()
plt.show()
"""

#59. ハイパーパラメータの探索
import optuna

def objective_lg(trial):
    l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1)
    C = trial.suggest_loguniform('C', 1e-4, 1e4)
    lg = LogisticRegression(random_state=42, 
                            max_iter=10000, 
                            penalty='elasticnet', 
                            solver='saga', 
                            l1_ratio=l1_ratio, 
                            C=C)
    lg.fit(X_train, train['CATEGORY'])
    valid_pred = score_lg(lg, X_valid)
    valid_accuracy = accuracy_score(valid['CATEGORY'], valid_pred[1])

    return valid_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective_lg, timeout=600)
print('Best trial:')
trial = study.best_trial
print('  Value: {:.3f}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

lg = LogisticRegression(random_state=42, 
                        max_iter=10000, 
                        penalty='elasticnet', 
                        solver='saga', 
                        l1_ratio=trial.params['l1_ratio'], 
                        C=trial.params['C'])
lg.fit(X_train, train['CATEGORY'])

train_pred = score_lg(lg, X_train)
valid_pred = score_lg(lg, X_valid)
test_pred = score_lg(lg, X_test)

train_accuracy = accuracy_score(train['CATEGORY'], train_pred[1]) 
valid_accuracy = accuracy_score(valid['CATEGORY'], valid_pred[1]) 
test_accuracy = accuracy_score(test['CATEGORY'], test_pred[1]) 

print(f'正解率（学習データ）：{train_accuracy:.3f}')
print(f'正解率（検証データ）：{valid_accuracy:.3f}')
print(f'正解率（評価データ）：{test_accuracy:.3f}')