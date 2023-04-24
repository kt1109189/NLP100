#80.ID番号への変換
import pandas as pd
from collections import defaultdict
import string

train = pd.read_table('./train.txt')
d = defaultdict(int)
table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
for text in train['TITLE']:
    for word in text.translate(table).split():
        d[word] += 1
d = sorted(d.items(), key=lambda x: x[1], reverse=True)

word2id = {word: i + 1 for i, (word, cnt) in enumerate(d) if cnt > 1}

def tokenizer(text, word2id=word2id, unk=0):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return [word2id.get(word, unk) for word in text.translate(table).split()]

text = train.iloc[1, train.columns.get_loc('TITLE')]
print(f'テキスト: {text}')
print(f'ID列: {tokenizer(text)}')