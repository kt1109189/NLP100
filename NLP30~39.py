#30. 形態素解析結果の読み込み
filename = 'neko.txt.mecab'
sentences = []
morphs = []
with open(filename, mode='r') as f:
    for line in f: 
        if line != 'EOS\n':
            fields = line.split('\t')
            if len(fields) != 2 or fields[0] == '':
                continue
            else:
                attr =  fields[1].split(',')
                morph = {'surface': fields[0], 'base': attr[6], 'pos': attr[0], 'pos1': attr[1]}
                morphs.append(morph)
        else:
            sentences.append(morphs)
            morphs = []

#31. 動詞
ans = set()
for sentence in sentences:
    for morph in sentence:
        if morph['pos'] == '動詞':
            ans.add(morph['surface'])

#32.動詞の基本形
ans = set()
for sentence in sentences:
    for morph in sentence:
        if morph['pos'] == '動詞':
            ans.add(morph['base'])

#33.「AのB」
ans = set()
for sentence in sentences:
    for i in range(1, len(sentence)-1):
        if sentence[i-1]['pos'] == '名詞' and sentence[i]['surface'] == 'の' and sentence[i+1]['pos'] == '名詞':
            ans.add(sentence[i-1]['surface'] + 'の' + sentence[i+1]['surface'])

#34.名詞の連接
ans = set()
for sentence in sentences:
    nouns = ''
    num = 0
    for morph in sentence:
        if morph['pos'] == '名詞':  # 最初の形態素から順に、名詞であればnounsに連結し、連結数(num)をカウント
            nouns = ''.join([nouns, morph['surface']])
            num += 1
        elif num >= 2:  # 名詞以外、かつここまでの連結数が2以上の場合は出力し、nounsとnumを初期化
            ans.add(nouns)
            nouns = ''
            num = 0
        else:  # それ以外の場合、nounsとnumを初期化
            nouns = ''
            num = 0
    if num >= 2: 
        ans.add(nouns)

#35.単語の出現頻度
from collections import defaultdict
ans = defaultdict(int)
for sentence in sentences:
    for morph in sentence:
        if morph['pos'] != '記号':
            ans[morph['base']] += 1
ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)

#36. 頻度上位10語
import matplotlib.pyplot as plt
import japanize_matplotlib
keys = [a[0] for a in ans[0:10]]
values = [a[1] for a in ans[0:10]]
#plt.figure(figsize=(8, 4))
#plt.bar(keys, values)
#plt.show()

#37. 「猫」と共起頻度の高い上位10語
ans = defaultdict(int)
for sentence in sentences:
    if '猫' in [morph['surface'] for morph in sentence]:
        for morph in sentence:
            if morph['pos'] != '記号':
                ans[morph['base']] += 1
del ans['猫']
ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)
keys = [a[0] for a in ans[0:10]]
values = [a[1] for a in ans[0:10]]
#plt.figure(figsize=(8, 4))
#plt.bar(keys, values)
#plt.show()

#38.ヒストグラム
ans = defaultdict(int)
for sentence in sentences:
    for morph in sentence:
        if morph['pos'] != '記号':
            ans[morph['base']] += 1  # 単語数の更新(初登場の単語であれば1をセット)
ans = ans.values()
#plt.figure(figsize=(8, 4))
#plt.hist(ans, bins=100)
#plt.xlabel('出現頻度')
#plt.ylabel('単語の種類数')
#lt.show()

#39.zipfの法則
import math
ans = defaultdict(int)
for sentence in sentences:
    for morph in sentence:
        if morph['pos'] != '記号':
            ans[morph['base']] += 1
ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)
ranks = [r + 1 for r in range(len(ans))]
values = [a[1] for a in ans]
plt.figure(figsize=(8, 4))
plt.scatter(ranks, values)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('出現頻度順位')
plt.ylabel('出現頻度')
plt.show()