def n_gram(lst, n):
    return list(zip(*[lst[i:] for i in range(n)]))

str = "I am an NLPer"
words = n_gram(str.split(), 2)
chars = n_gram(str, 2)
print('単語bi-gram:', words)
print('文字bi-gram:', chars)