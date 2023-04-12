str1 = "paraparaparadise"
str2 = "paragraph"
X = set(zip(*[str1[i:] for i in range(2)]))
Y = set(zip(*[str2[i:] for i in range(2)]))
print('X:', X)
print('Y:', Y)
print('和:', X | Y)
print('積:', X & Y)
print('差:', X - Y)
print('Xにseが含まれるか:', {('s', 'e')} <= X)
print('Yにseが含まれるか:', {('s', 'e')} <= Y)