import re
str = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
str = re.sub('[,\.]', ' ', str)
splits = str.split()
ans = [len(i) for i in splits]
print(ans)