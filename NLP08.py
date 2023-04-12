def cipher(str):
    rep = [chr(219 - ord(x)) if x.islower() else x for x in str]
    return ''.join(rep)

msg = "This text is encrypted."
msg = cipher(msg)
print('暗号化:', msg)
msg = cipher(msg)
print('復号化:', msg)