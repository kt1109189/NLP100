import json
import re
import requests

#20.JSONデータの読み込み
with open('./Desktop/NLP100/jawiki-country.json', mode='r') as f:
    for line in f:
        line = json.loads(line)
        if line['title'] == 'イギリス':
            text = line['text']
            break

#21.カテゴリ名を含む行を抽出
pattern = r'^(.*\[\[Category:.*\]\].*)$'
result = '\n'.join(re.findall(pattern, text, re.MULTILINE))
#print(result)

#22.カテゴリ名の抽出
pattern = r'^.*\[\[Category:(.*?)(?:\|.*)?\]\].*$'
result = '\n'.join(re.findall(pattern, text, re.MULTILINE))
#print(result)

#23.セクション構造
pattern = r'^(\={2,})\s*(.+?)\s*\={2,}.*$'
result = '\n'.join(i[1] + ':' + str(len(i[0]) - 1) for i in re.findall(pattern, text, re.MULTILINE))
#print(result)

#24.ファイル参照の抽出
pattern = r'\[\[ファイル:(.+?)\|'
result = '\n'.join(re.findall(pattern, text))
#print(result)

#25.テンプレートの抽出
pattern = r'^\{\{基礎情報.*?$(.*?)^\}\}'
template = re.findall(pattern, text, re.MULTILINE + re.DOTALL)
#print(template)
pattern = r'^\|(.+?)\s*=\s*(.+?)(?=(?=\n\|)|(?=\n$))'
result = dict(re.findall(pattern, template[0], re.MULTILINE + re.DOTALL))
#for k, v in result.items():
#    print(k + ': ' + v)

def remove_markup(text):
    #26.強調マークアップの除去
    pattern = r'\'{2,5}'
    text = re.sub(pattern, '', text)
    #27.内部リンクマークアップの除去
    pattern = r'\[\[(?:[^|]*?\|)??([^|]*?)\]\]'
    text = re.sub(pattern, r'\1', text)
    #28.MediaWikiマークアップの除去
    pattern = r'https?://[\w!?/\+\-_~=;\.,*&@#$%\(\)\'\[\]]+'
    text = re.sub(pattern, '', text)
    pattern = r'<.+?>' 
    text = re.sub(pattern, '', text)
    pattern = r'\{\{(?:lang|仮リンク)(?:[^|]*?\|)*?([^|]*?)\}\}' 
    text = re.sub(pattern, r'\1', text)
    return text
result_rm = {k: remove_markup(v) for k, v in result.items()}
#for k, v in result_rm.items():
#    print(k + ': ' + v)

#29.国旗画像のurlを取得する
def get_url(text):
    url_file = text['国旗画像'].replace(' ', '_')
    url = 'https://commons.wikimedia.org/w/api.php?action=query&titles=File:' + url_file + '&prop=imageinfo&iiprop=url&format=json'
    data = requests.get(url)
    return re.search(r'"url":"(.+?)"', data.text).group(1)
print(get_url(result))