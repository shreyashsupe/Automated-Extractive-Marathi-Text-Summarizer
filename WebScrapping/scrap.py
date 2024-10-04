from bs4 import BeautifulSoup as bs
import urllib.request,sys,time
import requests
import pandas as pd
import re
def remove_tags(text):
    return TAG_RE.sub('', text)
visited='https://marathi.abplive.com/news/pune'
max_corp=10
url=visited
page=requests.get('{}'.format(url))
soup=bs(page.content,'html.parser')
LINKS=[]
links_ = [a.get('href') for a in soup.find_all('a', href=True)]
for i in links_:
  i = i.strip()
  i = i.split('?')[0]
  i = i.split('#')[0]
  i = urllib.parse.unquote(i)
  if i not in LINKS and i.startswith(visited):
    LINKS.append(i)
LINKS = list(set(LINKS))
doc=[]
parartname=[]
doclink=[]
i=0
TAG_RE = re.compile(r'<[^>]+>')
for i in range(len(LINKS)):
    if len(doclink)<=max_corp:
        paragraphs = []
        page=requests.get('{}'.format(LINKS[i]))
        soup=bs(page.content,'html.parser')
        try:
            artname=soup.find(class_='article-title')
            breadcrumbs = soup.find(class_='article-data _thumbBrk uk-text-break').findAll('p')
            doclink.append(LINKS[i])
            i+=1
            for x in breadcrumbs:
                paragraphs.append(remove_tags(str(x)))
            for i in range(len(paragraphs)):
                paragraphs[i]=paragraphs[i].split('\xa0')
            parartname.append(remove_tags(str(artname)))
            doc.append(paragraphs)

        except AttributeError:
            pass
    else:
        break
print(doclink)
for i in range(len(doc)):
    f='./Document/abpmajha'+str(41+i)+'.txt'
    try:
        stri=' '
        tp=open(f, 'w',encoding="utf-8")
        tp.write(parartname[i])
        for j in range(len(doc[i])):
            x=stri.join(doc[i][j]) 
            tp.write(x)
            tp.write('\n')
        tp.close()
    except OSError:
        print('Failed creating the file')
    else:
        print('File created')