# -*- coding: utf-8 -*-
'''
Created on Jul 20, 2015

@author: jordan
'''
import time
import codecs
import json
import math
import os
import logging
import re
import string
from stopwords import heb_stopwords


class Progress:
    def __init__(self):
        self.i = 0
        self.start = time.time()
        
    def progress(self,current, total):
        if(100*float(current)/total > self.i):
            self.i+=1
            elapsed = time.time() - self.start
            self.start = time.time()
            print(str(self.i)+" : "+str(elapsed))

def list2file(l,filename):
    f = codecs.open(filename, 'w','cp1255')
#     cw = csv.writer(f, delimiter='|')
    json.dump(l,f,indent=4, separators=(',', ': '),default=str)
#     for k,v in dict:
#         f.write(u' '.join(k).encode('utf-8')+','+str(v)+'\n')
#         cw.writerow([(str(k)),str(v)])
    f.close()

def get_pattern(opcodes, w1, w2):
    pattern = ''
    for o in opcodes:
        if o[0] == 'insert':
            pattern += w2[o[3]:o[4]]
        elif o[0] == 'delete':
            pattern += w1[o[1]:o[2]]
        elif o[0] == 'equal':
            pattern += '*'
        else:
#             no replacements
            return None
    return pattern

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


def get_json(path):
    data_json = []
    for file in os.listdir(path):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(path, file)) as data_file:
            data_json += json.load(data_file)
    return data_json


def get_json_data(path):
    data_json = get_json(path)
    data = {}
#     tags = {}
    for entry in data_json:
        if entry['model'] == "facebook_feeds.facebook_status":
            entry_id = entry['pk'] 
            if entry_id not in data:
                data[entry_id] = {"id":entry_id, "text":"", "tags":[], "feed":0}
            data[entry_id]["text"] = entry['fields']['content']
            data[entry_id]["feed"] = entry['fields']['feed']
        if entry['model'] == "kikartags.taggeditem":
            entry_id = entry['fields']['object_id'] 
            if entry_id not in data:
                data[entry_id] = {"id":entry_id, "text":"", "tags":[], "feed":0}
            data[entry_id]["tags"].append(str(entry['fields']['tag']))
    return data.values()


puncmarks = string.punctuation+u'״׳'
     
def clean_text(text):
    logging.debug(text)
    logging.debug(type(text))
    if type(text) != unicode: 
        text = unicode(text, encoding = 'utf-8', errors = 'replace')
#         text = text.decode('utf-8').encode('utf-8')
    text = text.lower()
#     text = re.sub('[^a-zA-Zא-ת0-9\"\\n]',' ', text)
#     add space before punctuation   
    text = re.sub(u'(\S)(['+puncmarks+']+)',r'\1 \2 ', text, flags=re.UNICODE)
#     add space after punctuation   
    text = re.sub(u'(['+puncmarks+']+)(\S)',r' \1 \2', text, flags=re.UNICODE)
    logging.debug(text)        
#         text = re.sub('(\w+)', self.wnl.lemmatize('\\1'),text) #TODO: check
    text = re.sub('\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in heb_stopwords])
#         text = ' '.join([word.encode('utf-8') for word in text.split() if word.encode('utf-8') not in heb_stopwords])
#         f = codecs.open('clean_text','w','utf-8')
#         f.write(text)
#         f.close()
    return text
