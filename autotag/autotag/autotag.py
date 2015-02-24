# -*- coding: utf-8 -*-
'''
Created on Sep 22, 2014

@author: jordan
'''
from collections import Counter
import operator
import re
import math
import itertools
import json
import codecs
import difflib
import string
import time
import numpy
# import scipy.sparse
# import scipy.spatial
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import logging
import logging.config
import os
from stopwords import heb_stopwords
import random
import csv
from itertools import izip
# import cPickle

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

def tf(word,words):
    return words.count(word)

def idf(word,statuses,N):
#     N = len(statuses)
    n = statuses.count(word)
    return math.log(float(N)/n)

def get_tags(statuses, stat_count):
    statuses = statuses.split('End of Status')
    N = len(statuses)
#     most_common = parse_common()
    f = open('tags', 'w')
    for status in statuses:
        words = status.split()
        counts = map(lambda x: tf(x,words)*idf(x,statuses,N)  , words)
        if(len(counts)==0):
            continue
        max_index, max_value = max(enumerate(counts), key=operator.itemgetter(1))
        tag = words[max_index]
        f.write('-------- Tag --------\n')
        f.write(tag +  ' : \n')
        f.write('-------- Status --------\n')
        f.write(status + '\n')
        f.write('--------\n')
    
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

def cluster_words(count):
    words = set(dict(count).keys())
    comb = itertools.combinations(words,2)
    cnum = nCr(len(words),2)
    print(cnum)
    s = difflib.SequenceMatcher()
    dist = {}
    prog = Progress()
    for i,c in enumerate(comb):
        prog.progress(i, cnum)
        s.set_seq1(c[0]) 
        s.set_seq2(c[1])
        dist[c] = s.real_quick_ratio()
        
def get_patterns(count,threshold):
#     וכשתלכנה - ו, כש, ת**נה, הלך
#    להגשים - ל, ה**י*, גשמ
#    find all frequent diffs between words
#     count = dict(count)
    words = set(dict(count).keys())
    print(len(count))
    f = codecs.open('matches', 'w', 'utf-8')
    prog = Progress()
    s = difflib.SequenceMatcher()
    patterns = []
    for w1,c1 in count:
        prog.progress(count.index((w1,c1)), len(count))
#         get close words
        s.set_seq1(w1)
        matches = set()
        for w2 in words:
            s.set_seq2(w2)
            if s.real_quick_ratio() >= threshold and \
               s.quick_ratio() >= threshold and \
               s.ratio() >= threshold:
                matches.add(w2)
                pattern = get_pattern(s.get_opcodes(),w1,w2)
                if pattern:
                    patterns.append(pattern)
#         matches = set(difflib.get_close_matches(w,words,100,0.8))
#         diff is a possible pattern
        words = words - matches
        if(len(matches)>1):
            f.write(w1+' : '+', '.join(matches)+'\n')
            pattern_count = Counter(patterns)
            pattern_count = pattern_count.most_common()
            f2 = codecs.open('patterns', 'w', 'utf-8')
            f2.writelines([unicode(p)+' : '+str(c)+'\n' for p,c in pattern_count])
            f2.close()
#     combos = list(itertools.combinations(count.keys(),2))
#     print(len(combos))
    f.close()
    pass 

def split_pattern(count):
#     ואת -> ו את
    words = set(dict(count).keys())
    print(len(words))    
    for w,c in count:
        if w[0] in u'ולכשמבה' and w[1:] in words:
            words.remove(w)
            count.remove((w,c))
    print(len(words))
    f = codecs.open('split_words', 'w', 'utf-8')
    for w,c in count:
        f.write(unicode(w) +' : '+ str(c)+'\n')
    f.close()    
    return words 
        
# def bag_of_words(data, words):
#     print(len(data))
#     print(len(words))
#     m = scipy.sparse.lil_matrix((len(data),len(words)))
#     prog = Progress()
#     for i,d in enumerate(data):
#         prog.progress(i, len(data))
#         dwords = d.split()
#         sum = 0
#         for w in dwords:
#             if w in words:
#                 sum+=1
#                 j = words.index(w)
#                 m[i,j] = 1
#         m[i,:] = m[i,:]/sum
#     return m

def get_distances(bag):
    bag = bag.tocsr()
    distmat = bag*bag.transpose()
    distmat = distmat.todense()
    numpy.fill_diagonal(distmat, 0)
    with file('distmat.txt', 'w') as outfile:
        numpy.savetxt(outfile, distmat)
    return distmat 

#    max similarity clustered together
def cluster_sentences(distmat):
    clusters = {}
    clustersN = 0
    for i in range(100):
        closest = distmat.argmax()
        closest = numpy.unravel_index(closest, distmat.shape)
        if closest[0] not in clusters and closest[1] not in clusters:
            clusters[closest[0]] = clustersN
            clusters[closest[1]] = clustersN
            clustersN += 1
        elif closest[0] not in clusters:
            clusters[closest[0]] = clusters[closest[1]]
        elif closest[1] not in clusters:
            clusters[closest[1]] = clusters[closest[0]]
        else:
            for j in clusters:
                if clusters[j] == clusters[closest[1]]:
                    clusters[j] = clusters[closest[0]]
        distmat[closest[0],closest[1]] = 0
            
    print(clusters)
    return clusters

def get_json_data(dir):
    data_json = []
    data = {}
#     tags = {}
    for file in os.listdir(dir):
        if not file.endswith(".json"):
            continue
        with open(dir+'/'+file) as data_file:
            data_json += json.load(data_file)
    for entry in data_json:
        if entry['model'] == "facebook_feeds.facebook_status":
            id = entry['pk'] 
            if id not in data:
                data[id] = {"id":id, "text":"", "tags":[], "feed":0}
            data[id]["text"] = entry['fields']['content']
            data[id]["feed"] = entry['fields']['feed']
        if entry['model'] == "kikartags.taggeditem":
            id = entry['fields']['object_id'] 
            if id not in data:
                data[id] = {"id":id, "text":"", "tags":[], "feed":0}
            data[id]["tags"].append(str(entry['fields']['tag']))
    return data.values()
     
        
# def get_data():
#     print('getting data')
#     with open('db/statuses.txt') as statuses_file:
#         statuses_json = json.load(statuses_file)
#     statuses = {status['pk']: status['fields']['content'] for status in statuses_json}
#     with open('db/tags.txt') as tags_file:
#         tags_data = json.load(tags_file)
#     tags = {}
#     for tag in tags_data:
#         id = tag['fields']['object_id'] 
#         if id not in tags:
#             tags[id] = []
#         tags[id].append(str(tag['fields']['tag']))
#     data = [DataEntry(i,statuses[i], tags[i]) for i in statuses if i in tags]
#     return data
        
class AutoTag:
    
    def __init__(self):
        logging.basicConfig(filename='log',level=logging.INFO)
        self.wnl = WordNetLemmatizer()

    def clean_text(self,text):
        text = unicode(text)
        text = text.lower()
    #     text = re.sub('[^a-zA-Zא-ת0-9\"\\n]',' ', text)
        puncmarks = string.punctuation+'״׳'
    #     add space before punctuation   
        text = re.sub('(\S)(['+puncmarks+']+)','\\1 \\2 ', text)
    #     add space after punctuation   
        text = re.sub('(['+puncmarks+']+)(\S)',' \\1 \\2', text)
        
        text = re.sub('(\w+)', self.wnl.lemmatize('\\1'),text)

        text = re.sub('\s+', ' ', text)
        
        text = ' '.join([word.encode('utf-8') for word in text.split() if word.encode('utf-8') not in heb_stopwords])
#         f = codecs.open('clean_text','w','utf-8')
#         f.write(text)
#         f.close()
        return text
        
    def count_data(self,data,name):
        data_count = Counter(data)
        data_count = sorted(data_count.items(), key=lambda item: item[1],reverse = True)
#         with open(name+'.json', 'w') as f:
# #             data_encoded = [(w,c) for w,c in data_count]
#             data_encoded = [(w.encode('utf-8'),c) for w,c in data_count]
#             json.dump(data_encoded, f, ensure_ascii=False, indent = 4, separators=[',',': ']) #TODO: fix
        with open(name+'.csv', 'w') as f:
            data_encoded = [(w,c) for w,c in data_count]
#             data_encoded = [(w.encode('utf-8'),c) for w,c in data_count]
            w = csv.writer(f, delimiter = ',')
            w.writerows(data_encoded)
#             pickle.dump(data_encoded, f) #TODO: fix
        return data_count

    def document_features(self,document):
        document = self.clean_text(document)
        document_words = set(document.split())
        features = {}
        for word in self.get_word_features():
            features['contains(%s)' % word] = (word in document_words)
        return features
    
    def get_word_features(self):
        with open('word_features','r') as word_file:
            word_features = pickle.load(word_file)
        return word_features
        
    
    def get_tags(self,data,n=100):
        tag_count = self.count_data([tag for entry in data for tag in entry['tags']],'tag_count.txt')
        tags = [tag for tag,count in tag_count if count > n]
#         tags = tags[:2] #TODO: remove
        return tags
    
    def train(self,data):
        tags = self.get_tags(data)
        print(len(tags))
#         data = [entry for entry in data if entry['tags']]
        for entry in data:
            entry['features'] = self.document_features(entry['text'])
#         print(len(data))
        classifier = {}
        for tag in tags:
            print(tags.index(tag, ))
#             featuresets = [(self.document_features(entry['text']), tag in entry['tags']) for entry in data if len(entry['tags']) > 0]
#             featuresets = [(entry['features'], tag in entry['tags']) for entry in data if entry['tags']]
            featuresets = [(entry['features'], tag in entry['tags']) for entry in data]
            classifier[tag] = nltk.NaiveBayesClassifier.train(featuresets)
            logging.info(str(tag)+': '+ str(classifier[tag].most_informative_features(10)))
        with open('classifier','w') as classifier_file:
            pickle.dump(classifier,classifier_file)
        return classifier
    
    def test(self,test_set):
        tags = self.get_tags(test_set)
        accuracy = {}
        data = self.clean_data(test_set)
        print("extracting features")
        for entry in data:
            entry['features'] = self.document_features(entry['text'])
        print("loading classifier")        
        with open('classifier','r') as classifier_file:
            classifier = pickle.load(classifier_file)
        print("testing:")                
        print(len(tags))
        for tag in tags:
            print(tags.index(tag, ))
            if tag not in classifier:
                continue
#             featuresets = [(self.document_features(entry['text']), tag in entry['tags']) for entry in test_set]
            featuresets = [(entry['features'], tag in entry['tags']) for entry in data]
            accuracy[tag] = nltk.classify.accuracy(classifier[tag],featuresets)
            print(tag+': '+str(accuracy[tag]))
        return accuracy 
    
    def test_tag(self,documents,tag,thresh):
        N = len(documents)
        print("cleaning")
        data = self.clean_data(documents)
        print("loading classifier")        
        with open('classifier','r') as classifier_file:
            classifier = pickle.load(classifier_file)
        if tag not in classifier:
            return []
        print("extracting features")
        for entry in data:
            entry['features'] = self.document_features(entry['text'])
        probs = []
        print("testing:")                
        for entry in data:
            prob = classifier[tag].prob_classify(entry["features"])
            if prob.prob(True) > thresh:
                probs.append((prob.prob(True),entry))
        probs = sorted(probs,reverse=True)        
        return probs
    
    def test_doc(self,document,tags,thresh):
        document["text"] = self.clean_text(document["text"])
        document["features"] = self.document_features(document["text"])
        with open('classifier','r') as classifier_file:
            classifier = pickle.load(classifier_file)
        probs = []
        for tag in tags:
            if tag not in classifier:
                continue
            prob = classifier[tag].prob_classify(document["features"])
            if prob.prob(True) > thresh:
                probs.append((prob.prob(True),tag))
        probs = sorted(probs,reverse=True)
        return probs
# most probable tag: input: status, n  output: list (top n most probable, + score)     

    def create_word_features(self,data):
        word_count = self.count_data([w for entry in data for w in entry['text'].split()],'word_count.txt')
        max_count = max([v for k,v in word_count])
#         word_features = [k for (k,v) in word_count if v > 10 and v < max_count/10] #TODO: use something more intelligent
        word_features = [k for k,v in word_count if v > 10 and v not in heb_stopwords]
#         print(len(word_features))
        with open('word_features','w') as word_file:
            pickle.dump(word_features,word_file)

    def clean_data(self,data):
        for i in range(len(data)):
            data[i]['text'] = self.clean_text(data[i]['text'])
        return data 

    def classify(self,data):
        print('cleaning')
        self.clean_data(data)
        print('extracting word features')
        self.create_word_features(data)
        print('training')
        classifier = self.train(data)


if __name__ == '__main__':
    data = get_json_data('db3')
    random.shuffle(data)
    tagged = [d for d in data if d["tags"]]
    untagged = [d for d in data if not d["tags"]]
    N = len(tagged)
    untagged = untagged[:N]#TODO: remove
    print(N)
    train_data = tagged[:-N/10]
    test_data = tagged[-N/10:]
    at = AutoTag()

#     print('classifying')
#     at.classify(train_data)
# 
#     print("testing")
#     at.test(test_data)
# 
#     print("tag")
#     dir = 'suggestions/'
#     print(len(untagged))
#     tags = at.get_tags(tagged)
#     new_tagged = [] 
#     for tag in tags[1:]:
#         print(tags.index(tag, ))
#         print(tag)
#         stats = at.test_tag(untagged, tag, 0.01)
#         for stat in stats:
#             entry = stat[1]
#             entry["tags"].append(tag)
#             entry["features"] = []
#             with open(dir+str(entry['id']), 'w') as f:
#                 json.dump(entry,f, indent = 4, separators=[',',': '])
#             new_tagged.append(entry)

#     for entry in untagged:
#         print(len(new_tagged))
#         if(len(new_tagged) > 10):
#             break
#         print(untagged.index(entry, ))
#         tags = at.test_doc(entry, tags, 0.01)
#         if tags:
#             entry["tags"] = tags
#             entry["features"] = []
#             with open(entry['id'], 'w') as f:
#                 json.dump(entry,f, indent = 4, separators=[',',': '])
#             new_tagged.append(entry)

#     with open('suggeted_tags', 'w') as f:
#         json.dump(new_tagged,f, indent = 4, separators=[',',': '])
        
#     tags = at.get_tags(data)
#     documents = [entry['text'] for entry in data]
#     print('test doc')
#     print(at.test_doc(data[0], tags, 0))
#     print('test tag')
#     print(tags[1])
#     print(at.test_tag(data[:200], tags[1], 0)[0][1])

    print('hk freq')
    dir = 'hk_freq/'
#     for entry in data:
#         entry['text'] = at.clean_text(entry['text'])
#     word_count = at.count_data([w for entry in data for w in entry['text'].split()],dir+'total')
#     words = [w for w,c in word_count if c > 40]
#     with open(dir+'freqs.csv', 'wb') as csvfile:
#         writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         writer.writerow(['HK']+words)
#   
#     hkwords = {}
#     with open('db3/1002_1002.json') as data_file:
#         data_json = json.load(data_file)
#     for json_entry in data_json:
#         name = json_entry['fields']['name']
#         print(name) 
#         if not name:
#             continue
#         name = name.encode('utf-8')
#         word_count = at.count_data([w for entry in data for w in entry['text'].split() if entry["feed"] == json_entry['pk']],dir+name)
# #         word_count = [(w.encode('utf-8'),c) for w,c in word_count]
#         word_dict = {w:c for w,c in word_count}
#         hkwords[name] = []
#         for word in words:
#             if word not in word_dict:
#                 hkwords[name].append(str(0))
#             else:
#                 hkwords[name].append(str(word_dict[word])) 
#         with open(dir+'freqs.csv', 'a') as csvfile:
#             writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#             writer.writerow([name]+hkwords[name])
#     
#     with open(dir+'freqs_t.csv', 'a') as csvfile:
#         writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         hks = hkwords.keys()
#         writer.writerow(['word']+hks)
#         for word in words:
#             writer.writerow([word] + [hkwords[hk][words.index(word)] for hk in hks])
         
    a = izip(*csv.reader(open(dir+"freqs.csv", "rb")))
    csv.writer(open(dir+"freqs_t.csv", "wb")).writerows(a)
    
#     print(tags)
#     bag = bag_of_words(statuses,word_features)
#     distmat = get_distances(bag)
#     clusters = cluster_sentences(distmat)
#     for i in clusters:
#         print()
#         print(statuses[i])
#     cluster_words(count)
#     words = split_pattern(count)
#     patterns = get_patterns(count,0.8)
#     get_tags(statuses, count)


# interface - get tag: statuses sorted by probability to be tagged
#             get status: tags sorted by probability to fit



#TODO: use data object (with id, text and tags) instead of tuple
#TODO: fix Dependencies
#TODO: print most informative features to file
#TODO: clean text in train, and test