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

def get_data():
    print('getting data')
    with open('db/statuses.txt') as statuses_file:
        statuses_json = json.load(statuses_file)
    statuses = {status['pk']: status['fields']['content'] for status in statuses_json}
    with open('db/tags.txt') as tags_file:
        tags_data = json.load(tags_file)
    tags = {}
    for tag in tags_data:
        id = tag['fields']['object_id'] 
        if id not in tags:
            tags[id] = []
        tags[id].append(str(tag['fields']['tag']))
    data = [(statuses[i], tags[i]) for i in statuses if i in tags]
    return data
        
        
class AutoTag:
    

    def clean_text(self,text):
    #     text = re.sub('[^a-zA-Zא-ת0-9\"\\n]',' ', text)
        puncmarks = string.punctuation+'״׳'
    #     add space before punctuation   
        text = re.sub('(\S)(['+puncmarks+']+)\s','\\1 \\2 ', text)
    #     add space after punctuation   
        text = re.sub('\s(['+puncmarks+']+)(\S)',' \\1 \\2 ', text)
        text = re.sub('\s+', ' ', text)
#         f = codecs.open('clean_text','w','utf-8')
#         f.write(text)
#         f.close()
        return text
        
    def count_data(self,data,name):
        data_count = Counter(data)
        data_count = sorted(data_count.items(), key=lambda item: item[1],reverse = True)
        with open(name, 'w') as f:
            data_encoded = [(w.encode('utf-8'),c) for w,c in data_count]
            json.dump(data_encoded, f, ensure_ascii=False, indent = 4, separators=[',',': '])
        return data_count

    def document_features(self,document):
        document_words = set(document.split())
        with open('word_features','r') as word_features_file:
            word_features = pickle.load(word_features_file)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features
    
    def get_tags(self,data,n=5):
        tag_count = self.count_data([tag for status,tags in data for tag in tags],'tag_count.txt')
        tags = [tag for tag,count in tag_count if count > n]
#         tags = tags[:20]
        return tags
    
    def train(self,data):
        tags = self.get_tags(data)
        print(len(tags))
        classifier = {}
        for tag in tags:
            print(tag)
            featuresets = [(self.document_features(doc), tag in tags) for (doc,tags) in data if len(tags) > 0]
            classifier[tag] = nltk.NaiveBayesClassifier.train(featuresets)
#             classifier[tag].show_most_informative_features(5)
        with open('classifier','w') as classifier_file:
            pickle.dump(classifier,classifier_file)
        return classifier
    
    def test_tag(self,documents,tag,thresh):
        N = len(documents)
        with open('classifier','r') as classifier_file:
            classifier = pickle.load(classifier_file)
#         featuresets = [(self.document_features(d), tag in c) for (d,c) in documents]
#         print(nltk.classify.accuracy(classifier, featuresets))
        probs = []
        for i in range(N):
            if tag not in classifier:
                continue
            prob = classifier[tag].prob_classify(self.document_features(documents[i]))
            if prob.prob(True) > thresh:
                probs.append((prob.prob(True),documents[i]))
        probs = sorted(probs,reverse=True)        
        return probs
    
    def test_doc(self,document,tags,thresh):
        with open('classifier','r') as classifier_file:
            classifier = pickle.load(classifier_file)
        probs = []
        for tag in tags:
            if tag not in classifier:
                continue
            prob = classifier[tag].prob_classify(self.document_features(document))
            if prob.prob(True) > thresh:
                probs.append((prob.prob(True),tag))
        probs = sorted(probs,reverse=True)
        return probs
# most probable tag: input: status, n  output: list (top n most probable, + score)     

    def create_word_features(self,data):
        word_count = self.count_data([w for s,t in data for w in s.split()],'word_count.txt')
        max_count = max([v for k,v in word_count])
        word_features = [k for (k,v) in word_count if v > 10 and v < max_count/10]
        with open('word_features','w') as word_file:
            pickle.dump(word_features,word_file)

    def classify(self,data):
        print('cleaning')
        data = [(self.clean_text(s),t) for s,t in data]
        print('extracting word features')
        self.create_word_features(data)
        print('classifying')
        classifier = self.train(data)
#         for status,count in test_statuses:
#             print(self.test_doc(status, big_tags, word_features, classifier, 0.5))


if __name__ == '__main__':
    data = get_data()
    at = AutoTag()
#     at.classify(data)
    tags = at.get_tags(data)
    documents = [s for s,t in data]
    print(at.test_doc(documents[0], tags, 0))
    print(tags[0])
    print(at.test_tag(documents[:200], tags[0], 0)[0][1])
    
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