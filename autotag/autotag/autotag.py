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
import codecs
import difflib
import string
import numpy
# import scipy.sparse
# import scipy.spatial
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import os
import random
import csv
from utils import get_json_data, get_json, clean_text
import logging
from stopwords import heb_stopwords
# import cPickle


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
    '''
    Automated tagging
    '''
    
    def __init__(self, res_dir='res'):
        self.wnl = WordNetLemmatizer()
#         self.res_dir = 'res'
#         self.res_dir = os.path.dirname(__file__)
        self.res_dir = res_dir
        if not os.path.exists(self.res_dir):
            os.mkdir(self.res_dir)

        logname = os.path.join(self.res_dir, 'log')
        open(logname,'w')
        logging.basicConfig(filename=logname, level=logging.INFO)

#     def set_res_dir(self, res_dir):
#         self.res_dir = res_dir
#         if not os.path.exists(self.res_dir):
#             os.mkdir(self.res_dir)

        
    def count_data(self,data,name):
        data_count = Counter(data)
        data_count = sorted(data_count.items(), key=lambda item: item[1],reverse = True)
#         with open(name+'.json', 'w') as f:
# #             data_encoded = [(w,c) for w,c in data_count]
#             data_encoded = [(w.encode('utf-8'),c) for w,c in data_count]
#             json.dump(data_encoded, f, ensure_ascii=False, indent = 4, separators=[',',': ']) #TODO: fix
        with open(os.path.join(self.res_dir, name+'.csv'), 'w') as f:
#             print(data_count)
#             data_encoded = [(w,c) for w,c in data_count]
            data_encoded = [(w.encode('utf-8'),c) for w,c in data_count]
            w = csv.writer(f, delimiter = ',')
            w.writerows(data_encoded)
#             pickle.dump(data_encoded, f) #TODO: fix
        return data_count

    def get_document_features(self,document):
        '''
        Extract features from the document.
        Current supported features are the existence of a word in the document  
        
        :param document: a dictionary with 'text' key and 'tags' key.
        '''
        document = clean_text(document)
        document_words = set(document.split())
        features = {}
        for word in self.get_word_features():
            features['contains(%s)' % word] = (word in document_words)
        return features
    
    def create_word_features(self,data):
        '''
        Create word list from data. Filtering rare words and stopwords.
        
        :param data: list of entries with 'text' key and 'tags' key.
        '''
        word_count = self.count_data([w for entry in data for w in entry['text'].split()], 'word_count')
        word_count = sorted(word_count, key = operator.itemgetter(1), reverse = True)
#         max_count = max([v for k,v in word_count])
#         word_features = [k for (k,v) in word_count if v > 10 and v < max_count/10] #TODO: use something more intelligent
        word_features = [k for k,v in word_count if v > 10 and v not in heb_stopwords] #TODO: I18n 
#         print(len(word_features))
        try:
            fpath = os.path.join(self.res_dir, 'word_features')
            with open(fpath,'w') as word_file:
                pickle.dump(word_features,word_file)
        except Exception:
            error_msg = 'could not write file'
            raise Exception(error_msg)

    def get_word_features(self):
        '''
        Get the feature word list from file 
        '''
        fpath = os.path.join(self.res_dir, 'word_features')
        with open(fpath, 'r') as word_file:
            word_features = pickle.load(word_file)
        return word_features
    
    
    def get_tags(self,data=[],n=50):
        '''
        Get tags. if given data, the tags are extracted from the data, else, from the classifiers
         
        :param data: list of entries with 'text' key and 'tags' key.
        :param n: minimal amount of tagged entries, for a tag to be included in tags list.  
        '''
        if not data:
            tags = []
            for fname in os.listdir(self.res_dir):
                fileName, fileExtension = os.path.splitext(fname)
                m = re.match('.clsfr',fileExtension)
                if not m:
                    continue
                tags.append(fileName)        
    #         tags = tags[:2] #TODO: remove
            return tags            
        tag_count = self.count_data([tag for entry in data for tag in entry['tags']],'tag_count')
        tags = [tag for tag,count in tag_count if count >= n]
#         tags = tags[:2] #TODO: remove
        return tags

    def get_most_informative_features(self, tag, n=10):
        '''
        Get the most informative features.
        
        :param tag:
        :param n: number of features that should return.
        '''
        classifier = self.load_classifier(tag)
#         features = [unicode(str(feature), encoding = 'utf-8', errors = 'replace') for feature in classifier.most_informative_features(n)]
        features = classifier.most_informative_features(n)
        features = [(classifier._feature_probdist[(True, fname)].logprob(fval), re.sub(r'contains\((.+)\)', r'\1', fname)) for fname, fval in features if fval and (True, fname) in classifier._feature_probdist]        
        return features

    def get_most_informative_features_from_data(self, important_data, base_data, n=10):
        '''
        Get the n most informative features accordnig to given data.
        
        :param important_data: data for which the features are important.
        :param base_data: general data, for which the features are not particularly important/
        :param n: number of features that should return.
        '''
        for entry in important_data:
            entry['features'] = self.get_document_features(entry['text'])
        for entry in base_data:
            entry['features'] = self.get_document_features(entry['text'])
        featuresets = []
        featuresets.extend([(entry['features'], True) for entry in important_data])
        featuresets.extend([(entry['features'], False) for entry in base_data])
        classifier = nltk.NaiveBayesClassifier.train(featuresets)
        self.dump_classifier(classifier, 'temp')
        features = self.get_most_informative_features('temp', n)
        os.remove(os.path.join(self.res_dir, 'temp.clsfr'))
        return features

    def dump_classifier(self, classifier, name):
        '''
        Dump classifier to file. 
        
        :param classifier: classifier to dump.
        :param name: name for the classifier.
        '''
        fname = unicode(name)+'.clsfr'
        try:
            with open(os.path.join(self.res_dir, fname),'w') as classifier_file:
                pickle.dump(classifier,classifier_file)
        except:
            error_msg = 'could not write file'
            raise Exception(error_msg)
    
    def load_classifier(self, name):
        '''
        Load classifier from file, by name. 
        
        :param name: classifier name
        '''
        fname = unicode(name) + '.clsfr'
        fpath = os.path.join(self.res_dir, fname)
        if not os.path.isfile(fpath):
            return None
        with open(fpath,'r') as classifier_file:
            classifier = pickle.load(classifier_file)
        return classifier
                
    def train(self, data, n=50, tags=None):
        '''
        Train classifiers for the given data, with the given tags. 
        
        :param data: list of entries with 'text' key and 'tags' key.
        :param n: minimal amount of tagged entries, for a tag to be included in tags list.
        :param tags: tags to use for classification.
        '''
        if not tags:
            tags = self.get_tags(data, n)
        print('tags: ' + str(tags))
#         data = [entry for entry in data if entry['tags']]
        for entry in data:
            entry['features'] = self.get_document_features(entry['text'])
        for tag in tags:
            print('training tag: ' + str(tag))
            self.train_tag(data, tag)


    def train_tag(self, data, tag):
        '''
        Train classifier for the given data, with the given tag. 
        
        :param data: list of entries with 'text' key and 'tags' key.
        :param tag: tag to use for classification.
        '''
#             featuresets = [(entry['features'], tag in entry['tags']) for entry in data if entry['tags']]
        featuresets = [(entry['features'], tag in entry['tags']) for entry in data]
        classifier = nltk.NaiveBayesClassifier.train(featuresets)
        self.dump_classifier(classifier, tag)
        features = self.get_most_informative_features(tag, 10)
        logging.info(unicode(tag)+u': '+ unicode(features))
                
    def test(self,data,tags=None):
        '''
        Test classifiers accuracy on data. if tags are not given, they are extracted from the classifiers. 
        
        :param data: list of entries with 'text' key and 'tags' key.
        :param tags: tags to test.
        '''
        accuracy = {}
        data = self.clean_data(data)
        if not tags:
            tags = self.get_tags()
        for entry in data:
            entry['features'] = self.get_document_features(entry['text'])
        print(len(tags))
        for tag in tags:
            classifier = self.load_classifier(tag)
            if not classifier:
                continue
#             featuresets = [(self.get_document_features(entry['text']), tag in entry['tags']) for entry in test_set]
            featuresets = [(entry['features'], tag in entry['tags']) for entry in data]
            accuracy[tag] = nltk.classify.accuracy(classifier,featuresets)
            print(str(tag)+': '+str(accuracy[tag]))
        return accuracy 
    
    def test_tag(self,documents,tag,thresh=0.3):
        '''
        test which documents should be tagged with the given tag 
        
        :param documents: list of entries with 'text' key and 'tags' key.
        :param tag: tag to test.
        :param thresh: threshold for the tag probability.
        '''
        probs = []
        N = len(documents)
        data = self.clean_data(documents)
        classifier = self.load_classifier(tag)
        if not classifier:
            return probs
        for entry in data:
            entry['features'] = self.get_document_features(entry['text'])
        for entry in data:
            prob = classifier.prob_classify(entry["features"])
            if prob.prob(True) > thresh:
                probs.append((prob.prob(True),entry))
        probs = sorted(probs,reverse=True)
        return probs
    
    def test_doc(self,document,tags,thresh=0.3):
        '''
        test which tags should tag the given document 
        
        :param document: an entry with 'text' key.
        :param tags: tags to test.
        :param thresh: threshold for the tag probability.
        :return probs: list of the most probable tags and their probability.
        '''
        document["text"] = clean_text(document["text"])
        document["features"] = self.get_document_features(document["text"])
        probs = []
        for tag in tags:
            classifier = self.load_classifier(tag)
            if not classifier:
                continue
            prob = classifier.prob_classify(document["features"])
            if prob.prob(True) > thresh:
                probs.append((prob.prob(True),tag))
        probs = sorted(probs,reverse=True)
        return probs
# most probable tag: input: status, n  output: list (top n most probable, + score)     
            
    def clean_data(self,data):
        for i in range(len(data)):
            data[i]['tags'] = [unicode(tag, encoding = 'utf-8') for tag in data[i]['tags']]
            data[i]['text'] = clean_text(data[i]['text'])
        return data 

    def classify(self, data, n=50, tags=None):
        '''
        build classifiers for the given data.
        
        :param data: list of entries with 'text' key and 'tags' key.
        :param n: minimal amount of tagged entries, for a tag to be included in tags list.
        :param tags: tags to use for classification
        '''
        print('cleaning')
        self.clean_data(data)
        print('extracting word features')
        self.create_word_features(data)
        print('training')
        self.train(data, n, tags)
    

def hk_freq(data_dir, hk_dir):
    print('hk freq')
    data = get_json_data(data_dir)
    at = AutoTag()
    for entry in data:
        entry['text'] = clean_text(entry['text'])
    if not os.path.isdir(hk_dir):
        os.mkdir(hk_dir)
    with open(hk_dir+'total', 'w') as f:
        pass
    word_count = at.count_data([w for entry in data for w in entry['text'].split()],hk_dir+'total')
    words = [w.encode('utf-8') for w,c in word_count if c > 40]
    with open(hk_dir+'freqs.csv', 'wb') as csvfile:
#         data_encoded = [w.encode('utf-8') for w,c in word_count if c > 40]
        w = csv.writer(csvfile, delimiter = ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow([u'HK']+words)
#         csvfile.write(','.join([u'HK']+words) + '\n')
   
    hkwords = {}
    data_json = get_json(data_dir)
    for json_entry in data_json:
        if json_entry['model'] != "facebook_feeds.facebook_feed":
            continue
        name = json_entry['fields']['name']
        print(name) 
        if not name:
            continue
        name = name.encode('utf-8')
        word_count = at.count_data([w for entry in data for w in entry['text'].split() if entry["feed"] == json_entry['pk']],hk_dir+name)
        word_dict = {w.encode('utf-8'):c for w,c in word_count}
        hkwords[name] = []
        for word in words:
            if word not in word_dict:
                hkwords[name].append(str(0))
            else:
                hkwords[name].append(str(word_dict[word])) 
        with open(hk_dir+'freqs.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
#             writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([name]+hkwords[name])
     
    
    with open(hk_dir+'freqs_t.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for name in hkwords:
            writer.writerow([name]+hkwords[name])
#         hks = hkwords.keys()
#         for word in words:
#             writer.writerow([word] + [hkwords[hk][words.index(word)] for hk in hks])

#     dir = 'hk_freq1/'
# #     hk_freq('db4', dir)
#     a = izip(*csv.reader(open(dir+"freqs.csv", "rb")))
#     csv.writer(open(dir+"freqs_t.csv", "wb")).writerows(a)


def main():
    main_data = get_json_data('res/db4')
    random.shuffle(main_data)
    tagged = [d for d in main_data if d["tags"]]
    untagged = [d for d in main_data if not d["tags"]]
    N = len(tagged)
    untagged = untagged[:N]#TODO: remove
    print(N)
    train_data = tagged[:-N/10]
    test_data = tagged[-N/10:]
    path = os.getcwd()
    at = AutoTag(os.path.join(path, 'res'))
    tags = at.get_tags()
    print(tags)

    print('classifying')
    at.classify(train_data, tags=tags)
   
    print("testing")
    print(at.test(test_data))

    tagged = [d for d in main_data if '231' in d["tags"]]
    untagged = [d for d in main_data if '231' not in d["tags"]][:100]
    print(len(tagged))
    print(at.get_most_informative_features_from_data(tagged, untagged, 10))


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

    

if __name__ == '__main__':
    main()

# interface - get tag: statuses sorted by probability to be tagged
#             get status: tags sorted by probability to fit



#TODO: use data object (with id, text and tags) instead of tuple
#TODO: fix Dependencies
#TODO: print most informative features to file
#TODO: clean text in train, and test