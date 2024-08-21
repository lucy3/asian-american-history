#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Dora Demszky (ddemszky@stanford.edu) and Lucy Li (lucy3_li@berkeley.edu)
import codecs
import glob
import string
import nltk
import re
from collections import defaultdict, Counter
import csv

stopwords = open("wordlists/stopwords/en/mallet.txt", "r").read().splitlines()
punct_chars = list((set(string.punctuation) | {'»', '–', '—', '-',"­", '\xad', '-', '◾', '®', '©','✓','▲', '◄','▼','►', '~', '|', '“', '”', '…', "'", "`", '_', '•', '*', '■'} - {"'"}))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
sno = nltk.stem.SnowballStemmer('english')
printable = set(string.printable)

def clean_text(text,
               remove_stopwords=True,
               remove_numeric=True,
               stem=False,
               remove_short=True, 
               round_dates=False):
    # lower case
    text = text.lower()
    # eliminate urls
    text = re.sub(r'http\S*|\S*\.com\S*|\S*www\S*', ' ', text)
    # substitute all other punctuation with whitespace
    text = replace.sub(' ', text)
    # replace all whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    # strip off spaces on either end
    text = text.strip()
    # make sure all chars are printable
    text = ''.join([c for c in text if c in printable])
    words = text.split()
    if remove_stopwords:
        words = [w for w in words if w not in stopwords]
    if remove_numeric:
        words = [w for w in words if not w.isdigit()]
    if round_dates: 
        new_words = []
        for w in words: 
            if len(w) == 4 and (w.startswith('1') or w.startswith('2')) and w.isdigit(): 
                rounded = str(int(round(int(w), -1)))
                new_words.append(rounded)
            else: 
                new_words.append(w)
        words = new_words
    if stem:
        words = [sno.stem(w) for w in words]
    if remove_short:
        words = [w for w in words if len(w) >= 3]
    return words

def split_terms_into_sets(people_terms_path): 
    '''
    This is customized to split terms into AAPI words
    or non-AAPI words (these are allocated to "not_marks"). 
    '''
    possible_marks = set() 
    not_marks = set() # everything else
    with open(people_terms_path, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split(',')
            term = contents[0].lower()
            if contents[2] == 'aapi': 
                possible_marks.add(term)
            else: 
                not_marks.add(term)
    return possible_marks, not_marks

def get_word_to_category(people_terms_path): 
    '''
    word2dem = {'bridesmaid':['women'], 'latina': ['women', 'latinx']}
    This was customized for AAPI vs. all other terms. 
    '''
    word2dem = defaultdict(set)
    with open(people_terms_path, 'r') as infile: 
        for line in infile: 
            contents = line.strip().split(',')
            term = contents[0]
            category = contents[1]
            if category == 'aapi': 
                word2dem[term].add(category)
                if '-' in term: 
                    new_term = term.replace('-', ' ')
                    word2dem[new_term].add(category)
            else: 
                word2dem[term].add('other')
    return word2dem

def get_aapi_term_dict(aapi_terms, nlp): 
    aapi_terms_dict = defaultdict(list) # { term token length : terms } 
    for term in aapi_terms: 
        term_doc = nlp(term, disable=['parser', 'tagger', 'entity'])
        new_term = []
        for tok in term_doc: 
            new_term.append(tok.text)
        aapi_terms_dict[len(new_term)].append(term)
        if '-' in term: 
            term = term.replace('-', ' ')
            term_doc = nlp(term, disable=['parser', 'tagger', 'entity'])
            new_term = []
            for tok in term_doc: 
                new_term.append(tok.text)
            aapi_terms_dict[len(new_term)].append(term)
    return aapi_terms_dict

def get_people_terms(people_terms_path): 
    '''
    Custom for this project. 
    Returns two sets of terms: one for AAPI, one for everyone else
    '''
    aapi_terms = set()
    other_terms = set()
    with open(people_terms_path, 'r') as infile: 
        reader = csv.reader(infile)
        for row in reader: 
            term = row[0].lower()
            category = row[1]
            if category == 'aapi': 
                aapi_terms.add(term)
                aapi_terms.add(term.replace('-', ' '))
            else: 
                other_terms.add(term)
    return aapi_terms, other_terms

def get_people_terms_by_cat(people_terms_path): 
    '''
    Returns a set of all people terms and a mapping
    from term to race/ethnicity category
    '''
    race_eth_cats = {} # term to race/ethnicity category
    all_terms = set() # all terms that refer to people
    with open(people_terms_path, 'r') as infile: 
        reader = csv.reader(infile)
        for row in reader: 
            term = row[0].lower()
            category = row[1]
            cat_type = row[2]
            if cat_type == 'race/ethnicity': 
                race_eth_cats[term] = category
                race_eth_cats[term.replace('-', ' ')] = category
            all_terms.add(term)
    # a term cannot have a term within it that is already considered
    # a person to avoid double counting their presence 
    all_terms_clean = set()
    for term in all_terms: 
        if '-' in term: 
            tokens = set(term.replace('-', ' ').split())
            if tokens & all_terms: 
                continue
        all_terms_clean.add(term)
    return race_eth_cats, all_terms_clean

def get_book_txts(path, splitlines=False, verbose=False):
    print('Getting books from', path)
    bookfiles = sorted([f for f in glob.glob(path + '/*.txt')])
    books = {}
    for f in bookfiles:
        txt = codecs.open(f, 'r', encoding='utf-8').read()
        if splitlines:
            txt = txt.splitlines()
        title = f.split('/')[-1].replace('.txt', '')
        books[title] = txt
        if verbose: 
            print(title)
    print("Finished getting books.")
    return books