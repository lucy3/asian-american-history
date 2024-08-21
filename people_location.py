'''
Author: Lucy Li

@input: coref resolved texts
@output: 
- location_result: a json of {term : [sentence IDs]} for each book
'''
import spacy
from helpers import *
from collections import defaultdict
import json
import argparse
import os
from tqdm import tqdm
from nltk import ngrams

parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', required=True)
parser.add_argument('--output_prefix', required=True)
parser.add_argument('--people_terms', required=True)

args = parser.parse_args()

def get_n_gramlist(nngramlist, toks, n=2):   
    # stack overflow said this was fastest
    for s in ngrams(toks,n=n):        
        nngramlist.append(' '.join(s))                
    return nngramlist

def main(): 
    aapi_terms, other_terms = get_people_terms(args.people_terms)
    race_eth_cats, all_terms = get_people_terms_by_cat(args.people_terms)
    
    # Load your usual SpaCy model (one of SpaCy English models)
    nlp = spacy.load('en_core_web_trf')
    
    # tokenize aapi terms
    # terms range from 1 to 3 tokens
    aapi_terms_dict = get_aapi_term_dict(aapi_terms, nlp)

    # load books
    books = get_book_txts(args.input_dir, splitlines=True)

    location_dir = args.output_prefix + '_people_locations'
    term_dir = args.output_prefix + '_term_locations'
    noun_dir = args.output_prefix + '_nouns'
    race_eth_dir = args.output_prefix + '_race_eth'
    name_dir = args.output_prefix + '_names'
    
    os.makedirs(location_dir, exist_ok=True)
    os.makedirs(noun_dir, exist_ok=True)
    os.makedirs(race_eth_dir, exist_ok=True)
    os.makedirs(name_dir, exist_ok=True)
    os.makedirs(term_dir, exist_ok=True)
    
    for title, textbook_lines in books.items():
        print(title)
        location_result = defaultdict(list) # {term : [sentence IDs]}
        term_result = defaultdict(list) # {term : [sentence IDs]}
        noun_result = defaultdict(list) # {term : [noun chunks]}
        race_eth_counts = Counter()
        name_counts = Counter()
        sentence_ID = 0
        for line in tqdm(textbook_lines):
            doc = nlp(line)
            
            for sent in doc.sents:
                sentence_ID += 1
                # count any appearance of AAPI terms
                toks = [tok.text.lower() for tok in sent]
                ngramlist = [tok.text.lower() for tok in sent]
                ngramlist = get_n_gramlist(ngramlist, toks, n=2)
                ngramlist = get_n_gramlist(ngramlist, toks, n=3)
                overlap = set(ngramlist) & aapi_terms
                for term in overlap: 
                    if 'online' not in args.output_prefix and \
                    term.startswith('indian') and 'asian ' + term not in set(ngramlist): 
                        continue
                    term_result[term].append(sentence_ID)
                
                # count only nouns or people appearances
                is_aapi = False # whether sentence contains AAPI mention
                for chunk in sent.noun_chunks: 
                    toks = [tok.text.lower() for tok in chunk]
                    ngramlist = [tok.text.lower() for tok in chunk]
                    ngramlist = get_n_gramlist(ngramlist, toks, n=2)
                    ngramlist = get_n_gramlist(ngramlist, toks, n=3)
                    overlap = set(ngramlist) & aapi_terms
                    for term in overlap: 
                        if 'online' not in args.output_prefix and \
                        term.startswith('indian') and 'asian' not in set(toks): 
                            continue
                        noun_result[term].append(chunk.text.strip())
                        
                    noun_phrase = chunk.text.lower()
                    head_token = chunk.root.text.lower()
                    # check that head of noun is a person
                    if head_token not in all_terms:
                        continue
                    
                    # record location of AAPI term
                    for ngram in ngramlist: 
                        # special case where in textbooks, 
                        # Indian often refers to Native Americans, not Asians
                        if 'online' not in args.output_prefix and \
                        ngram.startswith('indian') and 'asian' not in set(toks): 
                            race_eth_counts['all'] += 1
                            continue
                            
                        if ngram in aapi_terms: 
                            is_aapi = True
                            location_result[ngram].append(sentence_ID)
                        if ngram in race_eth_cats: 
                            race_eth_counts[race_eth_cats[ngram]] += 1
                        if ngram in all_terms: 
                            race_eth_counts['all'] += 1
                
                # get named people in sentence
                if is_aapi: 
                    for ent in sent.ents: 
                        if ent.label_ != 'PERSON': continue
                        name_counts[ent.text] += 1
            
        with open(noun_dir + '/' + title + '.json', 'w') as outfile: 
            json.dump(noun_result, outfile)
        
        with open(location_dir + '/' + title + '.json', 'w') as outfile: 
            json.dump(location_result, outfile)
            
        with open(term_dir + '/' + title + '.json', 'w') as outfile: 
            json.dump(term_result, outfile)
            
        with open(race_eth_dir + '/' + title + '.json', 'w') as outfile: 
            json.dump(race_eth_counts, outfile)
            
        with open(name_dir + '/' + title + '.json', 'w') as outfile: 
            json.dump(name_counts, outfile)
            
if __name__ == '__main__':
    main()