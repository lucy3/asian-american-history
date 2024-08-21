"""
Copied from https://github.com/ddemszky/textbook-analysis/blob/master/get_descriptors.py
But removed the lines associated with getting descriptors for famous people, and
removed merge entities. 
Added "chunk_size" variable to adjust for books where each line is very long. 
"""

import argparse
from helpers import *
import spacy
import os
import math

parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', required=True)
parser.add_argument('--output_prefix', required=True)
parser.add_argument('--people_terms', required=True)

args = parser.parse_args()

def run_depparse(people, aapi_term_dict, textbook_lines, title, nlp): 
    '''
    Get adjectives and verbs associated with frequent named entities
    and common nouns referring to people.
    @inputs: 
    - possible_marks: words that may mark common nouns with a social group, e.g. "black"
    - word2dem: word to demographic category
    - textbook_lines: strings of textbook content in a list
    - title: title of book
    - outfile: opened file
    - nlp: spacy pipeline
    '''
    print("Running dependency parsing for", title)
    # Break up every textbook into 2k line chunks to avoid spaCy's text length limit 
    j = 0
    k = 0
    num_lines = len(textbook_lines)
    chunk_size = 1000
    if num_lines < 1000: 
        # this is a book where lines may be long
        chunk_size = 10
    res = []
    for i in range(0, num_lines, chunk_size):
        chunk = '\n'.join(textbook_lines[i:i+chunk_size])
        doc = nlp(chunk)
        k += 1
        print("Finished part", k, "of", math.ceil(num_lines/chunk_size))
        
        for chunk in doc.noun_chunks: 
            j += 1
            noun_phrase = chunk.text.lower()
            head_token = chunk.root.text.lower()
            
            # check that head of noun is a person
            if head_token in people:
                target_term = chunk.root.head.text.lower()
                noun_tokens = set(noun_phrase.split())
                dem = 'other'
                
                # check if noun chunk is aapi
                for term_length in aapi_term_dict: 
                    if term_length == 1: 
                        # unigram
                        unigram_terms = set(aapi_term_dict[term_length])
                        overlap = unigram_terms & noun_tokens
                        if overlap: 
                            dem = 'aapi'
                        # indian edge case
                        if 'online' not in args.output_prefix and \
                            ('indian' in overlap or 'indians' in overlap) and \
                            'asian' not in set(noun_tokens): 
                            dem = 'other'
                    else: 
                        # bigram or trigram
                        for term in aapi_term_dict[term_length]: 
                            if term in noun_phrase: 
                                dem = 'aapi'
                                break
                    if dem == 'aapi': break
                
                if chunk.root.dep_ == 'nsubj' and (chunk.root.head.pos_ == 'VERB'): 
                    res.append((str(j), title, noun_phrase, dem, target_term, chunk.root.head.pos_, chunk.root.dep_))
                
                if chunk.root.dep_ == 'nsubjpass' and chunk.root.head.pos_ == 'VERB': 
                    res.append((str(j), title, noun_phrase, dem, target_term, chunk.root.head.pos_, chunk.root.dep_))
                
                if (chunk.root.dep_ == 'obj' or chunk.root.dep_ == 'dobj') and chunk.root.head.pos_ == 'VERB': 
                    res.append((str(j), title, noun_phrase, dem, target_term, chunk.root.head.pos_, chunk.root.dep_))
                    
    return res

def main(): 
    aapi_terms, other_terms = get_people_terms(args.people_terms)
    # load spacy
    nlp = spacy.load("en_core_web_trf")
    aapi_term_dict = get_aapi_term_dict(aapi_terms, nlp)
    # load books
    books = get_book_txts(args.input_dir, splitlines=True)
    res = []
    people = aapi_terms | other_terms
    for title, textbook_lines in books.items():
        res.extend(run_depparse(people, aapi_term_dict, textbook_lines, title, nlp))
    outfile = codecs.open(args.output_prefix + '_people_descriptors.csv', 'w', encoding='utf-8')
    fieldnames = ['token_ID', 'filename', 'entity', 'category', 'word', 'POS', 'rel']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    for tup in res: 
        if type(tup[3]) == list or type(tup[3]) == set: 
            for d in tup[3]: 
                out_dict = {'token_ID' : tup[0], 
                            'filename': tup[1], 
                            'entity':tup[2], 
                            'category': d, 
                            'word':tup[4], 
                            'POS': tup[5], 
                            'rel': tup[6]
                            }
                writer.writerow(out_dict)
        else: 
            out_dict = {'token_ID' : tup[0], 
                            'filename': tup[1], 
                            'entity':tup[2], 
                            'category': tup[3], 
                            'word':tup[4], 
                            'POS': tup[5], 
                            'rel': tup[6]
                            }
            writer.writerow(out_dict)
    outfile.close()

if __name__ == '__main__':
    main()
