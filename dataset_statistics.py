"""
For each book and overall, outputs
the following: 
# of sentences 
# of tokens
# of unique tokens (vocabulary)

In addition, it outputs a file 
"""

import spacy
from helpers import *
from collections import defaultdict, Counter
import json
import argparse
import os
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', required=True)
parser.add_argument('--output_prefix', required=True)

args = parser.parse_args()

def main(): 
    nlp = spacy.load('en_core_web_trf', exclude=["transformer", "ner", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    nlp.add_pipe('sentencizer')
    
    books = get_book_txts(args.input_dir, splitlines=True)
    
    num_sents = Counter()
    num_tokens = Counter()
    token_set = defaultdict(set)
    
    for title, textbook_lines in books.items():
        print(title)
        for line in tqdm(textbook_lines):
            doc = nlp(line, disable=["transformer", "ner", "tagger", "parser", "attribute_ruler", "lemmatizer"])
            
            num_sents[title] += len(list(doc.sents))
            num_tokens[title] += len(doc)
            for token in doc: 
                token_set[title].add(token.text.lower())
    
    data = {}
    all_tokens = set()
    for title in num_sents: 
        data[title] = [num_sents[title], num_tokens[title], len(token_set[title])]
        all_tokens.update(token_set[title])
    
    data['OVERALL'] = [sum(num_sents.values()), sum(num_tokens.values()), len(all_tokens)]
    
    df = pd.DataFrame.from_dict(data, orient='index',
                       columns=['Sentences', 'Tokens', 'Unique Tokens'])
    
    df.to_csv(args.output_prefix + '_stats.csv')

if __name__ == '__main__':
    main()