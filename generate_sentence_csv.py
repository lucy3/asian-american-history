"""
Output format: 
TX,bookid,book_filename,history,sentence_id,sentence
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
parser.add_argument('--state_name', required=True)
parser.add_argument('--output_prefix', required=True)

args = parser.parse_args()

def main(): 
    nlp = spacy.load('en_core_web_trf', exclude=["transformer", "ner", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    nlp.add_pipe('sentencizer')
    
    books = get_book_txts(args.input_dir, splitlines=True)
    book_id = 0
    sent_id = 0
    
    with open(args.output_prefix + '_data.csv', 'w') as csvfile:
        fieldnames = ['state', 'book_id', 'book_filename', 'subject', 'sentence_id', 'sentence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for title, textbook_lines in books.items():
            print(title)
            for line in tqdm(textbook_lines):
                doc = nlp(line, disable=["transformer", "ner", "tagger", "parser", "attribute_ruler", "lemmatizer"])
                for sent in doc.sents: 
                    d = {}
                    d['state'] = args.state_name
                    d['book_id'] = book_id
                    d['book_filename'] = title
                    d['subject'] = 'history'
                    d['sentence_id'] = sent_id
                    d['sentence'] = sent.text.strip().encode('utf-8')
                    writer.writerow(d)
                    sent_id += 1
            book_id += 1


if __name__ == '__main__':
    main()