'''
Get topic probability of sentences containing 

python get_topic_probabilities.py --sentence_file ./logs/combined_coref_data.csv --topic_dir ./topics/topics_50/ --people_terms wordlists/people_terms.csv --output_file ./logs/combined_coref_data_topics_50.csv
'''
import numpy as np
import argparse
from helpers import *
import csv
from tqdm import tqdm
import torch
import spacy

parser = argparse.ArgumentParser()

parser.add_argument('--sentence_file', required=True)
parser.add_argument('--topic_dir', required=True)
parser.add_argument('--people_terms', required=True)
parser.add_argument('--output_file', required=True)

args = parser.parse_args()

def main(): 
    '''
    Sentence order follows the order in sentence_file. 
    
    The output augments sentence_file with the aapi terms in the sentence, and also its topic probabilities.
    '''
    aapi_terms, other_terms = get_people_terms(args.people_terms)
    people = aapi_terms | other_terms
    
    spacy.require_gpu()
    
    doc_topic_file = '%sdoc-topics.gz' % args.topic_dir
    doc_topics = open(doc_topic_file).read().splitlines() # list of topics
    
    print("Getting doc topic matrix...")
    doc_topics_matrix = [] # doc x topics
    for i, doc in enumerate(doc_topics): 
        contents = doc.split('\t')
        topics = [float(i) for i in contents[2:]]
        doc_topics_matrix.append(topics)
    doc_topics_matrix = np.array(doc_topics_matrix)
    
    nlp = spacy.load('en_core_web_trf')
    
    aapi_term_dict = get_aapi_term_dict(aapi_terms, nlp)
    
    print("Reading in lines...")
    num_lines = 874127 # hardcoded, calculated using wc -l on sentence_file
    with open(args.output_file, 'w', encoding='utf-8') as csvfile:
        fieldnames = ['state', 'book_id', 'book_filename', 'subject', 'sentence_id', 'sentence', 'aapi']
        for i in range(doc_topics_matrix.shape[1]): 
            fieldnames.append('topic_' + str(i))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        with open(args.sentence_file, 'r', encoding='utf-8') as infile: 
            reader = csv.DictReader(infile)
            sent_num = 0
            for row in tqdm(reader, total=num_lines): 
                sent = row['sentence'] 
                tokens = clean_text(sent,
                                 stem=False,
                                 remove_short=True,
                                 remove_stopwords=True, 
                                 round_dates=True)
                if len(tokens) < 5: 
                    for i in range(doc_topics_matrix.shape[1]): 
                        row['topic_' + str(i)] = ''
                    row['aapi'] = ''
                    writer.writerow(row)
                    continue

                doc = nlp(sent)
                
                terms = set()
                # check if there are aapi people in sentence
                for chunk in doc.noun_chunks: 
                    noun_phrase = chunk.text.lower()
                    head_token = chunk.root.text.lower()
                    if head_token in people:
                        noun_tokens = set(noun_phrase.split())
                        for term_length in aapi_term_dict: 
                            if term_length == 1: 
                                # unigram
                                unigram_terms = set(aapi_term_dict[term_length])
                                overlap = unigram_terms & noun_tokens
                                for term in overlap: 
                                    # edge case where "Indian" in textbooks
                                    # does not usually refer to Indians from Asia
                                    if row['state'] != 'Online' and \
                                    term.startswith('indian') and 'Asian ' + term.title() not in sent: 
                                        continue
                                    terms.add(term)
                            else: 
                                # bigram or trigram
                                for term in aapi_term_dict[term_length]: 
                                    if term in noun_phrase: 
                                        terms.add(term)
                if terms: 
                    row['aapi'] = ', '.join(terms)
                else: 
                    row['aapi'] = ''
                for i in range(doc_topics_matrix.shape[1]): 
                    row['topic_' + str(i)] = round(doc_topics_matrix[sent_num][i], 5)
                writer.writerow(row)
                sent_num += 1

if __name__ == '__main__':
    main()