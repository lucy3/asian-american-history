import argparse
from sklearn.manifold import TSNE
import csv
import json
import numpy as np
from sklearn import mixture
from collections import defaultdict, Counter
from nltk.stem.wordnet import WordNetLemmatizer
import scipy

ROOT = '/data0/lucy/asian-american-textbooks/'
GLOVE = '/data/dbamman/glove/glove.840B.300d.txt'
CA_DESCRIPTORS = ROOT + 'results/ca_people_descriptors.csv'
TX_DESCRIPTORS = ROOT + 'results/tx_people_descriptors.csv'

def get_glove_vectors(vocab): 
    glove_vecs = {}
    with open(GLOVE, 'r') as infile:
        for line in infile: 
            contents = line.rstrip().split(" ")
            word = contents[0]
            if word in vocab: 
                vec = np.array([float(i) for i in contents[1:]])
                glove_vecs[word] = vec
    return glove_vecs
                
def reduce_vectors(vectors): 
    vocab_list = sorted(vectors.keys())
    m = []
    for word in vocab_list: 
        m.append(vectors[word])
    m = np.array(m)
    twod_m = TSNE(n_components=2, random_state=32).fit_transform(m)
    ret = {}
    for i, word in enumerate(vocab_list): 
        vec = list(twod_m[i])
        ret[word] = vec
    return ret
                
def get_verbs(input_file): 
    agent_verbs = set()
    patient_verbs = set()
    aapi_agent_verbs = set()
    aapi_patient_verbs = set()
    verb_counts = Counter()
    with open(input_file, 'r') as infile: 
        reader = csv.DictReader(infile)
        for row in reader: 
            ID = row['token_ID']
            title = row['filename']
            category = row['category']
            word = row['word']
            pos = row['POS']
            relation = row['rel']
            if relation == 'nsubj' and pos == 'VERB':
                word = WordNetLemmatizer().lemmatize(word, 'v')
                agent_verbs.add(word)
                if category == 'aapi': 
                    aapi_agent_verbs.add(word)
                verb_counts[word] += 1
            elif pos == 'VERB' and (relation == 'dobj' or relation == 'nsubjpass'): 
                word = WordNetLemmatizer().lemmatize(word, 'v')
                patient_verbs.add(word)
                if category == 'aapi': 
                    aapi_patient_verbs.add(word)
                verb_counts[word] += 1
    return {'agent_verbs': agent_verbs, 
            'patient_verbs': patient_verbs, 
            'aapi_agent_verbs': aapi_agent_verbs, 
            'aapi_patient_verbs': aapi_patient_verbs,
            'verb_counts': verb_counts}

def write_twod_vecs(twod_vectors, state_items, outfile, state): 
    agent_verbs = state_items['agent_verbs']
    patient_verbs = state_items['patient_verbs']
    aapi_agent_verbs = state_items['aapi_agent_verbs']
    aapi_patient_verbs = state_items['aapi_patient_verbs']
    for verb in agent_verbs: 
        if verb in aapi_agent_verbs and verb in twod_vectors: 
            outfile.write(state + '\tagent\taapi\t' + verb + '\t' + str(twod_vectors[verb][0]) + '\t' + str(twod_vectors[verb][1]) + '\n')
        elif verb in twod_vectors: 
            outfile.write(state + '\tagent\tother\t' + verb + '\t' + str(twod_vectors[verb][0]) + '\t' + str(twod_vectors[verb][1]) + '\n')
    for verb in patient_verbs: 
        if verb in aapi_patient_verbs and verb in twod_vectors: 
            outfile.write(state + '\tpatient\taapi\t' + verb + '\t' + str(twod_vectors[verb][0]) + '\t' + str(twod_vectors[verb][1]) + '\n')
        elif verb in twod_vectors: 
            outfile.write(state + '\tpatient\tother\t' + verb + '\t' + str(twod_vectors[verb][0]) + '\t' + str(twod_vectors[verb][1]) + '\n')
            
def get_twod_vecs(ca_items, tx_items, vectors): 
    twod_vectors = reduce_vectors(vectors)
    outfile = open('./results/verb_embeddings.tsv', 'w')
    outfile.write('state\trole\tcategory\tverb\tdim1\tdim2\n')
    write_twod_vecs(twod_vectors, ca_items, outfile, 'CA')
    write_twod_vecs(twod_vectors, tx_items, outfile, 'TX')
    outfile.close()
    
def find_n_components(vectors, vocab): 
    '''
    Gets k based on BIC
    '''
    sorted_vocab = sorted(vocab & set(vectors.keys()))
    X = []
    for verb in sorted_vocab: 
        X.append(vectors[verb])
    X = np.array(X)
    n_comps = [5, 10, 15, 25, 50, 75, 100, 125]
    lowest_bic = np.infty
    bics = []
    for n_comp in n_comps: 
        print(n_comp)
        gmm = mixture.GaussianMixture(n_components=n_comp, random_state=42)
        gmm.fit(X)
        bic = gmm.bic(X)
        bics.append(bic)
    print(bics)
    
def cluster_glove(vectors, vocab, verb_type, k, total_verb_counts): 
    sorted_vocab = sorted(vocab & set(vectors.keys()))
    X = []
    for verb in sorted_vocab: 
        X.append(vectors[verb])
    X = np.array(X)
    gmm = mixture.GaussianMixture(n_components=k, random_state=0)
    labels = gmm.fit_predict(X)
    
    # save probabilities and vocab order
    probs = gmm.predict_proba(X)
    np.save(ROOT + 'results/' + str(k) + '_' + verb_type + '_probs.npy', probs)
    with open(ROOT + 'results/' + str(k) + '_' + verb_type + '_vocab.txt', 'w') as outfile: 
        for w in sorted_vocab: 
            outfile.write(w + '\n') 
    
    # get top N representatives of each cluster 
    reps = {}
    N = 3
    for i in range(gmm.n_components):
        rep = ''
        density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(X)
        top_n_idx = np.argpartition(density, -N)[-(N + 10):] # index of top N vectors in X
        found = 0
        for idx in top_n_idx: 
            w = sorted_vocab[idx]
            # get up to N reps that occur often
            if total_verb_counts[w] > 10: 
                rep += sorted_vocab[idx] + ', '
                found += 1
                if found == N: break
        reps[i] = rep.strip()
        
    clusters = defaultdict(list) # {label : [verbs]} 
    for i, label in enumerate(labels): 
        clusters[str(label) + ' @ ' + reps[label]].append(sorted_vocab[i])
    outfile_name = ROOT + 'results/' + str(k) + '_' + verb_type + '_verb_clusters.json'
    with open(outfile_name, 'w') as outfile: 
        json.dump(clusters, outfile)
        
def main(): 
    ca_items = get_verbs(CA_DESCRIPTORS)
    tx_items = get_verbs(TX_DESCRIPTORS)
    total_verb_counts = ca_items['verb_counts'] + tx_items['verb_counts']
    vocab = ca_items['agent_verbs'] | ca_items['patient_verbs'] | tx_items['agent_verbs'] | tx_items['patient_verbs']
    agent_verbs = ca_items['agent_verbs'] | tx_items['agent_verbs']
    patient_verbs = ca_items['patient_verbs'] | tx_items['patient_verbs']
    vectors = get_glove_vectors(vocab)
    
    cluster_glove(vectors, patient_verbs, 'patient', 100, total_verb_counts)
    cluster_glove(vectors, patient_verbs, 'patient', 50, total_verb_counts)
    cluster_glove(vectors, agent_verbs, 'agent', 100, total_verb_counts)
    cluster_glove(vectors, agent_verbs, 'agent', 50, total_verb_counts)
    cluster_glove(vectors, patient_verbs, 'patient', 75, total_verb_counts)
    cluster_glove(vectors, patient_verbs, 'patient', 25, total_verb_counts)
    cluster_glove(vectors, agent_verbs, 'agent', 75, total_verb_counts)
    cluster_glove(vectors, agent_verbs, 'agent', 25, total_verb_counts)
    
if __name__ == '__main__':
    main()
    