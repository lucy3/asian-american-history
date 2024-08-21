import csv

input_file = './wordlists/people_terms.csv'

terms = set()
with open(input_file, 'r') as infile: 
    reader = csv.reader(infile)
    for row in reader: 
        terms.add(row[0])
        
print(', '.join(sorted(terms)))