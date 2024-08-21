'''
stitch together broken sentences by checking for '.' or '!' or '?' and lowercase beginning of next line
previous line should not be a section title
'''
import os

IN_FOLDER = './data/source_txt_ca/'
OUT_FOLDER = './data/clean_ca/'

def is_title(line): 
    '''
    Title detection 
    '''
    # some titles are all caps
    if line == line.upper(): 
        return True
    # some titles have uppercase first letter on most longer words
    tokens = line.split()
    if len(tokens) > 10: # long line, likely not title
        return False
    is_title = True
    for tok in tokens: 
        # this skips over words such as "the", "a", "of"
        if len(tok) > 3 and tok != tok.title(): 
            is_title = False
    return is_title

def sanity_check(): 
    '''
    Check that no words were dropped in stitching process by
    comparing the start of source and clean texts. 
    '''
    for f in os.listdir(IN_FOLDER): 
        if f.startswith('.'): continue
        print(f)
        lines = ''
        with open(OUT_FOLDER + f, 'r') as infile: 
            i = 0
            for line in infile: 
                lines += ' ' + line
                i += 1
                if i > 1000: 
                    break
        lines = lines.strip()
        old_lines = ''
        with open(IN_FOLDER + f, 'r') as infile: 
            i = 0
            for line in infile: 
                old_lines += ' ' + line
                i += 1
                if i > 1000: 
                    break
        old_lines = old_lines.strip()

        old_toks = lines.split()[:1000]
        new_toks = old_lines.split()[:1000]

        for i, tok in enumerate(old_toks): 
            if old_toks[i] != new_toks[i]: 
                print(i, old_toks[i-10:i+10], new_toks[i-10:i+10])

        assert ' '.join(lines.split()[:1000]) == ' '.join(old_lines.split()[:1000])

def clean_texts(): 
    punct_set = set(['.', '!', '?'])
    
    for f in os.listdir(IN_FOLDER): 
        if f.startswith('.'): continue
        print(f)
        outpath = OUT_FOLDER + f
        with open(outpath, 'w') as outfile: 
            with open(IN_FOLDER + f, 'r') as infile: 
                curr_output = ''
                for line in infile: 
                    line = line.strip()
                    if line == '': 
                        continue
                    # check if current line is stop 
                    if is_title(line): 
                        # write out current output and line
                        if curr_output != '': 
                            outfile.write(curr_output + '\n') 
                        outfile.write(line + '\n') 
                        curr_output = ''
                    elif curr_output != '' and curr_output[-1] in punct_set: 
                        outfile.write(curr_output + '\n') 
                        curr_output = line
                    elif curr_output == '': 
                        curr_output = line
                    elif line[0] == line[0].lower(): 
                        # attach line to output
                        curr_output += ' ' + line
                    else: 
                        # ambiguous if should connect lines, so don't
                        if curr_output != '': 
                            outfile.write(curr_output + '\n') 
                        curr_output = line
        
def main(): 
    clean_texts()
    sanity_check()

if __name__ == '__main__':
    main()