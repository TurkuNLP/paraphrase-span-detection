# filter data to keep only examples where the target is one, complete sentence to ensure oracle EM 100% for sentence-level baselines


import ufal.udpipe as udpipe
import json
import argparse
import sys
import hashlib
import os

def norm(text):
    return " ".join(text.split())

def conllu2sent(conllu_str):
    sentences = []
    lines = conllu_str.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("# text ="): # this has the sentence without tokenization
            sentences.append(line.replace("# text = ", "").strip())
    return sentences
        


def main(args):
    with open(args.file, "rt", encoding="utf-8") as json_file:
        data = json.load(json_file)
        print(len(data["data"]), "examples in the original data")

    print("Loading udpipe", file=sys.stderr)
    model = udpipe.Model.load(args.udpipe_model)
    pipeline = udpipe.Pipeline(model,"tokenize","none","none","conllu") # return conllu
    print("Done", file=sys.stderr)
    cache = {} # do not segment the same context many times, rather cache the results
    
    filtered_data = []

    for i in range(len(data['data'])):
        c = data['data'][i]['context'].replace("\n\n", "\n") # remove empty lines
        c_hash = hashlib.sha256(c.encode('utf-8')).hexdigest()
        if c_hash in cache:
            docs = cache[c_hash]
        else:
            docs = pipeline.process(c)
            cache[c_hash] = docs
        if len(data['data'][i]['answers']['text']) == 0: # negative example
            a = ''
        else:
            a = data['data'][i]['answers']['text'][0]
        
        context_sentences = [norm(e) for e in conllu2sent(docs)]
        
        if norm(a) not in context_sentences:
            continue
            
        filtered_data.append(data["data"][i])
        
        
    data["data"]=filtered_data
    print("Data after filtering:", len(data["data"]))
    
    with open(args.output, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--udpipe_model', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    
    main(args)
