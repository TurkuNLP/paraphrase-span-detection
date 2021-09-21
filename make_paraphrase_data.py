import json
import argparse
import gzip
import sys

    
def read_data(args):
    with open(args.file, 'rt', encoding="utf-8") as f:
        data = json.load(f)
    with gzip.open(args.context, 'rt', encoding="utf-8") as f:
        contexts = json.load(f)
    return data, contexts

def refine_data(data, contexts):
    """
    This function applys the setting of queston answering data creation for paraphrase detection. The data can be used for training a BERT with 
    run_qa.py
    """
    qa_data = []
    context_notavailable = 0

    for i, d in enumerate(data):

        """
        {
            "context": {
                "beg1": 6255,
                "beg2": 6943,
                "doc1": "75a7a1acd57a8949f140d89ee56d3bcbf66e22ea",
                "doc2": "b16b2855a37035f52240fa8bd7a67e621e9c3fc8",
                "end1": 6332,
                "end2": 7040
            },
            "fold": 80,
            "goeswith": "episode-30966",
            "label": "4<",
            "rewrites": [
                [
                    "Viisivuotias tyttärenikin pystyisi tehdä tuon, eikä hän ole mikään ruudinkeksijä.",
                    "Minun 5-vuotias tyttäreni osaisi tehdä tuota ja hän ei ole suinkaan kirkkain lamppu solariumissa."
                ]
            ],
            "txt1": "Viisivuotias tyttärenikin pystyisi samaan, eikä hän ole mikään ruudinkeksijä.",
            "txt2": "Minun 5-vuotias tyttäreni osaisi tehdä tuota ja hän ei ole suinkaan kirkkain lamppu solariumissa."
        },
        """
        if d["context"] == None: # context is not available
            context_notavailable += 1
            continue
            
        label = d['label']
    
        # para1 question, para2 answer
        question = d["txt1"] # cannot be empty!
        answers = {}
        context_doc_id = d["context"]["doc2"]
        context_doc_text = contexts[context_doc_id]
        if label == "1" or label == "2": # negative example
            answers['answer_start'] = []
            answers['text'] = []
        else:
            answers['answer_start'] = [d['context']['beg2']]
            answers['text'] = [context_doc_text[d['context']['beg2']:d['context']['end2']]]
            #print("DOC:", answers["text"][0], file=sys.stderr)
            #print("ORG:", d["txt2"], "\n", file=sys.stderr)
        example = {'id': str(len(qa_data)+1), 'question': question, 'answers': answers, 'context':context_doc_text, 'label':label}
        qa_data.append(example)
    
        # para2 question, para1 answer
        question = d["txt2"] # cannot be empty!
        answers = {}
        context_doc_id = d["context"]["doc1"]
        context_doc_text = contexts[context_doc_id]
        if label == "1" or label == "2": # negative example
            answers['answer_start'] = []
            answers['text'] = []
        else:
            answers['answer_start'] = [d['context']['beg1']]
            answers['text'] = [context_doc_text[d['context']['beg1']:d['context']['end1']]]
            #print("DOC:", answers["text"][0], file=sys.stderr)
            #print("ORG:", d["txt1"], "\n", file=sys.stderr)
        example = {'id': str(len(qa_data)+1), 'question': question, 'answers': answers, 'context':context_doc_text, 'label':label}
        qa_data.append(example)
    
    print("Context not available:", context_notavailable, file=sys.stderr)
    print("Paraphrases used:", int(len(qa_data)/2), file=sys.stderr)
    return qa_data

  
def main(args):

    data, contexts = read_data(args)

    qa_data = refine_data(data, contexts)
    
    dataset = {}
    dataset["version"] = "0.0.0"
    dataset["data"] = qa_data
    
    print("Number of QA examples:", len(qa_data), file=sys.stderr)
    
    with open(args.output, "wt", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, required=True)
    parser.add_argument('--context', '-c', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    
    args = parser.parse_args()
    
    main(args)
    
