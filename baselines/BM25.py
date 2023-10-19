from numpy import log as ln
from collections import Counter

class BM25(object):
    # implementation follows this: https://en.wikipedia.org/wiki/Okapi_BM25

    def __init__(self, k=1.2, b=0.75):
        
        self.k = k
        self.b = b
        self.N = None          # number of documents in the collection
        self.avgdl = None # average document length in the collection (in words)
        self.df = None           # document frequencies for each term
        self.idf = None
        
    def train(self, documents):
        # "train" the model using given documents splitted into words
        # training means calculate N, avgdl, df, idf
        doc_lengths = []
        df = {}
        for doc in documents:
            tokens = doc.lower().split()
            doc_lengths.append(len(tokens))
            for token in list(set(tokens)): # unique tokens
                if token not in df:
                    df[token] = 0
                df[token] += 1
        self.N = len(doc_lengths)
        self.avgdl = sum(doc_lengths)/len(doc_lengths)
        self.df = df
        
        # idf
        self.idf = {}
        for term in self.df:
            self.idf[term] = self.calculate_idf(term)

    def calculate_idf(self, term):
        assert term in self.df
        idf = ln(((self.N - self.df[term] + 0.5)/(self.df[term] + 0.5)) + 1)
        return idf
        
    def embed(self, text):
        # a wrapper function to be similar with other classes
        if isinstance(text, str):
            return text
        if isinstance(text, list) and len(text)==1:
            return text[0]
        assert False

    def score(self, q, doc):
        q_tokens = q.lower().split()
        doc_tokens = doc.lower().split()
        doc_tf = Counter(doc_tokens)
        score = 0.0
        for term in q_tokens:
            if term not in self.idf:
                continue
            term_idf = self.idf[term]
            tf = doc_tf[term] # returns zero if term does not occur in this document
            t_score = term_idf * ((tf * (self.k + 1)) / (tf + self.k * (1 - self.b + self.b*(len(doc_tokens)/self.avgdl))))
            score += t_score
        return float(score)
        
        
        
        
if __name__=="__main__":

    documents = ["a b c", "a b b", "a a"]
    print(documents)
    bm = BM25()
    bm.train(documents)
    print(bm.N, bm.avgdl, bm.df, bm.idf)
    
    queries=["a", "a b", "c", "d"]
    
    for q in queries:
        print("Q:", q)
        for d in documents:
            s = bm.score(q, d)
            print("D:", d, s)
        print()
        
        
        
    # cross-checking the returned values against this: https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables
    
#    documents = ["Shane", "Shane C", "Shane P Connelly", "Shane Connelly", "Shane Shane Connelly Connelly", "Shane Shane Shane Connelly Connelly Connelly"]
#    bm = BM25(b=1.0, k=5.0)
#    bm.train(documents)

#    queries = ["Shane"]
#    for q in queries:
#        print("Q:", q)
#        for d in documents:
#            s = bm.score(q, d)
#            print("D:", d, s)
#        print()


