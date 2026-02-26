from collections import defaultdict

import preprocessing
from Trie import Trie
from KneserNey import KneserNeyLM
from preprocessing import load_data
import os
class NextWordPredictor:
    def __init__(self):
        self.trie = Trie()
        self.lm = KneserNeyLM()
    def load(self,path):
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for file in files:
            content = load_data(file)
            tokens = preprocessing.preprocessing(content)
            self.lm.append(tokens)
            tokens_with_counts = preprocessing.tokens_with_counts(tokens)
            for token in tokens_with_counts.keys():
                self.trie.insert(token, tokens_with_counts[token])

    def predict_next(self,context):
        return self.lm.predict_next(context)

    def predict(self,context,word):
        # word = sentence.split()[-1]
        # context = " ".join(sentence.split()[:-1])
        # print(sentence)
        candidates = self.trie.starts_with(word)
        scored = defaultdict(int)
        for candidate in candidates:
            scored[candidate] = self.lm.probability(context, candidate)

        # Rank by probability
        ranked = sorted(
            scored.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:5]