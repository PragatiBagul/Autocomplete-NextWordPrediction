from functools import lru_cache

from utils import build_ngrams, build_kn_stats


class KneserNeyLM:
    def __init__(self):
        self.all_grams = {}
    def append(self,tokens,max_n=5,discount=0.75):
        self.discount = discount
        self.max_n = max_n
        self.all_grams = build_ngrams(tokens,self.all_grams, max_n)
        (
            self.context_counts,
            self.unique_continuations,
            self.continuation_prob
        ) = build_kn_stats(self.all_grams)
    def lambda_weight(self, context):
        count_h = self.context_counts.get(context, 0)
        if count_h == 0:
            return 0.0
        return (
            self.discount * len(self.unique_continuations[context])
            / count_h
        )

    # @lru_cache(maxsize=100000)
    def probability(self, context, word):
        """
        Recursive Kneser–Ney probability
        """
        context = tuple(context)
        # Base case: empty context
        if len(context) == 0:
            return self.continuation_prob.get(word, 0.0)

        count_hw = self.all_grams.get(context, {}).get(word, 0)
        count_h = self.context_counts.get(context, 0)

        first_term = (
            max(count_hw - self.discount, 0) / count_h
            if count_h > 0 else 0.0
        )

        backoff = self.lambda_weight(context) * self.probability(
            context[1:], word
        )

        return first_term + backoff

    def predict_next(self, context, top_k=5):
        """
        Return top-k next-word predictions
        """
        context = tuple(context[-self.max_n:])

        scores = {}
        for word in self.continuation_prob:
            scores[word] = self.probability(context, word)

        return sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]