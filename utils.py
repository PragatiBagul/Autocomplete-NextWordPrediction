# ============================================================
# Kneser–Ney Language Model (up to 5-grams)
# ============================================================

from collections import defaultdict
from functools import lru_cache

DISCOUNT = 0.75
MAX_N = 5


# ------------------------------------------------------------
# 1. Build n-gram counts (your original logic, cleaned)
# ------------------------------------------------------------

def build_ngrams(tokens,all_grams = {}, max_n=5):
    """
    all_grams[context_tuple][next_word] = count
    """
    for i in range(1, len(tokens)):
        for p in range(1, max_n + 1):
            if i >= p:
                context = tuple(tokens[i - p:i])
                word = tokens[i]

                if context not in all_grams:
                    all_grams[context] = {}

                all_grams[context][word] = all_grams[context].get(word, 0) + 1

    return all_grams


# ------------------------------------------------------------
# 2. Build Kneser–Ney statistics
# ------------------------------------------------------------

def build_kn_stats(all_grams):
    context_counts = defaultdict(int)
    unique_continuations = defaultdict(set)
    preceding_contexts = defaultdict(set)

    for context, next_words in all_grams.items():
        for word, count in next_words.items():
            context_counts[context] += count
            unique_continuations[context].add(word)
            preceding_contexts[word].add(context)

    total_unique_ngrams = sum(len(v) for v in all_grams.values())

    continuation_prob = {
        word: len(contexts) / total_unique_ngrams
        for word, contexts in preceding_contexts.items()
    }

    return context_counts, unique_continuations, continuation_prob