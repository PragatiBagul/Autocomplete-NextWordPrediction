import heapq
import math
class TrieNode:
    def __init__(self, k):
        self.children = {}
        self.is_word = False
        self.frequency = 0.0              # valid only if is_word = True
        self.top_k = []                 # min-heap of (frequency, word)
        self.k = k                      # max size of heap

class Trie:
    def __init__(self, k=5,max_weight=20.0):
        self.root = TrieNode(k)
        self.k = k
        self.max_weight = max_weight
    def _update_top_k(self, node, word, frequency):
        """
        Maintains a bounded min-heap of size <= k.
        Ensures no duplicates and highest-frequency words retained.
        """
        heap = node.top_k

        # Remove existing entry if word already exists
        for i, (freq, w) in enumerate(heap):
            if w == word:
                heap[i] = (frequency, word)
                heapq.heapify(heap)
                return

        if len(heap) < node.k:
            heapq.heappush(heap, (frequency, word))
        else:
            if frequency > heap[0][0]:
                heapq.heappushpop(heap, (frequency, word))

    def insert(self, word, frequency):
        node = self.root
        path = []
        path.append(self.root)
        #Traverse or create path
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode(self.k)
            node = node.children[char]
            path.append(node)
        node.is_word = True

        # Log scaled incremental update
        delta = math.log1p(frequency)
        node.frequency = min(node.frequency + delta,self.max_weight)

        updated_frequency = node.frequency
        for node in path:
            self._update_top_k(node, word, updated_frequency)
    def search(self, word):
        """
        Returns True if exact word exists.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]

        return node.is_word

    def starts_with(self, prefix):
        """
        Returns top-K suggestions sorted by descending frequency.
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # Return highest frequency first
        return sorted(node.top_k, key=lambda x: -x[0])