import os
from random import shuffle

import numpy as np
import pandas as pd
import re
import unicodedata
import random

from preprocessing import load_data

MIN_LENGTH = 2
MAX_LENGTH = 20

def clean_sentence(sentence):
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = unicodedata.normalize('NFKD', sentence)
    sentence = sentence.lower()
    return sentence

import random

def get_random_substring(sentence):
    words = sentence.split()
    total_words = len(words)

    if total_words <= MIN_LENGTH:
        return None, None

    upper = min(MAX_LENGTH, total_words - 1)

    if MIN_LENGTH > upper:
        return None, None

    length = random.randint(MIN_LENGTH, upper)

    context = " ".join(words[:length])
    target = words[length]

    return context, target
def generate_dataset(file_content, data):
    for line in file_content.split('\n'):
        sentence = clean_sentence(line)
        context, target = get_random_substring(sentence)

        if context is not None:
            data.append((context, target))

path = './data'
files = [
    os.path.join(path, f)
    for f in os.listdir(path)
    if os.path.isfile(os.path.join(path, f))
]

data = []

for file in files:
    content = load_data(file)
    generate_dataset(content, data)

# Build DataFrame ONCE
df_train = pd.DataFrame(data, columns=['context', 'target']).sample(1000)
df_test = pd.DataFrame(data, columns=['context', 'target']).sample(200)
df_train.to_csv('./dataset/train.csv', index=False)
df_test.to_csv('./dataset/test.csv', index=False)