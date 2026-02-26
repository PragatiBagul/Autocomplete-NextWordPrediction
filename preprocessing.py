import re
import unicodedata
from collections import Counter

def load_data(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print('File not found')
def preprocessing(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    # Remove urls, emails
    content = unicodedata.normalize('NFKD', content)
    content = content.replace('\n', ' ')
    content = " ".join(content.split())
    content = content.lower()
    tokens = content.split()
    return tokens

def tokens_with_counts(tokens):
    tokens_with_counts = Counter(tokens)
    return tokens_with_counts