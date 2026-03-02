import os
import math
import random
random.seed(42)

if not os.path.exists('input.txt'):
    import urllib.request
    name_url='https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(name_url,'input.txt')
docs=[l.strip() for l in open('input.txt').read().strip().split("\n") if l.strip()]
random.shuffle(docs)
print(f'num docs:{len(docs)}')

uchars=sorted(set(''.join(docs)))
BOS=len(uchars)
vocab_size=len(uchars)+1
print(f'vocab size :{vocab_size}')

