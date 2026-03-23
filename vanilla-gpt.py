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

class value:
    __slots__=('data','grad','_children','_local_grads')

    def __init__(self,data,children=(),local_grads=()):
        self.data=data
        self.grad=0
        self._children=children
        self._local_grads=local_grads
    
    def __add__(self,other):
        other=other if isinstance(other,value) else value(other)
        return value(self.data+other.data,(self,other),(1,1))
    def __mul__(self,other):
        other=other if isinstance(other,value) else value(other)
        return value(self.data*other.data,(self,other),(other.data,self.data))
    
    def __pow__(self,other): return value(self.data**other,(self,),(other*self.data**(other-1),))
    def log(self): return value(math.log(self.data),(self,),(1/self.data,))
    def exp(self): return value(math.exp(self.data),(self,),(math.exp(self.data),))
    def relu(self): return value(max(0,self.data),(self,),(float(self.data>0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad=1
        for v in reversed(topo):
            for child, local_grads in zip(v._children,v._local_grads):
                child.grad+=local_grads*v.grad
    
'''
x = value(2.0)
y = value(3.0)

a = x * y
b = x + y
c = a * b
d = c.relu()
e = d + x
f = e * y

f.backward()
print(x.grad)
print(y.grad)
'''
n_embd=16 # latent space size(embedding)
n_head=4
n_layer=1
block_size=16
head_dim=n_embd//n_head
matrix= lambda nout,nin,std=0.8:[[value(random.gauss(0,std)) for _ in range(nin)]for _ in range(nout)]
state_dict={'wte':matrix(vocab_size,n_embd),'wpe': matrix(block_size,n_embd),'lm_head':matrix(vocab_size,n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq']=matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk']=matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv']=matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo']=matrix(n_embd, n_embd,std=0)
    state_dict[f'layer{i}.mlp_fc1']=matrix(4*n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2']=matrix(n_embd, 4*n_embd,std=0)
params=[p for mat in state_dict.values() for row in mat for p in row]
print(f'number of parameter are : {len(params)}')
    
    