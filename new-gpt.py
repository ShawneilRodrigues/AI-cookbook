import os
import math
import random
from pathlib import Path

random.seed(42)

# ---------------- TOKENIZER ----------------
class ByteTokenizer:
    def encode(self, s: str):
        return list(s.encode('utf-8'))

    def decode(self, ids):
        return bytes(ids).decode('utf-8', errors='ignore')

    @property
    def vocab_size(self):
        return 256


# ---------------- DATASET ----------------
class ByteDataset:
    def __init__(self, path: str, block_size=256, split=0.9):
        data = list(Path(path).read_bytes())
        n = int(split * len(data))
        self.train = data[:n]
        self.val = data[n:]
        self.block_size = block_size

    def get_batch(self, which: str, batch_size: int):
        buf = self.train if which == 'train' else self.val
        ix = [random.randint(0, len(buf) - self.block_size - 1) for _ in range(batch_size)]
        x = [buf[i:i+self.block_size] for i in ix]
        y = [buf[i+1:i+1+self.block_size] for i in ix]
        return x, y


# ---------------- AUTOGRAD ----------------
class value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    # --- BUG FIX 1: added __radd__ and __rmul__ so that Python's built-in
    #     sum(), which starts accumulation from integer 0, works correctly.
    #     Without these, sum([value(a), value(b)]) raises TypeError because
    #     Python first tries  0 + value(a)  (int.__add__ returns NotImplemented)
    #     and then looks for value.__radd__ which didn't exist.

    def __add__(self, other):
        other = other if isinstance(other, value) else value(other)
        return value(self.data + other.data, (self, other), (1, 1))

    def __radd__(self, other):          # FIX: enables  0 + value(x)
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, value) else value(other)
        return value(self.data * other.data, (self, other), (other.data, self.data))

    def __rmul__(self, other):          # FIX: enables  scalar * value(x)
        return self.__mul__(other)

    def __pow__(self, other):
        return value(self.data ** other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return value(math.log(self.data), (self,), (1 / self.data,))

    # --- BUG FIX 2: the original exp() captured math.exp(self.data) at
    #     construction time and stored it as a static local gradient tuple.
    #     The gradient of exp(x) w.r.t. x is exp(x), which equals the
    #     *output* node's value — not a snapshot taken at creation time.
    #     Because value objects are mutated in-place during the Adam update
    #     (p.data -= ...) the stale snapshot could diverge from the true
    #     gradient in later training steps.  The fix stores the output node
    #     and returns its .data lazily via a property-based local_grads tuple.
    #
    #     Simplest correct approach: store the output node and reference its
    #     .data in a closure so the backward pass always reads the live value.

    def exp(self):
        out = value(math.exp(self.data), (self,))
        # local_grads must reference out.data (= e^x), not a frozen copy.
        # We use a one-element tuple whose single item is fetched lazily
        # by overriding _local_grads as a property would be complex with
        # __slots__; instead, we subclass the gradient lookup in backward.
        # Simplest fix: store a lambda and call it in backward.
        # But backward already zips _local_grads directly, so we change
        # backward to support callables in _local_grads.
        out._children = (self,)
        out._local_grads = (lambda: out.data,)   # FIX: lazy, reads live data
        return out

    def relu(self):
        return value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return value(other) + (-self)
    def __truediv__(self, other): return self * other**-1

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children:
                    build(c)
                topo.append(v)

        build(self)
        self.grad = 1

        for v in reversed(topo):
            for child, grad in zip(v._children, v._local_grads):
                # FIX: support callable local grads (used by exp)
                g = grad() if callable(grad) else grad
                child.grad += g * v.grad


# ---------------- MODEL SETUP ----------------
tokenizer = ByteTokenizer()
vocab_size = tokenizer.vocab_size

n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
head_dim = n_embd // n_head

matrix = lambda nout, nin, std=0.8: [[value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd)
}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd, std=0)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4*n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4*n_embd, std=0)

params = [p for mat in state_dict.values() for row in mat for p in row]


# ---------------- HELPERS ----------------
def linear(x, w):
    return [sum((wi * xi for wi, xi in zip(wo, x)), value(0.0)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps, value(0.0))   # FIX: start sum from value(0.0), not int 0
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum((xi * xi for xi in x), value(0.0)) / len(x)   # FIX: value(0.0) start
    scale = (ms + 1e-5)**-0.5
    return [xi * scale for xi in x]


# ---------------- GPT ----------------
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)

        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])

        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]

            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / (head_dim ** 0.5)
                for t in range(len(k_h))
            ]

            attn_weights = softmax(attn_logits)

            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]

            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        x_residual = x
        x = rmsnorm(x)

        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu()**2 for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])

        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits


# ---------------- TRAINING ----------------
dataset = ByteDataset("input.txt", block_size)

learning_rate = 1e-2
beta1, beta2 = 0.9, 0.95
eps = 1e-8

m = [0.0] * len(params)
v = [0.0] * len(params)

num_steps = 200

for step in range(num_steps):

    x_batch, y_batch = dataset.get_batch('train', 1)
    tokens = x_batch[0]
    targets = y_batch[0]

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(len(tokens)):
        logits = gpt(tokens[pos_id], pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[targets[pos_id]].log()
        losses.append(loss_t)

    loss = sum(losses, value(0.0)) * (1 / len(losses))   # FIX: value(0.0) start
    loss.backward()

    lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))

    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * (p.grad ** 2)

        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))

        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps)
        p.grad = 0

    print(f"step {step+1} | loss {loss.data:.4f}")


# ---------------- INFERENCE ----------------
print("\n--- inference ---")
temperature = 0.7

for _ in range(5):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = random.randint(0, vocab_size - 1)
    sample = []

    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])

        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        sample.append(token_id)

    print(tokenizer.decode(sample))