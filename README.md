# 🧠 Transformer Core Modules for Language Translation (TensorFlow)

This project implements **core components of the Transformer architecture** using TensorFlow/Keras — the same architecture used in models like **ChatGPT** and other large language models.

These reusable modules include:

- 🔄 Self-Attention
- 🔁 Cross-Attention
- ⚡ Feed-Forward Network (FFN)

These layers are essential for tasks such as **language translation**, **text generation**, and **sequence modeling**.

---

## 📁 Project Structure

```
.
├── attention_modules.py      # Contains self_attention, cross_attention, and feed_forward functions
├── README.md                 # Project documentation (this file)
└── requirements.txt          # Python dependencies
```

---

## ⚙️ Modules Overview

### 1. Self-Attention Layer

```python
def self_attention(input_shape, prefix='att', mask=False, **kwargs)
```

- Multi-head attention over a single input (query = key = value)
- Includes residual connection and layer normalization
- Used in Transformer encoders and decoders

---

### 2. Cross-Attention Layer

```python
def cross_attention(input_shape, context_shape, prefix='att', **kwargs)
```

- Allows decoder to focus on encoder output (query ≠ key/value)
- Essential for encoder-decoder models like those used in machine translation

---

### 3. Feed-Forward Network (FFN)

```python
def feed_forward(input_shape, model_dim, ff_dim, dropout=0.1, prefix='ff')
```

- Two dense layers with ReLU and dropout
- Adds non-linearity and capacity
- Includes residual connection and normalization

---

## 🧪 Example Usage

```python
import tensorflow as tf
from attention_modules import self_attention, cross_attention, feed_forward

# Self-Attention
self_att_layer = self_attention(input_shape=(64, 128), num_heads=4, key_dim=32)
self_att_layer.summary()

# Cross-Attention
cross_att_layer = cross_attention(input_shape=(64, 128), context_shape=(64, 128), num_heads=4, key_dim=32)
cross_att_layer.summary()

# Feed-Forward
ffn_layer = feed_forward(input_shape=(64, 128), model_dim=128, ff_dim=512)
ffn_layer.summary()
```

---

## 📘 Understanding the Architecture

| Component           | Role                                      | Implemented?   |
|---------------------|-------------------------------------------|----------------|
| Self-Attention      | Learn intra-sequence dependencies         | ✅ Yes        |
| Cross-Attention     | Focus on encoder outputs during decoding  | ✅ Yes        |
| Feed-Forward Layer  | Non-linear transformation                 | ✅ Yes        |
| Positional Encoding | Capture token order                       | ❌ No         |
| Token Embedding     | Convert words to vectors                  | ❌ No         |
| Output Softmax      | Generate probabilities for next token     | ❌ No         |
| Encoder/Decoder Stack | Model depth and abstraction             | ❌ No         |

> These modules form the **core of a Transformer**, and can be used to build a full translation model.

---

## 📦 Installation

1. Clone the repo:

```bash
git clone https://github.com/jaiganesh2108/transformer-modules.git
cd transformer-modules
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

> Minimal `requirements.txt`:

```
tensorflow>=2.10
```

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙌 Contributions

Contributions are welcome! Please fork this repository, make changes, and submit a pull request.

---

## 📚 References

- [Attention Is All You Need – Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)
- [TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)

---

## 💡 Future Work

- [ ] Add positional encoding
- [ ] Stack encoder-decoder blocks
- [ ] Build full Transformer model
- [ ] Train on translation datasets (e.g., English ↔ French)

---

## 🧠 Built With

- TensorFlow
- Keras
- Python 3.8+
