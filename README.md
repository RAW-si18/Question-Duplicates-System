# Question Duplicates Detection with Siamese Networks

This project leverages Siamese networks for detecting duplicate questions in natural language processing (NLP).

## Overview

Siamese networks process two input vectors with shared weights to produce comparable outputs. This project uses triplet loss and cosine similarity to evaluate vector similarity.

### Key Components

1. **Vectorize and Embed**: Convert strings into vector embeddings.
2. **LSTM Layer**: Process embeddings through an LSTM layer.
3. **Normalize**: Normalize the vectors \(v_1\) and \(v_2\).
4. **Cosine Similarity**: Calculate cosine similarity between question pairs.

### Triplet Loss

Train the model by comparing:
- An anchor input (A)
- A positive input (P)
- A negative input (N)

Objective: Minimize the distance between A and P, maximize the distance between A and N.

\[ \mathcal{L}(A, P, N) = \max \left(\|\mathrm{f}(A)-\mathrm{f}(P)\|^{2} - \|\mathrm{f}(A)-\mathrm{f}(N)\|^{2} + \alpha, 0 \right) \]

## Implementation Instructions

### Required Libraries
- TensorFlow

### Model Implementation

Key Layers:
- **Sequential Model**: `tf.keras.models.Sequential`
- **Embedding Layer**: `tf.keras.layers.Embedding`
- **LSTM Layer**: `tf.keras.layers.LSTM`
- **Global Average Pooling**: `tf.keras.layers.GlobalAveragePooling1D`
- **Lambda Layer**: `tf.keras.layers.Lambda` (for normalization)
- **Input Layer**: `tf.keras.layers.Input`
- **Concatenate Layer**: `tf.keras.layers.Concatenate`

### Steps:
1. Create an embedding layer.
2. Add an LSTM layer.
3. Normalize the output vectors.
4. Concatenate the normalized outputs.
5. Implement triplet loss during training.

### Hard Negative Mining

Implement `TripletLoss` with hard negative mining. Positive examples are duplicates \(q1_i\) and \(q2_i\); all other pairs \(q1_i, q2_j\) (\(i \neq j\)) are negative examples.

Loss components:

\[
\begin{align*}
\mathcal{Loss_1(A,P,N)} &= \max \left( -\cos(A,P) + \text{mean}_{\text{neg}} + \alpha, 0 \right) \\
\mathcal{Loss_2(A,P,N)} &= \max \left( -\cos(A,P) + \text{closest}_{\text{neg}} + \alpha, 0 \right) \\
\mathcal{Loss(A,P,N)} &= \text{mean}(\mathcal{Loss_1} + \mathcal{Loss_2})
\end{align*}
\]

### Conclusion

Follow these instructions to implement a Siamese network with triplet loss for detecting duplicate questions using TensorFlow. This approach ensures accurate and efficient identification of similar question pairs in NLP tasks.