This README file contains in-detail info about all the code file

# Model.py
The Notations used are -> 

B = batch size

S = source seq length, T = target seq length (both ≤ seq_len from config)

d_model = model width (e.g., 512)

h = number of attention heads (e.g., 8)

d_k = d_model / h (must divide evenly)

Tensors shown as (dim1, dim2, …)

Masks are broadcastable booleans (or 0/−inf after conversion) applied inside attention.

# nn.Module

in PyTorch nn.module is the base class of all neural network components

## Key Features of nn.Module

A.) Encapsulation of Layers & Parameters -> Any layers you define (like nn.Linear, nn.Conv2d, etc.) inside a subclass of nn.Module are automatically registered as part of the model. Trainable weights and biases are stored in nn.Parameter objects, which nn.Module tracks.

B.) Forward Method -> You implement the forward() method to define how input data flows through your network.PyTorch automatically handles the backward() pass for gradients using autograd, so you don’t need to implement that manually.

C.) Module Nesting -> You can combine multiple submodules (e.g., layers, blocks, or even other models) into one nn.Module. This makes it easy to build complex architectures.

D.) Utilities -> Provides .parameters() and .named_parameters() to access trainable parameters. Provides .train() and .eval() modes (for dropout, batchnorm, etc.). Provides .to(device) to move everything (model + parameters) to GPU/CPU easily.

# Layer Normalization

Normalization is the solution to the vanishing gradient problem, it takes the input activations and scales it up if all the activations are really small

return self.alpha * (x - mean) / (std + self.eps) + self.bias

Applies the layer-norm formula token-wise:

(x - mean) centers each token’s feature vector/ (std + eps) scales it to ~unit variance (numerically safe).

self.alpha (γ) then rescales each feature; self.bias (β) shifts it.

Broadcasting: (features,) expands to (batch, seq_len, features) automatically.

# feed forward block

This code defines a PyTorch module called FeedForwardBlock, which is a common component in transformer architectures. The block consists of two linear (fully connected) layers with a ReLU activation and dropout applied in between.

In the constructor (init), d_model specifies the input and output feature size (matching the model's hidden dimension), while d_ff is the size of the intermediate layer (usually larger, such as 4 times d_model). The dropout parameter controls the probability of randomly zeroing elements during training, which helps prevent overfitting.

The forward method describes how data flows through the block: the input x is first passed through the first linear layer (expanding its dimension to d_ff), then a ReLU activation is applied, followed by dropout for regularization. The result is then projected back to the original dimension d_model using the second linear layer. This structure allows the model to learn complex, non-linear transformations of the input while maintaining the same input and output shape, which is essential for stacking multiple layers in a transformer.

A subtle point is that dropout is only active during training; during evaluation, it has no effect. Also, the use of torch.relu ensures non-linearity, which is crucial for the model's expressiveness.

# input Embeddings

This code defines a PyTorch module called InputEmbeddings, which is responsible for converting input token indices into dense vector representations (embeddings) suitable for transformer models. The class inherits from nn.Module, making it compatible with PyTorch's model and training infrastructure.

In the init method, d_model specifies the dimensionality of the embeddings (i.e., the size of each embedding vector), and vocab_size is the number of unique tokens in the vocabulary. The nn.Embedding layer is initialized to map each token index to a d_model-dimensional vector.

The forward method takes an input tensor x of shape (batch, seq_len), where each element is a token index. It passes x through the embedding layer, resulting in a tensor of shape (batch, seq_len, d_model). The output is then scaled by math.sqrt(self.d_model). This scaling is recommended in the original Transformer paper ("Attention is All You Need") to help stabilize the variance of the embeddings, making training more effective.

A subtle point is that the scaling by sqrt(d_model) is important for model convergence, but it's easy to overlook. Also, the embedding layer will learn the best representation for each token during training.

# Positional Encoding

This code defines a PyTorch module called PositionalEncoding, which is a key component in transformer models. Since transformers do not have any inherent sense of sequence order, positional encodings are added to the input embeddings to inject information about the position of each token in the sequence.

In the init method, the class constructs a positional encoding matrix pe of shape (seq_len, d_model), where seq_len is the maximum sequence length and d_model is the embedding dimension. The encoding uses a combination of sine and cosine functions at different frequencies, as described in the original "Attention is All You Need" paper. Even indices in the embedding dimension use sine, while odd indices use cosine. This allows the model to learn relationships between positions in a way that generalizes to sequences longer than those seen during training.

The positional encoding matrix is unsqueezed to add a batch dimension and registered as a buffer, meaning it is stored with the model but not updated during training. In the forward method, the positional encodings are added to the input tensor x, and dropout is applied for regularization. The use of requires_grad_(False) ensures that the positional encodings are not updated during backpropagation.

A subtle point is that the positional encoding is sliced to match the actual sequence length of the input, allowing the model to handle variable-length sequences efficiently. This approach enables the transformer to make use of order information without relying on recurrence or convolution.

# Residual Connection

This code defines a PyTorch module called ResidualConnection, which is a standard building block in transformer architectures. The purpose of this block is to wrap another layer (the sublayer) with both layer normalization and a residual (skip) connection, along with dropout for regularization.

In the init method, the class initializes a dropout layer and a custom LayerNormalization instance, both parameterized by the number of features (hidden size). Layer normalization helps stabilize and accelerate training by normalizing the input across the feature dimension.

The forward method takes two arguments: x (the input tensor) and sublayer (a function or module representing the operation to be wrapped, such as attention or a feed-forward block). The input x is first normalized, then passed through the sublayer. Dropout is applied to the sublayer's output, and finally, the original input x is added back to the result. This addition forms the residual connection, which helps gradients flow through deep networks and mitigates the vanishing gradient problem.

A subtle point is that normalization is applied before the sublayer (pre-norm), which is a common transformer variant that can improve training stability. The use of dropout within the residual path further helps prevent overfitting.
