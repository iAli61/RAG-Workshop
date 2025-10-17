# Neural Networks

Neural Networks are computational models inspired by the human brain, consisting of interconnected nodes (neurons) organized in layers that process information through weighted connections.

## Architecture

A typical neural network consists of:

### Input Layer
Receives the raw input features and passes them to the next layer.

### Hidden Layers
Intermediate layers that perform transformations on the input data. Each neuron applies:
1. Weighted sum of inputs: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
2. Activation function: a = f(z)

Common activation functions include:
- **ReLU**: f(x) = max(0, x)
- **Sigmoid**: f(x) = 1 / (1 + e⁻ˣ)
- **Tanh**: f(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)

### Output Layer
Produces the final predictions. The structure depends on the task:
- Classification: Softmax activation for multi-class problems
- Regression: Linear activation for continuous outputs

## Training Process

Neural networks are trained using backpropagation:

1. **Forward Pass**: Input data flows through the network to produce predictions
2. **Loss Calculation**: Measure the difference between predictions and actual values
3. **Backward Pass**: Compute gradients of the loss with respect to each weight
4. **Weight Update**: Adjust weights using gradient descent or variants (Adam, SGD, RMSprop)

## Types of Neural Networks

- **Feedforward Networks**: Basic architecture with unidirectional flow
- **Convolutional Neural Networks (CNNs)**: Specialized for image data with convolutional layers
- **Recurrent Neural Networks (RNNs)**: Process sequential data with feedback connections
- **Transformers**: Use attention mechanisms for parallel processing of sequences

## Applications

Neural networks excel at:
- Image classification and object detection
- Natural language processing
- Speech recognition
- Time series forecasting
- Recommendation systems
