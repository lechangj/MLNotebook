## RNN (Recurrent Neural Network)

- Sequence data
- Examples
  - time-stamped sales data
  - sentences
  - heartbeat data
  - audio/music data - speech recognition
  - stock price
  - car trajectories
  - airline passengers
  - weather tracking
- **Order** is important.
- In RNN, let the neuron know about its previous history of outputs by feeding its output back into istself as an input.
- Cells that are a function of inputs from previous time stpes are also known as _memory cells_.

- The sahpe of a sequence: 3-D array of size N X T X D
  - N = # samples
  - D = # features
  - T = # time steps in the sequence
  - Ex) Location Data: N - Single trip to work, D - lat, lng pair, T -  # of (lat,lng) measurements
<br>
- Variable length sequences
  - constant length sequences with zeros so that you can use numpy array for efficiency
<br>
- Forecast in Sequence data using loop (Formula)
```{python}
  x = last value of train set
  predictions = []
  for i in range (length_of_forecast):
    x_next = model.predict(x)
    predictions.append(x_next)
    x = concat(x[1:], x_next)
```
  <br><br>

### Architecture of RNN

- Sequence to Sequence (Many to Many)
- Sequence to Vector (Many to One)
- Vector to Sequence (One to Many)
  <br><br>

### The disadvantages of a basic RNN

- We only remember the previous output. Need to keep track of longer history, not just short term history.
- The vanishing gradient arises during training.
  <br><br>

### Vanishing Gradients

- The gradient is used in our calculation to adjust weights and biases in our network.
- Backpropagation goes backwards from the output to the input layer, propagating the error gradient.
- For deeper networks, as you go back to the lower layers, gradients often get smaller, eventually causing weights to never change at lower levels.
- When n hidden layers use an activation like sigmoid, softmax(use sigmoid internally), n small derivatives are multiplied together. The gradient could decrease exponentially as we propagate down to the initial layers.
  <br><br>

### How to avoid the vanishing gradient issue

- Use other activation functions such as **ReLU, Leaky ReLU, Exponential Linear Unit(ELU)**
- Perform **batch normalization**, where your model will normalize each batch using the batch mean and standard deviation.
- Choose the different initialization of weights can also help alleviate the issue (**Xavier Initialization**).
- **Gradient clipping**, where gradients are cut off before reaching a predetermined limit(e.g., cut off gradients to be between -1 and 1)
  <br><br>

### LSTM (long Short-Term Memory)

- Previous solutions could _slow down training_ because of the length of timeseries input.
- Another issue is that after a while the network will begin to forget the first inputs, as information is lost at each step going through the RNN. We need some sort of **Long-Term memory** for the networks.
- LSTM comprises of 4 gates(Forget, Input, Output and Update). **Gates** optionally let information go through.
- Glossary
  - Long Term memory ($ c \_{t} $)
  - Short Term memory ($ h \_{t} $)
  - Input ($ x \_{t} $)
  - Output ($ o \_{t} $)

$$ f _{t} = \sigma(W _{f}·[h _{t-1}, x _{t}] + b _{f}) $$
$$ i _{t} = \sigma(W _{i}·[h _{t-1}, x _{t}] + b _{i}) $$
$$ \tilde{C _{t}} = tanh(W _{C}·[h _{t-1}, x _{t}] + b _{C}) $$
$$ C _{t} = f _{t} * C _{t-1} + i _{t} * \tilde{C _{t}} $$
$$ o _{t} = \sigma(W _{o}·[h _{t-1}, x _{t}] + b _{o}) $$
$$ h _{t} = o _{t} * C _{t} $$

- LSTM Variations
	- LSTM with peepholes
	- Gated Recurrent Unit (GRU)

