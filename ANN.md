## Perceptron Model

$$ \hat{y} = \sum \limits _{i=1} ^{n} x _{i} w _{i} + b _{i} $$

The bias $ b _{i} $ is the threshold value.
Even $ x _{i} = 0 $, it can affect the result $ \hat{y} $.

The weight $ w \_{i} $ can be changed. <br><br>

### Activation Functions $f(z)$

where, $z = {x w + b} $

1. Sigmoid($\sigma$): $-1 < output < 0$
   $$ \sigma = {1 \over 1 + e ^{(-z)}} $$
2. Hyperbolic tangent($tanh$): $-1 < output < 1$
   $$ cosh x = {e ^{x} + e ^{-x} \over 2}$$
   $$ sinh x = {e ^{x} - e ^{-x} \over 2}$$
   $$ tanh x = {sinhx \over coshx}$$
3. Rectified Linear Unit($ReLU$): $0 < output < \infty$, dealing with the vanishing gradient issue
4. Softmax: For multi class classification<br><br>

### Multiclass classification

#### Non-exclusive classes

- Photos can have multiple tags.
- Sigmoid function for output layer activation function

#### Mutually exclusive classes

- A photo can be grayscale or full color.
- Softmax function
  $$ f(z) _{i} = {e ^{z _{i}} \over \sum \limits _{j=1} ^{k} e ^{z _{j}}} $$
for $ i = 1, ..., k$

- The sum of all probabilities will be equal to one.
- The model returns the probabilities of each class and the target class chosen will have the highest probability.<br><br>

### Cost Functions and Gradient Descent

1. Quadratic cost function<br>
   - Simply calculate the differnce between the real value against our predicted values.
   - The larger step size can overshooting the minimum point.
   - Learning rate or step size can be different in adaptive gradient descent => Adam (2015)<br>
2. Cross entropy cost function for classification problems
   - The assumption is that your model predicts a probability distribution for each classs $ i = 1, 2, ..., c$
   - For a binary classification
     $$ -(ylog(p) + (1-y)log(1-p))$$
   - For M number of classes > 2
     $$ - \sum \limits _{c=1} ^{M} {y _{0,c}\log(p _{0,c})} $$
     <br><br>

### Backpropagation

- Fundamentally we want to know how the cost function results change with respect to the weights in the network, so we can update the weights to minimize the cost function.<br>

- The main idea of the backpropagation is that we can use the gradient to go back through the network and adjust our weights and biases to minimize the output of the error vector on the last output layer.<br><br>
