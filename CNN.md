## CNN

- Computer vision is a general term of using computer programs to process image data.
- Filters are essentially an _image kernel_, which is a small matrix applied to an entire image.
- Filters allow us to transform images.

  - 3 x 3 filters (Ex. blur filters)
  - Multiply by filter weights
  - Sum the result

- We can also edit our stride distnace.

- In the context of CNNs, these filters are referred to as **convolution kernels**.
- The process of passing them over an image is known as **convolution**.
- During convolution, we would _loose borders_. We can **pad** the image with more values. This allows us to preserve the image size.
- Convolution as **pattern finding** (_Convolution_ vs _Cross-Correlation_ or _Pearson-Correlation_)
  $$ a \cdot b = \sum \limits _{i=1} ^{N} a _{i} b _{i} = |a||b| cos \theta _{ab} $$
  - The dot product can be thought of as a correlation measure.
  - If high positive correlation -> dot product is large and positive.
  - If high negative correlation -> dot product is large and negative.
  - If orthogonal (no correlation) -> dot product is zero.
  - Convolution _filters out_ everything not related to the pattern contained in the filter.<br><br>
- Convolution as **shared-parameter matrix mulplication / feature transformer**

  - _Parameter Sharing_ / _Weight Sharing_
  - _Translational invariance_ vs fully connected (dense) neural network => **Shared pattern finder**

  <br><br>

### Convolution on Color images

#### 2-D (2-D dot product) - a grayscale pattern-finder

$$ (A * w) _{ij} = \sum \limits _{i'=1} ^{K} \sum \limits \_{j'=1} ^{K} A(i + i', j + j')w(i',j') $$

#### 3-D (3-D dot product) - a color pattern-finder

$$ (A * w) _{ij} = \sum \limits _{c=1} ^{3} \sum \limits _{i'=1} ^{K} \sum \limits _{j'=1} ^{K} A(i + i', j + j', c)w(i',j',c) $$

#### 3-D convolutions with the same uniformity including the features

$$ B = A * w $$
$$ Shape(A) = H x W x C _{1} $$
$$ Shape(w) = C _{1} x H x W x C _{2} $$
$$ Shape(B) = H x W x C _{2} $$
$$ B(i,j,c) = \sum \limits _{i'=1} ^{K} \sum \limits _{j'=1} ^{K} \sum \limits _{c'=1} ^{C _{1}} A(i + i', j + j', c')w(c',i',j',c) $$

- We realized we previously only defined convolution for 2-D (grayscale) images and corresponding 2-D filters.
- We extended this to color images by saying the filter should just have the same depth, and then we dot along all 3 axes.
- This breaks uniformity, because input is 3-D but output is still 2-D
- So we wouldn't be able to stack multiple convolutions sequentially.
- We extended this further by noting that each layer should find multiple features (using multiple filters)
- This results in multiple 2-D outputs (one per filter) which we can stack to get out a 3-D image once again.
- Input to the neural network is a true color image: H x W x 3
- But after subsequent convolutions, we'll just have H x W x (arbitrary #)
- We call these "feature maps" (e.g. each 2-D image is a map that tell us where the feature is found)

<br><br>

### Problems with ANNs for MNIST data

1. Large amount of parameters: over 100,000 for tiny 28 x 28 images.
2. We lose all 2D information by flattening out the image.
3. Will only work on very similar, well centered images.

### A CNN can use convolution layers to help alleviate these issues.

1. A convolutional layer is created when we apply multiple image filters to the input images. The layer wil then be trained to figure out the best filter weight values.
2. A CNN also helps reduce parameters by focusing on local connectivity.

   - Not all neurons will be fully connected.
   - Instead, neurons are only connected to a subset of local neurons in the next layer (these end up being the filters).
   - The number of filters works as the number of features.
   - For color, 3 channel and each each color channel will have intensity values.
   - The shape of the color array then has 3 dimensions, i.e. height, width, color channels.

3. Often convolutioned layers are fed into another convolutional layer. This allows the networks to discover patterns within patterns, usually with more complexity for later convolutional layers.<br><br>

### Pooling layers

- Even with local connecctivity, when dealing with color images and possibly 10s or 100s of filters we will have a large amount of parameters.
- We can use pooling layers to reduce this.
- Pooling layers accept convolutional layers as input.
- We can reduce the size with subsampling. Ex) Max Pooling, Average Pooling
- This pooling layer will end up removing a lot of information, even a small pooling kernel of 2 x 2 with a stride of 2 will remove 75% of the input data.
- Translational invariance: I don't care _where_ in the image the feature occurred, I just care that it did.
  <br><br>

### Dropout

- Dropout can be thought of as a form of regularization to help prevent overfitting.
- During training, units are randomly dropped, along with their connections.
- This helps prevent units from co-adapting too much
  <br><br>

### Architectures

- LeNet-5 (Named after Yann LeCun)
- AlexNet
- Google LeNet
- ResNet

#### Stage1: A series of convolution layer and pooling layers combinations - "Feature Transformer"
- Why convolution followed by pooling?
	- If the filter size stays the same, but the image shrinks, then the portion of the image that the filter covers increases!
	- The input images shrinks.  Since filters stay the same size, thy find increasingly large patterns (relative to the image) => CNNs learn _hierarchical_ features.
	- We loose spatial information -> We don't care where the feature was found.
	- We gain information in terms of what features were found.
#### Stage2: A series of dense layers (Fully connected layers) - "Nonlinear Classifier"
- Dense layer expects a 1-D input vector
- We need to use _Flatten()_
- Global Max Pooling as an Flatten alternative -> handling different-sized images
#### With CNNs, the conventions are pretty standard.

- Small filters relative to image, e.g. 3x3, 5x5, 7x7
- Repeat: convolution -> pooling -> convolution -> pooling ...
- Increate the number of feature maps, e.g. 32 -> 64 -> 128 ...


  <br><br>

### One Hot Encoding

- One Hot Encoding is a process of converting categorical data variables so they can be provided to machine learning algorithms to improve predictions.
- One hot encoding is a crucial part of feature engineering for machine learning.
- Categorical data refers to variables that are made up of label values, for example, a **color** variable could have the values **red**, **blue** and **green**.
- Some machine learning algorithms can work directly with categorical data depending on implementation, such as a decision tree, but most require any inputs or outputs variables to be a number, or number in value. This means that any categorical data must be mapped to integers.

- One hot encoding is one method of converting data to prepare it for an algorithm and get a better prediction. With one hot, we convert each categorical value into a new categorical column and assig a binary value of _1_ or _0_ to these columns. Each integer value is represented as a binary vector. All the values are zero, and the index is marked with a 1.

- **Pandas**: **get_dummies()**

- **Keras** provides numpy utility library, which provides functions to perform actions on numpy arrays. Using the method **to_categorical()**, a numpy array (or) a vector which has integers that represent different categories, can be converted into a numpy array (or) a matrix which has binary values and has columns equal to the number of categories in the data.

  <br><br>



