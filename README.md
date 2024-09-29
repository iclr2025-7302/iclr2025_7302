# Scalable Approximate Belief Propagation for Bayesian Neural Networks

### Setup
It is useful to create a new user-wide Julia env that is not constrained to one repository or directory. Julia offers this feature as "shared env", which can be used when activating julia in the command line. VSCode also has some button (at the bottom) that allows to set the julia env.


You can create a new Julia env called "myenv" like this:
``` bash
julia --project=@myenv --threads auto
```

Next, install all dependencies:
``` Julia
import Pkg; Pkg.add(["Adapt", "BenchmarkTools", "CalibrationErrors", "Distributions", "GraphRecipes", "Graphs", "HDF5", "Integrals", "InvertedIndices", "IrrationalConstants", "KernelAbstractions", "MLDatasets", "NNlib", "Plots", "Polyester", "ProgressBars", "QuadGK", "SpecialFunctions", "StatsBase", "Tullio"])
```

If the machine has a CUDA GPU, then also install CUDA additionally. All code should work with or without CUDA, but training large networks is obviously faster with CUDA.
``` Julia
import Pkg; Pkg.add("CUDA")
```


### Getting started

It is probably a good idea to start by running the training code in `general_training_scripts/mnist.jl`. After first getting a feeling for how the different high-level APIs come together, it will probably be easier to explore the implementation subsequently.

Nevertheless, here is a short overview of the most important files in the `lib`:

* **factor_graph.jl**: Implementation of the FactorGraph (neural net) and its layers as well as higher-level functions for full-network operations. Also contains a Trainer object that stores required information during training.

* **message_equations.jl**: Implementations of message equations for factors such as LeakyReLU, Convolutions, or Softmax.

* **messages_gaussian_mult.jl**: A multiplication library for operations "A * B + C" where A can be either Gaussian or Float and where the operands can be scalars, vectors, matrices, or tensors. This library generalizes the sum and product factors.

* **gaussian.jl**: Gaussian1d type with lots of operations around it. There is also a barely-used multivariate GaussianDist.

All of our experimental code is available in the different files on top-level.


### Benchmarking against PyTorch
There is a file `general_training_scripts/mnist.py` that contains a script for training on MNIST using PyTorch. If I want to compare some architecture, these will typically be my steps:
1. Define some architecture and train the FactorGraph for a while
2. Setup PyTorch with the identical architecture and choose a name for saving the model outputs.
3. Load PyTorch outputs into the Julia script (see mnist files below the training loop).
4. Plot out-of-distribution recognition and calibration of the two models (MP vs. PyTorch).


### About Factor Graphs
We initially implemented a full factor graph with all its concepts: stateful variables, factors, and message equations. However, this design is inefficient and leads to unintuitive neural network code.

Our FactorGraph represents a whole neural network layer by layer. Each layer object can be thought of as a subgraph that connects input variables (inputs / activations) with output variables (pre-activations / outputs). The messages to the inputs or outputs of a layer are then computed with stateless message equations.

We also constrain the flow of messages to a coordinated "forward pass" or "backward pass" throughout the network's factor graph. We therefore don't have to store messages and can directly pass the outgoing message of one layer to the next one. If a layer needs to store any additional information, it stores it internally.

Another big change is that now there is only one layer object (per layer) and it gets reused for different training examples. Each layer stores the messages to the weights for some number of training examples. After iterating on that batch for a while, a combined message of all current training examples to the weights is stored in the Trainer object and the layer resets its internal messages.
