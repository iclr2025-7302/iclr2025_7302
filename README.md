# Scalable Approximate Belief Propagation for Bayesian Neural Networks

### Setup
Our library and experiments are mostly written in Julia.
You need version 1.11.0 or higher.
For information on how install it, see [here](https://julialang.org/downloads/).
Or just run
``` bash
curl -fsSL https://install.julialang.org | sh
```

It is useful to create a new user-wide Julia env that is not constrained to one repository or directory. Julia offers this feature as "shared env", which can be used when activating julia in the command line. VSCode also has some button (at the bottom) that allows to set the julia env.


You can create a new Julia env called "bnn_mp" like this:
``` bash
julia --project=@bnn_mp --threads auto
```

Next, install all dependencies:
``` Julia
import Pkg; Pkg.add(["Adapt", "BenchmarkTools", "CalibrationErrors", "Distributions", "GraphRecipes", "Graphs", "HDF5", "Integrals", "InvertedIndices", "IrrationalConstants", "KernelAbstractions", "MLDatasets", "NNlib", "Plots", "Polyester", "ProgressBars", "QuadGK", "SpecialFunctions", "StatsBase", "Tullio"])
```

If the machine has a CUDA GPU, then also install CUDA additionally. All code should work with or without CUDA, but training large networks is obviously faster with CUDA.
``` Julia
import Pkg; Pkg.add("CUDA")
```

For running the experiments you also need to createthe conda environment specified in `conda_environment.yml`:
``` bash
conda env create -f conda_environment.yaml
```

### Running the experiments

All of our experimental code is available in the different files on top-level.
You can run all experiments at once by running 
``` bash
./run_all_experiments.sh
```
The **CIFAR-10** experiments can be run individually via
``` bash
conda init;
conda deactivate;
conda activate mp-bnns-experiments;
julia --project=@bnn_mp cifar10_exp.jl &&
python cifar10_exp.py --optimizer=AdamW &&
python cifar10_exp.py --optimzier=IVON &&
julia --project=@bnn_mp cifar10_eval.jl
```
The **synthetic** data demonstration can be run individually via
``` bash
```

If you want to run some experiments individually, have a look at the `run_all_experiments.sh` script.
It should be self-explanatory.

### Library Overview

It is probably a good idea to start by running the training code in `example_training_scripts/mnist_regression.jl`. 
After first getting a feeling for how the different high-level APIs come together, it will probably be easier to explore the implementation subsequently.

Nevertheless, here is a short overview of the most important files in the `lib`:

* **factor_graph.jl**: Implementation of the FactorGraph (neural net) and its layers as well as higher-level functions for full-network operations. Also contains a Trainer object that stores required information during training.

* **message_equations.jl**: Implementations of message equations for factors such as LeakyReLU, Convolutions, or Softmax.

* **messages_gaussian_mult.jl**: A multiplication library for operations "A * B + C" where A can be either Gaussian or Float and where the operands can be scalars, vectors, matrices, or tensors. This library generalizes the sum and product factors.

* **gaussian.jl**: Gaussian1d type with lots of operations around it. There is also a barely-used multivariate GaussianDist.

### Library Architecture Design Decisions
We initially implemented a full factor graph with all its concepts: stateful variables, factors, and message equations. However, this design is inefficient and leads to unintuitive neural network code.

Our FactorGraph represents a whole neural network layer by layer. Each layer object can be thought of as a subgraph that connects input variables (inputs / activations) with output variables (pre-activations / outputs). The messages to the inputs or outputs of a layer are then computed with stateless message equations.

We also constrain the flow of messages to a coordinated "forward pass" or "backward pass" throughout the network's factor graph. We therefore don't have to store messages and can directly pass the outgoing message of one layer to the next one. If a layer needs to store any additional information, it stores it internally.

Another big change is that now there is only one layer object (per layer) and it gets reused for different training examples. Each layer stores the messages to the weights for some number of training examples. After iterating on that batch for a while, a combined message of all current training examples to the weights is stored in the Trainer object and the layer resets its internal messages.
