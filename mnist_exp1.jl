include("lib/utils.jl")
@CUDA_RUN using CUDA, KernelAbstractions, CUDA.CUDAKernels

include("lib/datasets.jl")
include("lib/factor_graph.jl")
include("lib/plotting.jl")
include("./exp_base.jl")
using HDF5

# CUDA.device!(7)

seed = 98
results = Dict()
is_torch = false # to load torch labels
dir = "./models/experiment1"
mkpath(dir) # create dir if it doesn't exist
nts = [80, 160, 320, 640, 1280, 2560, 5120, 10240, 60000]


###
### Regression MLP
###
layers = [
    (:Flatten,),
    (:Linear, 256),
    (:LeakyReLU, 0.1),
    (:Linear, 256),
    (:LeakyReLU, 0.1),
    (:Linear, 10),
    (:Regression, 0.01)
]
run_experiment(layers, "linear256"; is_regression=true, dir, seed, nts)
# test_stored_models("linear256"; is_regression=true, dir, seed, nts, is_torch)

###
### Classification MLP
###
layers = [
    (:Flatten,),
    (:Linear, 256),
    (:LeakyReLU, 0.1),
    (:Linear, 256),
    (:LeakyReLU, 0.1),
    (:Linear, 10),
    (:Argmax, true)
]
run_experiment(layers, "linear256"; is_regression=false, dir, seed, nts)
# test_stored_models("linear256"; is_regression=false, dir, seed, nts, is_torch)

###
### Classification no-aug
###
layers = [
    (:Flatten,),
    (:Linear, 256),
    (:LeakyReLU, 0.1),
    (:Linear, 256),
    (:LeakyReLU, 0.1),
    (:Linear, 10),
    (:Argmax, false)
]
run_experiment(layers, "linear256_augfree"; is_regression=false, dir, seed, nts)
# test_stored_models("linear256_augfree"; is_regression=false, dir, seed, nts, is_torch)

###
### Softmax MLP
###
layers = [
    (:Flatten,),
    (:Linear, 256),
    (:LeakyReLU, 0.1),
    (:Linear, 256),
    (:LeakyReLU, 0.1),
    (:Linear, 10),
    (:Softmax,)
]
run_experiment(layers, "linear256_sm"; is_regression=false, dir, seed, nts)
# test_stored_models("linear256_sm"; is_regression=false, dir, seed, nts, is_torch)



###
### Regression Le-Net5
###
layers = [
    (:Conv, 6, 5, 2),
    (:LeakyReLU, 0.1),
    (:MaxPool, 2),
    (:Conv, 16, 5),
    (:LeakyReLU, 0.1),
    (:MaxPool, 2),
    (:Flatten,),
    (:Linear, 120),
    (:LeakyReLU, 0.1),
    (:Linear, 84),
    (:LeakyReLU, 0.1),
    (:Linear, 10),
    (:Regression, 0.01)
]
run_experiment(layers, "lenet"; is_regression=true, dir, seed, nts)
# test_stored_models("lenet"; is_regression=true, dir, seed, nts, is_torch)

###
### Classification Le-Net5
###
layers = [
    (:Conv, 6, 5, 2),
    (:LeakyReLU, 0.1),
    (:MaxPool, 2),
    (:Conv, 16, 5),
    (:LeakyReLU, 0.1),
    (:MaxPool, 2),
    (:Flatten,),
    (:Linear, 120),
    (:LeakyReLU, 0.1),
    (:Linear, 84),
    (:LeakyReLU, 0.1),
    (:Linear, 10),
    (:Argmax, true)
]
run_experiment(layers, "lenet"; is_regression=false, dir, seed, nts)
# test_stored_models("lenet"; is_regression=false, dir, seed, nts, is_torch)

###
### Classification Le-Net5 no-aug
###
layers = [
    (:Conv, 6, 5, 2),
    (:LeakyReLU, 0.1),
    (:MaxPool, 2),
    (:Conv, 16, 5),
    (:LeakyReLU, 0.1),
    (:MaxPool, 2),
    (:Flatten,),
    (:Linear, 120),
    (:LeakyReLU, 0.1),
    (:Linear, 84),
    (:LeakyReLU, 0.1),
    (:Linear, 10),
    (:Argmax, false)
]
run_experiment(layers, "lenet_augfree"; is_regression=false, dir, seed, nts)
# test_stored_models("lenet_augfree"; is_regression=false, dir, seed, nts, is_torch)

###
### Softmax Le-Net5
###
layers = [
    (:Conv, 6, 5, 2),
    (:LeakyReLU, 0.1),
    (:MaxPool, 2),
    (:Conv, 16, 5),
    (:LeakyReLU, 0.1),
    (:MaxPool, 2),
    (:Flatten,),
    (:Linear, 120),
    (:LeakyReLU, 0.1),
    (:Linear, 84),
    (:LeakyReLU, 0.1),
    (:Linear, 10),
    (:Softmax,)
]
run_experiment(layers, "lenet_sm"; is_regression=false, dir, seed, nts)
# test_stored_models("lenet_sm"; is_regression=false, dir, seed, nts, is_torch)


show(stdout, "text/plain", last.(sort(collect(results))))

# # Now save the dictionary
# open("experiment1_$(seed).csv", "w") do file
#     # Write the header
#     write(file, "Key,Value\n")

#     # Write each key-value pair to the file
#     for (key, value) in sort(collect(results))
#         write(file, "$key,$value\n")
#     end
# end
