include("lib/utils.jl")
@CUDA_RUN using CUDA, KernelAbstractions, CUDA.CUDAKernels

include("lib/datasets.jl")
include("lib/factor_graph.jl")
include("lib/plotting.jl")
include("./exp_base.jl")
using HDF5

# CUDA.device!(7)

seed = 98 # or 1938
results = Dict()
dir = "./models/experiment2"
mkpath(dir) # create dir if it doesn't exist


for width in [400, 800, 1200]
    layers = [
        (:Flatten,),
        (:Linear, width),
        (:LeakyReLU, 0.1),
        (:Linear, width),
        (:LeakyReLU, 0.1),
        (:Linear, 10),
        (:Regression, 0.01)
    ]
    run_experiment(layers, "linear$(width)"; is_regression=true, dir, seed)
    # test_stored_models("linear$(width)"; is_regression=true, dir, seed)

    layers = [
        (:Flatten,),
        (:Linear, width),
        (:LeakyReLU, 0.1),
        (:Linear, width),
        (:LeakyReLU, 0.1),
        (:Linear, 10),
        (:Argmax, true)
    ]
    run_experiment(layers, "linear$(width)"; is_regression=false, dir, seed)
    # test_stored_models("linear$(width)"; is_regression=false, dir, seed)
end

show(stdout, "text/plain", last.(sort(collect(results))))

# Now save the dictionary
open("experiment2_$(seed).csv", "w") do file
    # Write the header
    write(file, "Key,Value\n")

    # Write each key-value pair to the file
    for (key, value) in sort(collect(results))
        write(file, "$key,$value\n")
    end
end
