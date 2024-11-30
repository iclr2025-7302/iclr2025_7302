include("../lib/utils.jl")
@CUDA_RUN using CUDA, KernelAbstractions, CUDA.CUDAKernels

include("../lib/datasets.jl")
include("../lib/factor_graph.jl")
include("../lib/plotting.jl")
using HDF5

for nt in [640, 2560, 10240]
    # Parameters
    Random.seed!(98)
    batch_size = 320

    # Prepare training
    @CUDA_RUN println("Using CUDA")
    println("Num Threads: $(Threads.nthreads())")

    d = as_regression_dataset(normalize_X!(MNIST()))
    fg = create_factor_graph([
            size(d.X_train)[1:end-1],
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
        ], batch_size
    )
    @CUDA_RUN fg = adapt(CuArray, fg)

    println("Training data size: $(size(d.X_train, ndims(d.X_train)))")
    # nt = 640
    # nt = size(d.X_train, 4)
    println("Of which used: $nt")

    trainer = Trainer(fg, d.X_train[:, :, :, 1:nt], d.Y_train[:, 1:nt])
    # posterior_preds = forward_softmax(predict(fg, d.X_val))

    silent = false
    steps = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
    # steps = [1, 1, 2, 2, 3, 3, 4, 4]
    for (i, num_training_its) in enumerate(steps)
        train(trainer; num_epochs=1, num_training_its, silent)

        if (silent == false) || i == length(steps)
            evaluate(fg, d.X_val, d.Y_val; as_classification=true, silent)
        end
    end
end

###
### To evaluate a model beyond accuracy, remove the for loop above ("for nt in ...") so that the fg is available in global scope.
###
d2 = as_regression_dataset(normalize_X!(FashionMNIST()))
posterior_preds = forward_argmax(predict(fg, d.X_val))
posterior_preds2 = forward_argmax(predict(fg, d2.X_val))
@tullio labels[j] := argmax(d.Y_val[:, j]);

# Save to disk
model_name = "regression_jul1_conv_maxpool"
mkpath("./models/") # create dir if it doesn't exist
h5write("models/class_probs_julia_$model_name.h5", "data", posterior_preds)
h5write("models/class_probs2_julia_$model_name.h5", "data", posterior_preds2)
h5write("models/labels_julia_$model_name.h5", "data", labels)


###
### Compare with results from Torch
###
model_name = "regression_jul1_conv_maxpool"
posterior_preds = FloatType.(h5open("models/class_probs_julia_$model_name.h5", "r") do file
    read(file["data"])
end);
posterior_preds2 = FloatType.(h5open("models/class_probs2_julia_$model_name.h5", "r") do file
    read(file["data"])
end);
labels = (
    h5open("models/labels_julia_$model_name.h5", "r") do file
        read(file["data"])
    end
);

# Import class probabilities from pytorch
model_name = "regression_jun25_conv_maxpool"
probs_torch = FloatType.(h5open("models/class_probs_$model_name.h5", "r") do file
    read(file["data"])
end);
probs_torch2 = FloatType.(h5open("models/class_probs2_$model_name.h5", "r") do file
    read(file["data"])
end);
labels_torch = (
    h5open("models/labels_$model_name.h5", "r") do file
        read(file["data"])
    end
) .+ 1;


# plot_relative_calibration_curves(probs_torch, labels_torch, posterior_preds, labels);
# plot_ood_roc_curves(probs_torch, probs_torch2, posterior_preds, posterior_preds2);
plot_relative_calibration_curves(probs_torch, labels_torch, posterior_preds, labels; title=model_name, dpi=300);
plot_ood_roc_curves(probs_torch, probs_torch2, posterior_preds, posterior_preds2; title=model_name, dpi=300);


p = histogram(maximum(probs_torch, dims=1)[1, :], label="PyTorch / SGD", title="Distribution of max-class probabilities", bins=range(0, 1.0001, length=101), fillalpha=1.0)
histogram!(p, maximum(posterior_preds, dims=1)[1, :], label="Message Passing", title="Distribution of max-class probabilities", bins=range(0, 1.0001, length=101), fillalpha=0.7)
display(p);
