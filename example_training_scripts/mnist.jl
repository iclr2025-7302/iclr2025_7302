include("../lib/utils.jl")
@CUDA_RUN using CUDA, KernelAbstractions, CUDA.CUDAKernels

include("../lib/datasets.jl")
include("../lib/factor_graph.jl")
include("../lib/plotting.jl")
using HDF5


# for nt in [80, 160, 320, 640, 1280, 2560, 5120]
# Parameters
Random.seed!(98)
batch_size = 320

# Prepare training
@CUDA_RUN println("Using CUDA")
println("Num Threads: $(Threads.nthreads())")

d = normalize_X!(MNIST())
fg = create_factor_graph([
        size(d.X_train)[1:end-1],
        (:Flatten,),
        (:Linear, 200),
        (:LeakyReLU, 0.1),
        (:Linear, 10),
        (:Argmax, true)
    ], batch_size
)
@CUDA_RUN fg = adapt(CuArray, fg)

println("Training data size: $(size(d.X_train, ndims(d.X_train)))")
nt = 640
# nt = size(d.X_train, 4)
println("Of which used: $nt")

trainer = Trainer(fg, d.X_train[:, :, :, 1:nt], d.Y_train[1:nt])
# posterior_preds = predict(fg, d.X_val)

silent = false
steps = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
# steps = [1, 1, 2, 2, 3, 3, 4, 4]
for (i, num_training_its) in enumerate(steps)
    train(trainer; num_epochs=1, num_training_its, silent)

    if (silent == false) || i == length(steps)
        evaluate(fg, d.X_val, d.Y_val; silent)
    end
end
# end

d2 = normalize_X!(FashionMNIST())
posterior_preds = predict(fg, d.X_val)
posterior_preds2 = predict(fg, d2.X_val)
@tullio labels[j] := d.Y_val[j];

# Save to disk
model_name = "softmax_jul1_conv_maxpool"
mkpath("./models/") # create dir if it doesn't exist
h5write("models/class_probs_julia_$model_name.h5", "data", posterior_preds)
h5write("models/class_probs2_julia_$model_name.h5", "data", posterior_preds2)
h5write("models/labels_julia_$model_name.h5", "data", labels)


###
### Compare with results from Torch
###
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
model_name = "softmax_jul1_conv_maxpool"
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


# # Plot some wrong predictions
# num_print = 50
# for i in 1:length(preds)
#     if preds[i] != labels[i]
#         title = "Pred: $(preds[i]-1), Label: $(labels[i]-1)"
#         img = reverse(transpose(reshape(d.X_val[:, i], 28, 28)), dims=1)
#         display(heatmap(img, c=:grays, aspect_ratio=:equal, axis=false; title))

#         num_print -= 1
#         if num_print == 0
#             break
#         end
#     end
# end