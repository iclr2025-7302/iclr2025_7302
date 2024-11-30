include("lib/utils.jl")
@CUDA_RUN using CUDA, KernelAbstractions, CUDA.CUDAKernels

include("lib/datasets.jl")
include("lib/factor_graph.jl")
include("lib/plotting.jl")
using HDF5
using Serialization

d = normalize_X!(CIFAR10())

if !isfile("cifar10_factor_graph.jls")
    println("Training the model requires quite some memory and time. Do you want to download a pretrained model? (Y/N)")
    download_model = readline()
    if download_model == "Y" || download_model == "y"
        println("Downloading pretrained model...")
        run(`python cifar10_pretrained_download.py`)
    elseif download_model == "N" || download_model == "n"
        println("Training model from scratch...")
        Random.seed!(1337)
        batch_size = 128
        fg = create_factor_graph([
                size(d.X_train)[1:end-1], # (3, 32, 32)
                # First Block
                (:Conv, 32, 3, 0), # (32, 30, 30)
                (:LeakyReLU, 0.1),
                (:Conv, 32, 3, 0), # (32, 28, 28)
                (:LeakyReLU, 0.1),
                (:MaxPool, 2), # (32, 14, 14)
                # Second Block
                (:Conv, 64, 3, 0), # (64, 12, 12)
                (:LeakyReLU, 0.1),
                (:Conv, 64, 3, 0), # (64, 10, 10)
                (:LeakyReLU, 0.1),
                (:MaxPool, 2), # (64, 5, 5)
                # Head
                (:Flatten,), # (64*5*5 = 1600)
                (:Linear, 512), # (512)
                (:LeakyReLU, 0.1),
                (:Linear, 10), # (10)
                (:Argmax, true)
            ], batch_size)

        @CUDA_RUN fg = adapt(CuArray, fg)

        println("Training data size: $(size(d.X_train, ndims(d.X_train)))")

        trainer = Trainer(fg, d.X_train, d.Y_train)

        silent = false
        num_epochs = 25
        steps = [Int(ceil(epoch_number / 2)) for epoch_number in 1:num_epochs]

        for (i, num_training_its) in enumerate(steps)
            train(trainer; num_epochs=1, num_training_its, silent)

            if (silent == false) || i == length(steps)
                evaluate(fg, d.X_val, d.Y_val; silent)
            end
        end
        serialize("cifar10_factor_graph.jls", fg)
    else
        error("Invalid input")
        exit(1)
    end
end

fg = deserialize("cifar10_factor_graph.jls")

println("Predicting on test set...")
preds = predict(fg, d.X_val)
labels = d.Y_val

println("Prediction on the SVHN dataset for OOD detection...")
out_of_distribution_dataset = SVHN2()
out_of_distribution_preds = predict(fg, out_of_distribution_dataset.X_val)

println("Saving predictions and labels...")
save_dir = "experiment_results/experiment_cifar10"
mkpath(save_dir)
h5write(joinpath(save_dir, "MP.h5"), "preds", preds)
h5write(joinpath(save_dir, "MP.h5"), "labels", labels)
h5write(joinpath(save_dir, "MP_svhn.h5"), "preds", out_of_distribution_preds)
h5write(joinpath(save_dir, "MP_svhn.h5"), "labels", out_of_distribution_dataset.Y_val)