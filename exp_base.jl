include("lib/utils.jl")
@CUDA_RUN using CUDA, KernelAbstractions, CUDA.CUDAKernels

include("lib/datasets.jl")
include("lib/factor_graph.jl")
include("lib/plotting.jl")
using HDF5


function run_experiment(layers::Vector{<:Tuple}, fg_name::String; is_regression::Bool=true, dir::String="./models", seed::Int=98, nts=[60000])
    for nt in nts
        # Parameters
        Random.seed!(seed)
        batch_size = 320

        # Prepare training
        @CUDA_RUN println("Using CUDA")
        println("Num Threads: $(Threads.nthreads())")

        d = (is_regression
             ? as_regression_dataset(normalize_X!(MNIST()))
             : normalize_X!(MNIST()))
        fg = create_factor_graph([
                size(d.X_train)[1:end-1],
                layers...
            ], batch_size
        )
        @CUDA_RUN fg = adapt(CuArray, fg)

        println("Training data size: $(size(d.X_train, ndims(d.X_train)))")
        # nt = 1280
        # nt = size(d.X_train, 4)
        println("Of which used: $nt")

        trainer = Trainer(fg, d.X_train[:, :, :, 1:nt], copy(selectdim(d.Y_train, ndims(d.Y_train), 1:nt)))
        # posterior_preds = forward_softmax(predict(fg, d.X_val))

        silent = true
        steps = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
        # steps = [1, 1, 2, 2, 3, 3, 4, 4]
        for (i, num_training_its) in optional_progress_bar(enumerate(steps), !silent)
            train(trainer; num_epochs=1, num_training_its, silent)
        end

        # Evaluate
        acc = evaluate(fg, d.X_val, d.Y_val; as_classification=true, silent)
        d2 = (is_regression
              ? as_regression_dataset(normalize_X!(FashionMNIST()))
              : normalize_X!(FashionMNIST())
        )
        posterior_preds = (is_regression
                           ? forward_argmax(predict(fg, d.X_val, silent=true))
                           : predict(fg, d.X_val, silent=true)
        )
        posterior_preds2 = (is_regression
                            ? forward_argmax(predict(fg, d2.X_val, silent=true))
                            : predict(fg, d2.X_val, silent=true)
        )
        labels = []
        if is_regression
            @tullio labels[j] := argmax(d.Y_val[:, j])
        else
            labels = d.Y_val
        end
        ece, auroc, ood_auroc, nll = get_calibration_stats(posterior_preds, labels, posterior_preds2)

        # Store results
        model_name = "$(fg_name)_$(ifelse(is_regression, """regression""", """classification"""))_$(nt)_$(seed)"
        results[model_name] = "Accuracy: $acc, ECE: $(round(ece, digits=4)), AUROC: $(round(auroc, digits=4)), OOD_AUROC: $(round(ood_auroc, digits=4)), NLL: $nll"

        h5write("$(dir)/class_probs_julia_$(model_name).h5", "data", posterior_preds)
        h5write("$(dir)/class_probs2_julia_$(model_name).h5", "data", posterior_preds2)
        h5write("$(dir)/labels_julia_$(model_name).h5", "data", labels)
    end
end

function test_stored_models(fg_name::String; is_regression::Bool=true, dir::String="./models", seed::Int=98, nts=[60000], is_torch::Bool=false)
    for nt in nts
        # Retrieve model
        model_name = (is_torch
                      ? "$(fg_name)_$(ifelse(is_regression, """regression""", """classification"""))_$(nt)"
                      : "$(fg_name)_$(ifelse(is_regression, """regression""", """classification"""))_$(nt)_$(seed)"
        )
        opt_julia = is_torch ? "" : "julia_"

        posterior_preds = FloatType.(h5open("$(dir)/class_probs_$(opt_julia)$(model_name).h5", "r") do file
            read(file["data"])
        end)
        posterior_preds2 = FloatType.(h5open("$(dir)/class_probs2_$(opt_julia)$(model_name).h5", "r") do file
            read(file["data"])
        end)
        labels = (
            h5open("$(dir)/labels_$(opt_julia)$(model_name).h5", "r") do file
                read(file["data"])
            end
        ) .+ (is_torch ? 1 : 0)
        ece, auroc, ood_auroc, nll = get_calibration_stats(posterior_preds, labels, posterior_preds2)

        # Store results
        @tullio class_preds[j] := argmax(posterior_preds[:, j])
        acc = sum(class_preds .== labels) / length(labels)

        model_name = "$(fg_name)_$(ifelse(is_regression, """regression""", """classification"""))_$(lpad(nt,5,'0'))_$(seed)"
        results[model_name] = "Accuracy: $acc, ECE: $(round(ece, digits=4)), AUROC: $(round(auroc, digits=4)), OOD_AUROC: $(round(ood_auroc, digits=4)), NLL: $nll"

        # For copying specific results:
        # results[model_name] = round(acc, digits=4)
    end
    # For copying specific results:
    # results["$(fg_name)_$(ifelse(is_regression, """regression""", """classification"""))"] = -1
end