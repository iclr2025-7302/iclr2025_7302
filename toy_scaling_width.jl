include("lib/datasets.jl")
include("lib/factor_graph.jl")
include("lib/plotting.jl")


# Generate data
Random.seed!(98)
d = SineDataset(100, stretch=1.0, σ=0.05)
batch_size = 80

for w in [4, 16, 32, 64, 128]
    # w = 128
    fg = create_factor_graph([
            (1,),
            (:Linear, w),
            (:LeakyReLU, 0.1),
            (:Linear, 1),
            (:Regression, 0.05^2)
        ], batch_size
    )

    n_its = 500
    train_batch(fg, d.X_train, d.Y_train; num_training_its=n_its)

    # Evaluation
    X_pred = FloatType.(minimum(d.X_train)-0.8:0.005:maximum(d.X_train)+0.8)
    posterior_preds = predict(fg, Matrix(X_pred'))
    if ndims(posterior_preds) == 2 # Handle both old and new FactorGraph format
        posterior_preds = posterior_preds[1, :]
    end

    save_name = "width_$(w).png"
    plot_posterior_preds_pretty(d, Matrix(X_pred'), posterior_preds; ylims=(-1.7, 2), save_name, dpi=300)
end
