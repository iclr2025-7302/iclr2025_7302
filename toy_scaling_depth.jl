using Plots
using BenchmarkTools
include("lib/datasets.jl")
include("lib/factor_graph.jl")
include("lib/plotting.jl")


# seed = 98 # Works best
# seed = 753285 # Works okay/good
seed = 1293 # Works worst/okay

Random.seed!(seed)
d = SineDataset(250, stretch=2.0, σ=0.05)
batch_size = 200

# Training
for num_layers in 0:4
    w = 16

    layers = Vector{Tuple}()
    append!(layers, [(1,), (:Linear, w), (:LeakyReLU, 0.1)])
    for _ in 1:num_layers
        append!(layers, [(:Linear, w), (:LeakyReLU, 0.1)])
    end
    append!(layers, [(:Linear, 1), (:Regression, 0.05^2)])

    fg = create_factor_graph(convert(Vector{<:Tuple}, layers), batch_size)

    n_its = 500
    train_batch(fg, d.X_train, d.Y_train; num_training_its=n_its)

    # Evaluation
    X_pred = FloatType.(minimum(d.X_train)-1.5:0.005:maximum(d.X_train)+1.5)
    posterior_preds = predict(fg, Matrix(X_pred'))
    if ndims(posterior_preds) == 2 # Handle both old and new FactorGraph format
        posterior_preds = posterior_preds[1, :]
    end

    save_name = "depth_$(num_layers)_$seed.png"
    plot_posterior_preds_pretty(d, Matrix(X_pred'), posterior_preds; ylims=(-2, 3), save_name, dpi=300)
end
