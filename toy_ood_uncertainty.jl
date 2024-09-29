using Plots
using BenchmarkTools
include("lib/datasets.jl")
include("lib/factor_graph.jl")
include("lib/plotting.jl")

###
### Experiment from the paper:
###
Random.seed!(98)
d = SineDataset(250, stretch=1.0, σ=0.05, offset=-0.5)
batch_size = 200
fgs = fill(FactorGraph([], 1, Array), 100)

Threads.@threads for i in 1:100
    fg = create_factor_graph([
            (1,),
            (:Linear, 32),
            (:LeakyReLU, 0.1),
            (:Linear, 32),
            (:LeakyReLU, 0.1),
            (:Linear, 1),
            (:Regression, 0.05^2)
        ], batch_size)
    # @CUDA_RUN fg = adapt(CuArray, fg)

    n_its = 500
    trainer = Trainer(fg, d.X_train, d.Y_train)
    train(trainer; num_epochs=1, num_training_its=n_its, silent=true)
    # train_batch(fg, d.X_train, d.Y_train; num_training_its=n_its)

    fgs[i] = fg
end


x_pred = FloatType.(minimum(d.X_train)-20.0:0.05:maximum(d.X_train)+20.0)
X_pred = Matrix(x_pred')
Y_pred = NaturalGaussianTensor(length(fgs), length(x_pred));
y_true = generate_true_sine_data(x_pred);

for (i, fg) in ProgressBar(enumerate(fgs))
    posterior_preds = predict(fg, X_pred, silent=true)
    Y_pred[i, :] .= posterior_preds[1, :]
end

# Plot the true curve against the data
# p = plot(x_pred, y_true)
# scatter!(p, d.X_train[1,:], d.Y_train[1,:])

@tullio zs[i, j] := cdf_normal(y_true[j], Y_pred[i, j])
@tullio perc[j] := sum(0.16 < zs[i, j] < 0.84) / size(zs, 1)
@tullio perc2[j] := sum(0.025 < zs[i, j] < 0.975) / size(zs, 1)
@tullio perc3[j] := sum(0.0015 < zs[i, j] < 0.9985) / size(zs, 1)

# title="Coverage probability of prediction interval"
p = plot(x_pred, 100 .* perc, ylabel="Percent containing true data function", label="1σ", dpi=300)
plot!(p, x_pred, 100 .* perc2, label="2σ", dpi=300)
plot!(p, x_pred, 100 .* perc3, label="3σ", dpi=300)
savefig(p, "ood_uncertainty_picp.png")
display(p)


# Median stats
median(perc[x_pred.<-10]) # -> 61%
median(perc2[x_pred.<-10]) # -> 86%
median(perc3[x_pred.<-10]) # -> 93%

median(perc[x_pred.>10]) # -> 36%
median(perc2[x_pred.>10]) # -> 68%
median(perc3[x_pred.>10]) # -> 90%


# Obtain correlations
masses = []
percs_pos = []
percs_neg = []
for i in 1:100
    bl, bu = 0.5 - 0.5 * (i / 100), 0.5 + 0.5 * (i / 100)

    @tullio p[j] := sum(bl < zs[i, j] < bu) / size(zs, 1)
    push!(masses, i / 100)
    push!(percs_pos, median(p[x_pred.>10]))
    push!(percs_neg, median(p[x_pred.<-10]))
end
p = scatter(masses, percs_pos, label=">0", dpi=300)
scatter!(p, masses, percs_neg, label="<0")
display(p)
savefig(p, "ood_correlation.png")

Statistics.cor([masses..., masses...], [percs_pos..., percs_neg...]) # 0.901066001092147
Statistics.cor(masses, percs_pos) # 0.9607011468690524
Statistics.cor(masses, percs_neg) # 0.9929185540691821


# Plot 4 examples:
for i in 1:4
    p = plot_posterior_preds_pretty(d, X_pred, Y_pred[i, :]; ylims=(-20, 30), show_data=false, dpi=300)
    plot!(p, x_pred, y_true, label="True data function", color=:blue, dpi=300)
    savefig(p, "ood_uncertainty$i.png")
end



###
### Another experiment (sample true data-generating function from prior of network)
###
Random.seed!(98)
d = SineDataset(250, stretch=1.0, σ=0.05)
batch_size = 200
x_pred = FloatType.(minimum(d.X_train)-20.0:0.05:maximum(d.X_train)+20.0)
X_pred = Matrix(x_pred')

n = 100
zs = zeros(FloatType, n, length(x_pred))

Threads.@threads for i in 1:n
    fg = create_factor_graph([
            (1,),
            (:Linear, 32),
            (:LeakyReLU, 0.1),
            (:Linear, 32),
            (:LeakyReLU, 0.1),
            (:Linear, 1),
            (:Regression, 0.05^2)
        ], batch_size)


    function leaky_relu_elem(x::FloatType)
        return max(x, 0.1 * x)
    end

    function create_sample(fg::FactorGraph, X::Matrix{FloatType}, X2::Matrix{FloatType})
        W1 = first.(sample.(fg.layers[1].last_W_marginal))
        W2 = first.(sample.(fg.layers[3].last_W_marginal))
        W3 = first.(sample.(fg.layers[5].last_W_marginal))

        Y1 = transpose_permutedims(W3) * leaky_relu_elem.(transpose_permutedims(W2) * leaky_relu_elem.(transpose_permutedims(W1) * X))
        Y2 = transpose_permutedims(W3) * leaky_relu_elem.(transpose_permutedims(W2) * leaky_relu_elem.(transpose_permutedims(W1) * X2))
        return Y1, Y2
    end

    Y_train, Y_val = create_sample(fg, d.X_train, X_pred)
    d2 = Dataset(d.X_train, Y_train, d.X_val, Y_val)

    n_its = 50
    trainer = Trainer(fg, d2.X_train, d2.Y_train)
    train(trainer; num_epochs=1, num_training_its=n_its, silent=true)

    # Evaluation
    posterior_preds = predict(fg, X_pred, silent=true)
    if ndims(posterior_preds) == 2 # Handle both old and new FactorGraph format
        posterior_preds = posterior_preds[1, :]
    end

    zs_i = selectdim(zs, 1, i)
    @tullio zs_i[j] = cdf_normal(Y_val[1, j], posterior_preds[j])

    # p = plot_posterior_preds_pretty(d2, X_pred, posterior_preds; ylims=(-50, 50), dpi=300)
    # plot!(p, x_pred, Y_val[1, :], label="True data function", color=:blue, dpi=300)
    # display(p)
end

println("Minimum z-score: $(minimum(zs))\nMaximum z-score: $(maximum(zs))")

@tullio perc[j] := sum(0.16 < zs[i, j] < 0.84) / size(zs, 1)
@tullio perc2[j] := sum(0.025 < zs[i, j] < 0.975) / size(zs, 1)
@tullio perc3[j] := sum(0.0015 < zs[i, j] < 0.9985) / size(zs, 1)

p = plot(x_pred, 100 .* perc, ylabel="Percent containing true data function", label="1σ", dpi=300)
plot!(p, x_pred, 100 .* perc2, label="2σ", dpi=300)
plot!(p, x_pred, 100 .* perc3, label="3σ", dpi=300)
savefig(p, "ood_uncertainty_picp.png")
display(p)