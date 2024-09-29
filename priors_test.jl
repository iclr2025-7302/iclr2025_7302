include("lib/utils.jl")
@CUDA_RUN using CUDA, KernelAbstractions, CUDA.CUDAKernels

include("lib/datasets.jl")
include("lib/factor_graph.jl")
include("lib/plotting.jl")
using HDF5



###
### Test whole network
###
# Parameters
for i in 1:10
    Random.seed!(98)

    layers = Vector{Tuple}()

    append!(layers, [(28, 28, 1), (:Flatten,), (:Linear, 100), (:LeakyReLU, 0.1)])
    for _ in 1:i
        append!(layers, [(:Linear, 100), (:LeakyReLU, 0.1)])
    end
    append!(layers, [(:Linear, 10), (:Regression, 1e-8)])

    fg = create_factor_graph(convert(Vector{<:Tuple}, layers), 320)
    # fg = create_factor_graph([
    #         (28, 28, 1),
    #         (:Flatten,),
    #         (:Linear, 100),
    #         (:LeakyReLU, 0.1),
    #         (:Linear, 100),
    #         (:LeakyReLU, 0.1),
    #         (:Linear, 100),
    #         (:LeakyReLU, 0.1),
    #         (:Linear, 10),
    #         (:Regression, 0.000001)
    #     ], 320
    # )
    X = randn(FloatType, 28, 28, 1, 1)
    out = predict(fg, X; silent=true)
    # display(out)

    println("μ_μ: $(round(Statistics.mean(abs.(mean.(out))), digits=4)) σ2_μ: $(round(Statistics.var(mean.(out)), digits=4)) μ_σ^2: $(round(Statistics.mean(variance.(out)), digits=4))")
end


###
### Estimate LeakyReLU transformation of variables
###
leaks = exp.(collect(0:-0.1:-10))
m_m = []
v_m = []
m_v = []

n = Int(1e8)

for leak in ProgressBar(leaks)
    μ = CUDA.randn(FloatType, n)
    σ2 = CUDA.fill(2.0, n)
    x = GaussianTensor(; μ, σ2)
    free_if_CUDA!.((μ, σ2))

    out = forward_leaky_relu_factor.(x, leak)
    free_if_CUDA!(x)

    means = mean.(out)
    variances = variance.(out)
    free_if_CUDA!(out)

    push!(m_m, Statistics.mean(means))
    push!(v_m, Statistics.var(means))
    push!(m_v, Statistics.mean(variances))
    free_if_CUDA!.((means, variances))
end


X = hcat(ones(FloatType, length(leaks)), leaks)
w1 = X \ FloatType.(m_m)
y = X * w1

display(w1)
p = plot(leaks, m_m, label="Data", title="Average Error: $(Statistics.mean(abs.(y .- m_m)))", xlabel="leak", ylabel="Avg. Mean")
plot!(leaks, y, labels="Fit")
display(p)

X = hcat(ones(FloatType, length(leaks)), leaks, leaks .^ 2)
w2 = X \ FloatType.(v_m)
y = X * w2
display(w2)
p = plot(leaks, v_m, label="Data", title="Average Error: $(Statistics.mean(abs.(y .- v_m)))", xlabel="leak", ylabel="Avg. Variance")
plot!(leaks, y, labels="Fit")
display(p)

X = hcat(ones(FloatType, length(leaks)), leaks, leaks .^ 2)
w3 = X \ FloatType.(m_v)
y = X * w3
display(w3)
p = plot(leaks, m_v, label="Data", title="Average Error: $(Statistics.mean(abs.(y .- m_v)))", xlabel="leak", ylabel="Avg. Variance")
plot!(leaks, y, labels="Fit")
display(p)

# (0.5 - w2' * [1.0, 0.1, 0.1^2]) / (([w1; 0] .^ 2 .+ w2)' * [1.0, 0.1, 0.1^2])


###
### Demonstrate 1st and 2nd Layer transformation
###
n, m = 100, 100
σ = (1 / sqrt(n)) * min(1.0, sqrt(m / n))
for _ in 1:1
    println("1st Layer")
    x = randn(FloatType, n)

    μ_W = randn(FloatType, n, m) * σ
    σ2_W = 1.5 / n
    @tullio W[i, j] := Gaussian1d(; μ=μ_W[i, j], σ2=σ2_W)

    b = GaussianTensor(; μ=fill(0.0, m), σ2=fill(0.5, m))
    out = similar(b)

    forward_mult(x, W, b, out)
    display(out)
    display(Statistics.mean(variance.(out)))
end

for _ in 1:1
    # println("2nd Layer")
    leak = 0.1
    leak_x = [1.0, leak, leak^2]
    μ_μ_x = leak_x[1:2]' * w1
    σ2_μ_x = leak_x' * w2
    σ2_x = leak_x' * w3

    x = GaussianTensor(; μ=randn(FloatType, n), σ2=fill(2.0, n))
    x = forward_leaky_relu_factor.(x, leak)

    μ_W = randn(FloatType, n, m) * σ
    σ2_W = (1.5 - n * σ2_x * σ^2) / (n * (σ2_μ_x * (1 + μ_μ_x^2) + σ2_x))

    # To collect the parameters for the FactorGraph implementation, uncomment these two lines:
    # display(σ2_μ_x * (1 + μ_μ_x^2))
    # display(σ2_x)
    @tullio W[i, j] := Gaussian1d(; μ=μ_W[i, j], σ2=σ2_W)

    b = GaussianTensor(; μ=fill(0.0, m), σ2=fill(0.5, m))
    out2 = similar(b)

    forward_mult(x, W, b, out2)
    display(out2)
    display(Statistics.mean(variance.(out2)))
end