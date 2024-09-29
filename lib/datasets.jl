include("utils.jl")
using Tullio
import Distributions: Normal, pdf
import MLDatasets # Commented out for performance reasons. Increases import load from 6.3s to 12.2s in a test
import StatsBase
import Random
import Statistics

struct Dataset{A_F<:AbstractArray{FloatType}}
    X_train::A_F #(feature, sample)
    Y_train::Matrix{FloatType}

    X_val::A_F
    Y_val::Matrix{FloatType}
end

struct ClassificationDataset{A_F<:AbstractArray{FloatType}}
    X_train::A_F #(feature, sample)
    Y_train::Vector{Int}

    X_val::A_F
    Y_val::Vector{Int}

    num_classes::Int
end


###
### Dataset Transformations
###
function as_regression_dataset(d::ClassificationDataset)
    # One-hot encoding (in floats)
    Y_train = zeros(FloatType, d.num_classes, length(d.Y_train))
    Y_test = zeros(FloatType, d.num_classes, length(d.Y_val))

    Y_train[CartesianIndex.(d.Y_train, 1:length(d.Y_train))] .= 1
    Y_test[CartesianIndex.(d.Y_val, 1:length(d.Y_val))] .= 1

    return Dataset(d.X_train, Y_train, d.X_val, Y_test)
end

function normalize_X!(d::Union{Dataset,ClassificationDataset})
    μ_x, σ2_x = Statistics.mean(d.X_train), Statistics.var(d.X_train)

    d.X_train .= (d.X_train .- μ_x) ./ sqrt(σ2_x)
    d.X_val .= (d.X_val .- μ_x) ./ sqrt(σ2_x)
    return d
end


###
### Constructors that create a train-val-split
###
# TODO: Improve (similar to ClassificationDataset)
function Dataset(X::Matrix{FloatType}, Y::Matrix{FloatType}; train_perc::FloatType=FloatType(0.8))
    # Both X and y should have dimensions (k1, n) and (k2, n) respectively
    n = size(X, 2)
    @assert (length(size(X)) == 2)
    @assert (length(size(Y)) == 2) && (size(Y, 2) == n)

    indices = StatsBase.sample(1:n, round(Int, train_perc * n), replace=false)
    other_indices = collect(setdiff(Set(1:n), Set(indices)))

    X_train = X[:, indices]
    Y_train = Y[:, indices]
    X_val = X[:, other_indices]
    Y_val = Y[:, other_indices]

    return Dataset(X_train, Y_train, X_val, Y_val)
end

function ClassificationDataset(X::AbstractArray{FloatType}, y::Vector{Int}; train_perc::FloatType=FloatType(0.8))
    nd = ndims(X)
    n = size(X, nd)

    num_classes = maximum(y)
    @assert length(y) == n
    @assert minimum(y) == 1

    indices = StatsBase.sample(1:n, round(Int, train_perc * n), replace=false)
    other_indices = collect(setdiff(Set(1:n), Set(indices)))

    X_train = copy(selectdim(X, nd, indices))
    Y_train = y[indices]
    X_val = copy(selectdim(X, nd, other_indices))
    Y_val = y[other_indices]

    return ClassificationDataset(X_train, Y_train, X_val, Y_val, num_classes)
end


###
### Image Datasets
###
function MNIST()
    print("Loading MNIST...")
    X_train, Y_train = MLDatasets.MNIST(split=:train)[:]
    X_test, Y_test = MLDatasets.MNIST(split=:test)[:]
    println(" Done!")

    # Shuffle the order
    n_train, n_test = size(X_train, 3), size(X_test, 3)
    o1 = Random.shuffle(1:n_train)
    o2 = Random.shuffle(1:n_test)

    # Classes should begin with 1
    num_classes = 10
    Y_train .+= 1
    Y_test .+= 1

    # Reshape
    X_train = reshape(FloatType.(X_train[:, :, o1]), 28, 28, 1, :)
    X_test = reshape(FloatType.(X_test[:, :, o2]), 28, 28, 1, :)

    @assert all(1 .<= Y_train .<= num_classes)
    return ClassificationDataset(X_train, Y_train[o1], X_test, Y_test[o2], num_classes)
end

# Sorry for the code duplication
function FashionMNIST()
    print("Loading FashionMNIST...")
    X_train, Y_train = MLDatasets.FashionMNIST(split=:train)[:]
    X_test, Y_test = MLDatasets.FashionMNIST(split=:test)[:]
    println(" Done!")

    # Shuffle the order
    n_train, n_test = size(X_train, 3), size(X_test, 3)
    o1 = Random.shuffle(1:n_train)
    o2 = Random.shuffle(1:n_test)

    # Classes should begin with 1
    num_classes = 10
    Y_train .+= 1
    Y_test .+= 1

    # Reshape
    X_train = reshape(FloatType.(X_train[:, :, o1]), 28, 28, 1, :)
    X_test = reshape(FloatType.(X_test[:, :, o2]), 28, 28, 1, :)

    @assert all(1 .<= Y_train .<= num_classes)
    return ClassificationDataset(X_train, Y_train[o1], X_test, Y_test[o2], num_classes)
end

# Sorry for the code duplication again
function CIFAR10()
    print("Loading CIFAR10...")
    X_train, Y_train = MLDatasets.CIFAR10(split=:train)[:]
    X_test, Y_test = MLDatasets.CIFAR10(split=:test)[:]
    println(" Done!")

    # Shuffle the order
    n_train, n_test = size(X_train, ndims(X_train)), size(X_test, ndims(X_train))
    o1 = Random.shuffle(1:n_train)
    o2 = Random.shuffle(1:n_test)

    # Classes should begin with 1
    num_classes = 10
    Y_train .+= 1
    Y_test .+= 1
    @assert all(1 .<= Y_train .<= num_classes)

    return ClassificationDataset(FloatType.(X_train[:, :, :, o1]), Y_train[o1], FloatType.(X_test[:, :, :, o2]), Y_test[o2], num_classes)
end


###
### Simple Datasets
###
function toy_classification_ds(n::Int64)
    # Uniform sample from [0, stretch)
    X = randn(FloatType, 10, n)
    w = randn(FloatType, 10)

    y_pred = w' * X
    m_pred = median(y_pred)
    @tullio y[i] := ifelse(y_pred[i] < m_pred, 2, 1)
    return ClassificationDataset(X, y; train_perc=0.5)
end

function mixture_classification_ds(n::Int64; num_classes::Int=2, d::Int=10)
    # Uniform sample from [0, stretch)
    X = randn(FloatType, d, n)
    y = zeros(Int, n)
    for i in 1:num_classes
        i1 = Int(1 + floor(n / num_classes * (i - 1)))
        i2 = Int(ceil(n / num_classes * i))

        # Create random MVNormal params
        μ = randn(d)
        A = randn(d, d)
        Σ = A * A' + I * 1e-3  # Adding a small value to the diagonal for numerical stability
        mv_gaussian = MvNormal(μ, Σ)

        Xi = selectdim(X, 2, i1:i2)
        yi = (@view y[i1:i2])
        @tullio Xi[:, j] = rand(mv_gaussian)
        yi .= i
    end
    return ClassificationDataset(X, y; train_perc=0.5)
end

# Samples n points from a composite sine function and adds noise ~ N(0, σ^2)
function SineDataset(
    n::Int64;
    a::FloatType=FloatType(0.5),
    b::FloatType=FloatType(0.2),
    c::FloatType=FloatType(0.3),
    σ::FloatType=FloatType(0.05),
    stretch::Union{FloatType,Int64}=3,
    offset::FloatType=FloatType(0.0),
    leave_gap::Bool=false,
)
    # Uniform sample from [0, stretch)
    X = rand(FloatType, (1, n)) .+ offset
    if leave_gap
        # Push all the points in the middle 0.5 to the two edges
        @tullio X[i] = ifelse(X[i] > 0.25 && X[i] <= 0.5,
            X[i] - 0.25,
            ifelse(X[i] > 0.5 && X[i] <= 0.75,
                X[i] + 0.25,
                X[i],
            )
        )
    end
    X .*= stretch

    Y = a .* X + b .* sin.(FloatType(2π) .* X) + c .* sin.(FloatType(4π) .* X)
    Y = Y .+ rand(Normal(0, σ), size(Y))

    return Dataset(X, Y)
end

function generate_true_sine_data(x::Vector{FloatType}; a::FloatType=0.5, b::FloatType=0.2, c::FloatType=0.3)
    @tullio y[i] := a * x[i] + b * sin(2π * x[i]) + c * sin(4π * x[i])
    return y
end

function XOR_dataset(; n=1000)
    # Number of points to generate in each quadrant
    M = round(Int, n / 4)
    Random.seed!(1234)

    dist_axes = FloatType(0.5)
    abs_max = 5

    # Generate artificial data
    x = rand(M) * (abs_max - dist_axes) .+ dist_axes
    y = rand(M) * (abs_max - dist_axes) .+ dist_axes
    features_class0 = [x'; y']
    x = rand(M) * (abs_max - dist_axes) .- abs_max
    y = rand(M) * (abs_max - dist_axes) .- abs_max
    features_class0 = [features_class0 [x'; y']]

    x = rand(M) * (abs_max - dist_axes) .- abs_max
    y = rand(M) * (abs_max - dist_axes) .+ dist_axes
    features_class1 = [x'; y']
    x = rand(M) * (abs_max - dist_axes) .+ dist_axes
    y = rand(M) * (abs_max - dist_axes) .- abs_max
    features_class1 = [features_class1 [x'; y']]

    features = Matrix{FloatType}([features_class0 features_class1])
    labels = Matrix{Int}([zeros(2 * M)' ones(2 * M)'])
    return ClassificationDataset(features, 1 .+ labels[1, :])
end

function heteroskedastic_1D_regression(; n=500)
    model(x) = -(x + 0.5) * sin(3 * pi * x)
    noise(x) = 0.45 * (x + 0.5)^2
    features = rand(n) .- 0.5
    targets = model.(features) .+ noise.(features) .* randn(n)
    return Dataset(
        Matrix{FloatType}(reshape(features, (1, n))),
        Matrix{FloatType}(reshape(targets, (1, n)))
    )
end