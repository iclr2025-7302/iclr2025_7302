include("./utils.jl")
@CUDA_RUN using CUDA, KernelAbstractions, CUDA.CUDAKernels
include("./gaussian.jl")
include("./message_equations.jl")
include("./messages_gaussian_mult.jl")
using ProgressBars
using Statistics: mean as Stats_mean
using BenchmarkTools
using Tullio
using Adapt
import Random

###
### Factor Type
###
abstract type Factor end

abstract type WeightFactor <: Factor end
# Assumed to at least have members:
# - "m_to_weights" and "last_W_marginal"
# - "m_to_biases" and "last_bias_marginal"

# Some nodes have some internal state that must be reset before switching batches
# For Weight nodes, see "reset_for_batch" as well
function reset_batch_caches(node::Factor)
    return
end

###
### Utilities
###
function set_spectral_batch_messages(prior_W::AbstractArray{Gaussian1d})
    d_out, b = size(prior_W)[(end-1):end]
    d_in = prod(size(prior_W)[1:(end-2)])

    actual_prior = selectdim(prior_W, ndims(prior_W), size(prior_W, ndims(prior_W)))
    m_from_batches = selectdim(prior_W, ndims(prior_W), 1:(size(prior_W, ndims(prior_W))-1))

    # See spectral parametrization (https://arxiv.org/pdf/2310.17813.pdf)
    σ_l = (1 / sqrt(d_in)) * min(1, sqrt(d_out / d_in))

    # Assumes that the prior has homogeneous variance and that it is stored at the end
    @CUDA_RUN CUDA.allowscalar() do (prior_ρ = actual_prior[1].ρ)
    end
    @NOT_CUDA_RUN prior_ρ = actual_prior[1].ρ
    remaining_ρ = max(0, (1 / σ_l) - prior_ρ)

    # Set the messages from the batches so that the prior predictive has O(1) variance
    @assert all(m_from_batches .== Gaussian1d())
    m_from_batches .= Gaussian1d(0.0, remaining_ρ / (b - 1))
    return
end

function create_bias_prior(d_out::Int)
    return GaussianTensor(; μ=zeros(FloatType, d_out), σ2=fill(0.5, d_out))
end

function create_weight_prior(d_in::Int, d_out::Int, prior_σ2_val::Union{FloatType,Nothing}; post_leaky::Bool=false)
    # Choose the means of the prior according to Spectral parametrization (https://arxiv.org/pdf/2310.17813.pdf)
    σ_l = (1 / sqrt(d_in)) * min(1, sqrt(d_out / d_in))
    μ_prior = σ_l * randn(FloatType, (d_in, d_out))

    # See priors_test (or thesis) for a justification of this prior variance
    σ2_prior = fill(1.5 / d_in, size(μ_prior))
    if post_leaky
        a = 0.44958619556324186
        b = 0.8040586726631379
        σ2_prior .= (1.5 - min(1.0, d_out / d_in) * b) / (d_in * a + b)
    end


    if !isnothing(prior_σ2_val)
        σ2_prior .= prior_σ2_val
    end

    return GaussianTensor(; μ=μ_prior, σ2=σ2_prior)
end

function create_weight_prior_conv2d(k_h::Int, k_w::Int, f_in::Int, f_out::Int, out_h::Int, out_w::Int; post_leaky::Bool=false)
    # Each feature (one element of a feature map) is a sum of d_in transformed input features
    d_in = k_h * k_w * f_in

    # The total number of features
    d_out = f_out * out_h * out_w

    # Choose the means of the prior according to Spectral parametrization (https://arxiv.org/pdf/2310.17813.pdf)
    σ_l = (1 / sqrt(d_in)) * min(1, sqrt(d_out / d_in))
    μ_prior = σ_l * randn(FloatType, (k_w, k_h, f_in, f_out))

    # See priors_test (or thesis) for a justification of this prior variance
    σ2_prior = fill(1.5 / d_in, size(μ_prior))
    if post_leaky
        a = 0.44958619556324186
        b = 0.8040586726631379
        σ2_prior .= (1.5 - d_in * b * σ_l^2) / (d_in * a + b)
    end

    @tullio out[i, j, k, l] := Gaussian1d(; μ=μ_prior[i, j, k, l], σ2=σ2_prior[i, j, k, l])
    return out
end


# Update marginal by first dividing the old message out and then multiplying the new message in
function update_messages_to_weights(W_marginal::AbstractVector{Gaussian1d}, m_to_weights::AbstractVector{Gaussian1d}, new_m_to_weights::AbstractVector{Gaussian1d})
    @tullio W_marginal[i] = Gaussian1d(
        W_marginal[i].τ + new_m_to_weights[i].τ - m_to_weights[i].τ,
        W_marginal[i].ρ + new_m_to_weights[i].ρ - m_to_weights[i].ρ
    )
    m_to_weights .= new_m_to_weights
    return
end
function update_messages_to_weights(W_marginal::AbstractMatrix{Gaussian1d}, m_to_weights::AbstractMatrix{Gaussian1d}, new_m_to_weights::AbstractMatrix{Gaussian1d})
    @tullio W_marginal[i, j] = Gaussian1d(
        W_marginal[i, j].τ + new_m_to_weights[i, j].τ - m_to_weights[i, j].τ,
        W_marginal[i, j].ρ + new_m_to_weights[i, j].ρ - m_to_weights[i, j].ρ
    )
    m_to_weights .= new_m_to_weights
    return
end
function update_messages_to_weights(W_marginal::AbstractArray{Gaussian1d,4}, m_to_weights::AbstractArray{Gaussian1d,4}, new_m_to_weights::AbstractArray{Gaussian1d,4})
    @tullio W_marginal[i, j, k, l] = Gaussian1d(
        W_marginal[i, j, k, l].τ + new_m_to_weights[i, j, k, l].τ - m_to_weights[i, j, k, l].τ,
        W_marginal[i, j, k, l].ρ + new_m_to_weights[i, j, k, l].ρ - m_to_weights[i, j, k, l].ρ
    )
    m_to_weights .= new_m_to_weights
    return
end


# Multiplied with the prior, this would give the posterior of weights (aka weights marginal)
function store_likelihood_messages(node::WeightFactor, out_W::AbstractArray{Gaussian1d}, out_bias::AbstractVector{Gaussian1d}; β_EMA::FloatType=1.0)
    @assert size(out_W) == size(node.last_W_marginal)
    @assert size(out_bias) == size(node.last_bias_marginal)

    nd = ndims(node.m_to_weights)
    nb = ndims(node.m_to_biases)
    b = size(node.m_to_weights, nd)
    @assert nb == 2

    # Update likelihood of weights
    m_to_weights = selectdim(node.m_to_weights, nd, 1:(b-1)) # exclude prior
    out_W_new = prod(m_to_weights, dims=ndims(m_to_weights))
    out_W .= EMA.(out_W, reshape(out_W_new, size(out_W_new)[1:(end-1)]), β_EMA)
    free_if_CUDA!(out_W_new)
    # prod!(reshape(out_W, size(out_W)..., 1), m_to_weights)

    # Update likelihood of biases
    m_to_biases = selectdim(node.m_to_biases, nb, 1:(b-1)) # exclude prior
    prod!(reshape(out_bias, size(out_bias)..., 1), m_to_biases)
    return
end

# Multiply over out-most dimension
function recompute_marginal(node::WeightFactor)
    # Update W_marginal
    prod!(reshape(node.last_W_marginal, size(node.last_W_marginal)..., 1), node.m_to_weights)

    # Update bias_marginal
    prod!(reshape(node.last_bias_marginal, size(node.last_bias_marginal)..., 1), node.m_to_biases)
    return
end

function reset_for_batch(node::WeightFactor, new_prior::AbstractArray{Gaussian1d}, new_bias_prior::AbstractVector{Gaussian1d})
    @assert size(new_prior) == size(node.last_W_marginal)
    nd = ndims(node.m_to_weights)

    node.m_to_weights .= Gaussian1d()
    selectdim(node.m_to_weights, nd, size(node.m_to_weights, nd)) .= new_prior
    node.last_W_marginal .= new_prior

    node.m_to_biases[:, 1:end-1] .= Gaussian1d()
    node.m_to_biases[:, end] = new_bias_prior
    node.last_bias_marginal[:] = new_bias_prior
end

# Allow transferring layers to GPU (and back). Will put all array members on the respective device.
@inline function Adapt.adapt_structure(to, node::T) where {T<:Factor}
    fields = []
    for f in fieldnames(T)
        push!(fields, adapt(to, getproperty(node, f)))
    end
    return Base.typename(typeof(node)).wrapper(fields...)
end


# Takes a 3d input (h x w x f_in) and produces a 3d output (h_out x w_out x f_out)
mutable struct Conv2dFactor{V_G<:AbstractVector{Gaussian1d},M_G<:AbstractMatrix{Gaussian1d},A3_G<:AbstractArray{Gaussian1d,3},A4_G<:AbstractArray{Gaussian1d,4},A5_G<:AbstractArray{Gaussian1d,5}} <: WeightFactor
    # Store weights
    m_to_weights::A5_G
    last_W_marginal::A4_G
    new_m_to_weights::A4_G # temp array used in the backward pass

    # Store biases
    m_to_biases::M_G
    last_bias_marginal::V_G
    new_m_to_biases::V_G

    # Messages from the forward pass
    last_in_marginal::A3_G
    last_m_out::A3_G

    # Messages from the backward pass
    last_m_to_inputs::A4_G

    # Parameters
    padding::Int
end

# Creates LinearLayer (n, m) (plus a bias term) for batch size b
function Conv2dFactor(h::Int, w::Int, f_in::Int, f_out::Int, k::Int, b::Int, p::Int)
    @assert k < h && k < w
    out_h, out_w = (2 * p + h - k + 1), (2 * p + w - k + 1)

    # Create arrays for weights and biases
    m_to_weights = NaturalGaussianTensor(k, k, f_in, f_out, b + 1)
    last_W_marginal = NaturalGaussianTensor(k, k, f_in, f_out)
    new_m_to_weights = similar(last_W_marginal)

    m_to_biases = NaturalGaussianTensor(f_out, b + 1)
    last_bias_marginal = NaturalGaussianTensor(f_out)
    new_m_to_biases = NaturalGaussianTensor(f_out)

    # Generate priors
    m_to_weights[:, :, :, :, end] = create_weight_prior_conv2d(k, k, f_in, f_out, out_h, out_w; post_leaky=true)
    last_W_marginal[:, :, :, :] = m_to_weights[:, :, :, :, end]
    m_to_biases[:, end] = create_bias_prior(f_out)
    last_bias_marginal[:] = m_to_biases[:, end]

    # Other arguments:
    last_in_marginal = NaturalGaussianTensor(h, w, f_in)
    last_m_out = NaturalGaussianTensor(out_h, out_w, f_out)
    last_m_to_inputs = NaturalGaussianTensor(size(last_in_marginal)..., b)

    return Conv2dFactor(m_to_weights, last_W_marginal, new_m_to_weights, m_to_biases, last_bias_marginal, new_m_to_biases, last_in_marginal, last_m_out, last_m_to_inputs, p)
end

function forward_message(node::Conv2dFactor, m_in::AbstractArray{Gaussian1d,3}, i_in::Int)
    # Compute input marginals
    m_to_inputs = selectdim(node.last_m_to_inputs, 4, i_in)
    @tullio node.last_in_marginal[i, j, k] = m_in[i, j, k] * m_to_inputs[i, j, k]

    # Compute product-sum factor
    m_to_biases = selectdim(node.m_to_biases, 2, i_in)
    forward_conv_factor(node.last_in_marginal, node.last_W_marginal, node.last_bias_marginal, m_to_biases, node.last_m_out; padding=node.padding)
    return node.last_m_out
end

function forward_predict(node::Conv2dFactor, m_in::AbstractArray{Gaussian1d,4})
    # Compute weight marginal (just to be safe)
    recompute_marginal(node)

    return forward_conv_factor(m_in, node.last_W_marginal, node.last_bias_marginal; padding=node.padding)
end

function backward_message(node::Conv2dFactor, m_back::AbstractArray{Gaussian1d,3}, i_in::Int)
    # Product-sum factor backward:
    m_to_weights = selectdim(node.m_to_weights, 5, i_in)
    m_to_biases = selectdim(node.m_to_biases, 2, i_in)
    out_inputs = selectdim(node.last_m_to_inputs, 4, i_in)
    backward_conv_factor(node.last_in_marginal, node.last_W_marginal, node.last_bias_marginal, m_to_biases,
        node.last_m_out, m_back,
        out_inputs, node.new_m_to_weights, node.new_m_to_biases; padding=node.padding)

    # Update weight marginal and store new messages to weights in the proper place
    update_messages_to_weights(node.last_W_marginal, m_to_weights, node.new_m_to_weights)
    update_messages_to_weights(node.last_bias_marginal, m_to_biases, node.new_m_to_biases)

    # Make sure there is a min-precision going back
    @tullio out_inputs[i, j, k] = min_ρ(out_inputs[i, j, k], 1e-8)

    return out_inputs
end

function reset_batch_caches(node::Conv2dFactor)
    node.last_m_to_inputs .= Gaussian1d()
end


# Takes a 3d scalar input (not distributions)
mutable struct FirstConv2dFactor{V_G<:AbstractVector{Gaussian1d},M_G<:AbstractMatrix{Gaussian1d},A3_G<:AbstractArray{Gaussian1d,3},A3_F<:AbstractArray{FloatType,3},A4_G<:AbstractArray{Gaussian1d,4},A5_G<:AbstractArray{Gaussian1d,5}} <: WeightFactor
    # Store weights
    m_to_weights::A5_G
    last_W_marginal::A4_G
    new_m_to_weights::A4_G # temp array used in the backward pass

    # Store biases
    m_to_biases::M_G
    last_bias_marginal::V_G
    new_m_to_biases::V_G

    # Messages from the forward pass
    last_x_in::A3_F
    last_m_out::A3_G

    # Parameters
    padding::Int
end

function FirstConv2dFactor(h::Int, w::Int, f_in::Int, f_out::Int, k::Int, b::Int, p::Int)
    out_h, out_w = (2 * p + h - k + 1), (2 * p + w - k + 1)

    # Create arrays for weights and biases
    m_to_weights = NaturalGaussianTensor(k, k, f_in, f_out, b + 1)
    last_W_marginal = NaturalGaussianTensor(k, k, f_in, f_out)
    new_m_to_weights = similar(last_W_marginal)

    m_to_biases = NaturalGaussianTensor(f_out, b + 1)
    last_bias_marginal = NaturalGaussianTensor(f_out)
    new_m_to_biases = NaturalGaussianTensor(f_out)

    # Generate priors
    m_to_weights[:, :, :, :, end] = create_weight_prior_conv2d(k, k, f_in, f_out, out_h, out_w)
    last_W_marginal[:, :, :, :] = m_to_weights[:, :, :, :, end]
    m_to_biases[:, end] = create_bias_prior(f_out)
    last_bias_marginal[:] = m_to_biases[:, end]

    # Other arguments
    last_x_in = zeros(FloatType, h, w, f_in)
    last_m_out = NaturalGaussianTensor(out_h, out_w, f_out)

    return FirstConv2dFactor(m_to_weights, last_W_marginal, new_m_to_weights, m_to_biases, last_bias_marginal, new_m_to_biases, last_x_in, last_m_out, p)
end

function forward_message(node::FirstConv2dFactor, x_in::AbstractArray{FloatType,3}, i_in::Int)
    # Prepare arrays
    node.last_x_in .= x_in
    m_to_weights = selectdim(node.m_to_weights, 5, i_in)
    m_to_biases = selectdim(node.m_to_biases, 2, i_in)

    # Compute message
    forward_conv_factor(node.last_x_in, node.last_W_marginal, m_to_weights, node.last_bias_marginal, m_to_biases, node.last_m_out; padding=node.padding)
    return node.last_m_out
end

function forward_predict(node::FirstConv2dFactor, x_in::AbstractArray{FloatType,4})
    # Compute weight marginal (just to be safe)
    recompute_marginal(node)

    return forward_conv_factor(x_in, node.last_W_marginal, node.last_bias_marginal; padding=node.padding)
end

function backward_message(node::FirstConv2dFactor, m_back::AbstractArray{Gaussian1d,3}, i_in::Int)
    # Product-sum factor backward:
    m_to_weights = selectdim(node.m_to_weights, 5, i_in)
    m_to_biases = selectdim(node.m_to_biases, 2, i_in)
    backward_conv_factor(node.last_x_in, node.last_W_marginal, m_to_weights, node.last_bias_marginal, m_to_biases,
        node.last_m_out, m_back, node.new_m_to_weights, node.new_m_to_biases; padding=node.padding)

    # Update weight marginal and store new messages to weights in the proper place
    update_messages_to_weights(node.last_W_marginal, m_to_weights, node.new_m_to_weights)
    update_messages_to_weights(node.last_bias_marginal, m_to_biases, node.new_m_to_biases)
    return
end


###
### LinearLayer
###
mutable struct GaussianLinearLayerFactor{V_G<:AbstractVector{Gaussian1d},M_G<:AbstractMatrix{Gaussian1d},A_G<:AbstractArray{Gaussian1d,3}} <: WeightFactor
    # Store weights
    m_to_weights::A_G
    last_W_marginal::M_G
    new_m_to_weights::M_G # temp array used in the backward pass

    # Store biases
    m_to_biases::M_G
    last_bias_marginal::V_G
    new_m_to_biases::V_G

    # Messages from the forward pass
    last_in_marginal::V_G
    last_m_out::V_G

    # Messages from the backward pass
    last_m_to_inputs::M_G
end

# Creates LinearLayer (n, m) (plus a bias term) for batch size b
function GaussianLinearLayerFactor(n::Int, m::Int, b::Int, prior_σ2::Union{FloatType,Nothing})
    # Create arrays for weights and biases
    m_to_weights = NaturalGaussianTensor(n, m, b + 1)
    last_W_marginal = NaturalGaussianTensor(n, m)
    new_m_to_weights = similar(last_W_marginal)

    m_to_biases = NaturalGaussianTensor(m, b + 1)
    last_bias_marginal = NaturalGaussianTensor(m)
    new_m_to_biases = NaturalGaussianTensor(m)

    # Generate priors
    m_to_weights[:, :, end] = create_weight_prior(n, m, prior_σ2; post_leaky=true)
    last_W_marginal[:, :] = m_to_weights[:, :, end]
    m_to_biases[:, end] = create_bias_prior(m)
    last_bias_marginal[:] = m_to_biases[:, end]

    # Other arguments
    last_in_marginal = NaturalGaussianTensor(n)
    last_m_out = NaturalGaussianTensor(m)
    last_m_to_inputs = NaturalGaussianTensor(n, b)

    return GaussianLinearLayerFactor(m_to_weights, last_W_marginal, new_m_to_weights, m_to_biases, last_bias_marginal, new_m_to_biases, last_in_marginal, last_m_out, last_m_to_inputs)
end

function forward_message(node::GaussianLinearLayerFactor, m_in::AbstractVector{Gaussian1d}, i_in::Int)
    # Compute input marginals
    m_to_inputs = @view node.last_m_to_inputs[:, i_in]
    @tullio node.last_in_marginal[i] = min_ρ(m_in[i] * m_to_inputs[i], 1e-8)

    # Compute product-sum factor
    m_to_biases = selectdim(node.m_to_biases, 2, i_in)
    forward_mult(node.last_in_marginal, node.last_W_marginal, node.last_bias_marginal, m_to_biases, node.last_m_out)
    return node.last_m_out
end

function forward_predict(node::GaussianLinearLayerFactor, m_in::AbstractMatrix{Gaussian1d})
    # Compute weight marginal (just to be safe)
    recompute_marginal(node)

    # Compute forward prediction and scale variance in-place
    out = forward_mult(transpose_permutedims(m_in), node.last_W_marginal, node.last_bias_marginal)
    return transpose_permutedims(out)
end

function backward_message(node::GaussianLinearLayerFactor, m_back::AbstractVector{Gaussian1d}, i_in::Int; testing::Bool=false)
    # Product-sum factor backward:
    m_to_weights = selectdim(node.m_to_weights, 3, i_in)
    m_to_biases = selectdim(node.m_to_biases, 2, i_in)
    out_inputs = @view node.last_m_to_inputs[:, i_in]
    backward_mult(node.last_in_marginal, node.last_W_marginal, node.last_bias_marginal, m_to_biases,
        node.last_m_out, m_back,
        out_inputs, node.new_m_to_weights, node.new_m_to_biases)


    # Update weight marginal and store new messages to weights in the proper place
    update_messages_to_weights(node.last_W_marginal, m_to_weights, node.new_m_to_weights)
    update_messages_to_weights(node.last_bias_marginal, m_to_biases, node.new_m_to_biases)

    # Make sure there is a min-precision going back
    @tullio out_inputs[i] = min_ρ(out_inputs[i], 1e-8)

    if testing
        # For comparisons between old and new FactorGraph
        return out_inputs, [m_to_weights; reshape(m_to_biases, 1, :)] # TODO remove
    end
    return out_inputs
end

function reset_batch_caches(node::GaussianLinearLayerFactor)
    node.last_m_to_inputs .= Gaussian1d()
end


# In comparison to the GaussianLinearLayerFactor, this one expects an x (one training example) and not a v_in
mutable struct FirstGaussianLinearLayerFactor{V_F<:AbstractVector{FloatType},V_G<:AbstractVector{Gaussian1d},M_G<:AbstractMatrix{Gaussian1d},A_G<:AbstractArray{Gaussian1d,3}} <: WeightFactor
    # Store weights
    m_to_weights::A_G
    last_W_marginal::M_G
    new_m_to_weights::M_G # temp array used in the backward pass

    # Store biases
    m_to_biases::M_G
    last_bias_marginal::V_G
    new_m_to_biases::V_G

    # Messages from the forward pass
    last_x_in::V_F
    last_m_forward::V_G
end

# Creates LinearLayer (n, m) (plus a bias term) for batch size b
function FirstGaussianLinearLayerFactor(n::Int, m::Int, b::Int, prior_σ2::Union{FloatType,Nothing})
    # Create arrays for weights and biases
    m_to_weights = NaturalGaussianTensor(n, m, b + 1)
    last_W_marginal = NaturalGaussianTensor(n, m)
    new_m_to_weights = similar(last_W_marginal)

    m_to_biases = NaturalGaussianTensor(m, b + 1)
    last_bias_marginal = NaturalGaussianTensor(m)
    new_m_to_biases = NaturalGaussianTensor(m)

    # Generate priors
    m_to_weights[:, :, end] = create_weight_prior(n, m, prior_σ2)
    last_W_marginal[:, :] = m_to_weights[:, :, end]
    m_to_biases[:, end] = create_bias_prior(m)
    last_bias_marginal[:] = m_to_biases[:, end]

    # Other arguments
    last_x_in = zeros(FloatType, n)
    last_m_forward = NaturalGaussianTensor(m)

    return FirstGaussianLinearLayerFactor(m_to_weights, last_W_marginal, new_m_to_weights, m_to_biases, last_bias_marginal, new_m_to_biases, last_x_in, last_m_forward)
end

function forward_message(node::FirstGaussianLinearLayerFactor, x_in::AbstractVector{FloatType}, i_in::Int)
    # Prepare arrays
    node.last_x_in .= x_in
    m_to_weights = @view node.m_to_weights[:, :, i_in]
    m_to_biases = selectdim(node.m_to_biases, 2, i_in)

    # Compute forward message
    forward_mult(node.last_x_in, node.last_W_marginal, m_to_weights, node.last_bias_marginal, m_to_biases, node.last_m_forward)
    return node.last_m_forward
end

function forward_predict(node::FirstGaussianLinearLayerFactor, x_in::AbstractMatrix{FloatType})
    # Compute weight marginal (just to be safe)
    recompute_marginal(node)

    # Compute forward sum:
    out = forward_mult(transpose_permutedims(x_in), node.last_W_marginal, node.last_bias_marginal)
    return transpose_permutedims(out)
end

function backward_message(node::FirstGaussianLinearLayerFactor, m_back::AbstractVector{Gaussian1d}, i_in::Int; testing::Bool=false)
    # Compute backward message
    m_to_weights = @view node.m_to_weights[:, :, i_in]
    m_to_biases = selectdim(node.m_to_biases, 2, i_in)
    backward_mult(node.last_x_in, node.last_W_marginal, m_to_weights, node.last_bias_marginal, m_to_biases, node.last_m_forward, m_back, node.new_m_to_weights, node.new_m_to_biases)

    # Update weight marginal and store new messages to weights in the proper place
    update_messages_to_weights(node.last_W_marginal, m_to_weights, node.new_m_to_weights)
    update_messages_to_weights(node.last_bias_marginal, m_to_biases, node.new_m_to_biases)

    if testing
        return [m_to_weights; reshape(m_to_biases, 1, :)] # TODO: remove
    end
    return
end


###
### Activation Functions and other utils
###
mutable struct LeakyReLUFactor{A1_G<:AbstractArray{Gaussian1d},A2_G<:AbstractArray{Gaussian1d}} <: Factor
    leak::FloatType
    last_m_in::A2_G
    last_m_back::A2_G
    temp::A1_G # to avoid allocaton during forward messages
end

function LeakyReLUFactor(size_in::Tuple, b::Int, leak::FloatType)
    last_m_in = NaturalGaussianTensor(size_in..., b)
    last_m_back = NaturalGaussianTensor(size_in..., b)
    temp = NaturalGaussianTensor(size_in...)
    return LeakyReLUFactor(leak, last_m_in, last_m_back, temp)
end

function forward_message(node::LeakyReLUFactor, m_in::AbstractArray{Gaussian1d}, i_in::Int)
    nd = ndims(node.last_m_in)
    selectdim(node.last_m_in, nd, i_in) .= m_in # store input
    m_back = selectdim(node.last_m_back, nd, i_in)
    # TODO: Can we linearize the leaky relu for extreme cases? That should save runtime & be more precise

    @assert size(m_back) == size(m_in)
    out, leak = node.temp, node.leak
    if nd == 2
        @tullio out[i] = forward_leaky_relu_factor(m_in[i], m_back[i], leak)
    elseif nd == 4
        @tullio out[i, j, k] = forward_leaky_relu_factor(m_in[i, j, k], m_back[i, j, k], leak)
    else
        @assert false
    end
    return out
end

function forward_predict(node::LeakyReLUFactor, m_in::AbstractArray{Gaussian1d})
    # Has to use message approximation (instead of marginal approximation)
    return forward_leaky_relu_factor.(m_in, node.leak)
end

function backward_message(node::LeakyReLUFactor, m_back::AbstractArray{Gaussian1d}, i_in::Int)
    nd = ndims(node.last_m_in)
    selectdim(node.last_m_back, nd, i_in) .= m_back
    m_in = selectdim(node.last_m_in, nd, i_in)

    @assert size(m_back) == size(m_in)
    out, leak = node.temp, node.leak
    if nd == 2
        @tullio out[i] = min_ρ(backward_leaky_relu_factor(m_in[i], m_back[i], leak), 1e-8)
    elseif nd == 4
        @tullio out[i, j, k] = min_ρ(backward_leaky_relu_factor(m_in[i, j, k], m_back[i, j, k], leak), 1e-8)
    else
        @assert false
    end
    return out
end

function reset_batch_caches(node::LeakyReLUFactor)
    node.last_m_back .= Gaussian1d()
end


# Currently this factor is only available as a **global** AveragePool. That means it also produces a flattened output.
# It should be simple to adapt AvgPool to work with arbitrary sizes, similar to the implementation of MaxPool
mutable struct AvgPool2dFactor{V_G<:AbstractVector{Gaussian1d},A_G<:AbstractArray{Gaussian1d,3}} <: Factor
    # Messages from the forward pass
    last_m_in::A_G
    last_m_out::V_G

    # Messages from the backward pass
    last_m_to_inputs::A_G
end

# Creates LinearLayer (n, m) (plus a bias term) for batch size b
function AvgPool2dFactor(h::Int, w::Int, f_in::Int)
    last_m_in = NaturalGaussianTensor(h, w, f_in)
    last_m_out = NaturalGaussianTensor(f_in)
    last_m_to_inputs = similar(last_m_in)
    return AvgPool2dFactor(last_m_in, last_m_out, last_m_to_inputs)
end

function forward_message(node::AvgPool2dFactor, m_in::AbstractArray, i_in::Int)
    @tullio node.last_m_in[i] = m_in[i]
    forward_avg_pool_factor(m_in, node.last_m_out)

    @tullio node.last_m_out[k] = min_ρ(node.last_m_out[k], 1e-8)
    return node.last_m_out
end

function forward_predict(node::AvgPool2dFactor, m_in::AbstractArray)
    # Prepare a few arrays
    n_out = size(m_in, 4)
    m_out = similar(node.last_m_out, (size(node.last_m_out)..., n_out))

    # Compute all predictions
    for i in axes(m_in, 4)
        forward_avg_pool_factor(selectdim(m_in, 4, i), selectdim(m_out, 2, i))
    end
    return m_out
end

function backward_message(node::AvgPool2dFactor, m_back::AbstractVector, i_in::Int)
    backward_avg_pool_factor(node.last_m_in, node.last_m_out, m_back, node.last_m_to_inputs)
    assert_well_defined.(node.last_m_to_inputs)

    @tullio node.last_m_to_inputs[i, j, k] = min_ρ(node.last_m_to_inputs[i, j, k], 1e-8)
    return node.last_m_to_inputs
end


mutable struct MaxPool2dFactor{A_G<:AbstractArray{Gaussian1d,3}} <: Factor
    k::Int

    # Messages from the forward pass
    last_m_in::A_G
    last_m_out::A_G

    # Messages from the backward pass
    last_m_to_inputs::A_G
end

# Creates LinearLayer (n, m) (plus a bias term) for batch size b
function MaxPool2dFactor(h_in::Int, w_in::Int, f_in::Int, k::Int)
    # @assert (h_in % k == 0) && (w_in % k == 0)

    last_m_in = NaturalGaussianTensor(h_in, w_in, f_in)
    last_m_out = NaturalGaussianTensor(div(h_in, k), div(w_in, k), f_in)
    last_m_to_inputs = similar(last_m_in)
    return MaxPool2dFactor(k, last_m_in, last_m_out, last_m_to_inputs)
end

function forward_message(node::MaxPool2dFactor, m_in::AbstractArray{Gaussian1d,3}, i_in::Int)
    node.last_m_in .= m_in
    forward_max_pool_factor(m_in, (node.k, node.k), node.last_m_out)
    return node.last_m_out
end

function forward_predict(node::MaxPool2dFactor, m_in::AbstractArray{Gaussian1d,4})
    return forward_max_pool_factor(m_in, (node.k, node.k))
end

function backward_message(node::MaxPool2dFactor, m_back::AbstractArray{Gaussian1d,3}, i_in::Int)
    backward_max_pool_factor(node.last_m_in, (node.k, node.k), node.last_m_out, m_back, node.last_m_to_inputs)
    return node.last_m_to_inputs
end



# Flatten any input to a vector
mutable struct FlattenFactor <: Factor
    input_size::Tuple
end

function FlattenFactor()
    return FlattenFactor(())
end

function forward_message(node::FlattenFactor, m_in::AbstractArray, i_in::Int)
    if length(node.input_size) == 0
        node.input_size = size(m_in)
    end

    @assert size(m_in) == node.input_size
    return reshape(m_in, :)
end

function forward_predict(node::FlattenFactor, m_in::AbstractArray)
    n = size(m_in)[end]
    return copy(reshape(m_in, (:, n)))
end

function backward_message(node::FlattenFactor, y::AbstractVector, i_in::Int)
    @assert length(y) == prod(node.input_size)
    return reshape(y, node.input_size)
end

# If the Flatten is at the beginning, there will be no meaningful backward message
function backward_message(node::FlattenFactor, y::Nothing, i_in::Int)
    return
end


###
### Training Signals
###
mutable struct RegressionFactor{V_G<:AbstractVector{Gaussian1d}} <: Factor
    β2::FloatType # Assumed observation noise
    m_back::V_G # Just for allocation purposes
end

function RegressionFactor(n::Int, β2::FloatType)
    m_back = NaturalGaussianTensor(n)
    return RegressionFactor(β2, m_back)
end

function forward_message(node::RegressionFactor, m_in::AbstractVector{Gaussian1d}, i_in::Int)
    # No-op
    return
end

function forward_predict(node::RegressionFactor, m_in::AbstractMatrix{Gaussian1d})
    β2 = node.β2
    return add_variance.(m_in, β2)
end

function backward_message(node::RegressionFactor, y::AbstractVector{FloatType}, i_in::Int)
    β2 = node.β2
    @tullio node.m_back[i] = Gaussian1d(; μ=y[i], σ2=β2)
    return node.m_back
end


mutable struct SoftmaxFactor{V_G<:AbstractVector{Gaussian1d}} <: Factor
    last_m_in::V_G # One vector per example (the incoming forward message of the pre-activation)
    m_back::V_G # just for allocation purposes
end

function SoftmaxFactor(n::Int)
    last_m_in = NaturalGaussianTensor(n)
    m_back = NaturalGaussianTensor(n)
    return SoftmaxFactor(last_m_in, m_back)
end

function forward_message(node::SoftmaxFactor, m_in::AbstractVector{Gaussian1d}, i_in::Int)
    node.last_m_in[:] = m_in
    return
end

function forward_predict(node::SoftmaxFactor, m_in::AbstractMatrix{Gaussian1d})
    return forward_softmax(m_in)
end

function backward_message(node::SoftmaxFactor, y::Int, i_in::Int)
    @assert 1 <= y <= length(node.m_back)

    backward_softmax_integrals(node.last_m_in, y, node.m_back)
    return node.m_back
end


mutable struct ArgmaxFactor{V_G<:AbstractVector{Gaussian1d},M_G<:AbstractMatrix{Gaussian1d}} <: Factor
    last_m_in::V_G # One vector per example (the incoming forward message of the pre-activation)
    new_m_back::V_G
    m_back::M_G
    regularize::Bool
end

function ArgmaxFactor(n::Int, b::Int; regularize::Bool=false)
    last_m_in = NaturalGaussianTensor(n)
    new_m_back = NaturalGaussianTensor(n)
    m_back = NaturalGaussianTensor(n, b)
    return ArgmaxFactor(last_m_in, new_m_back, m_back, regularize)
end

function forward_message(node::ArgmaxFactor, m_in::AbstractVector{Gaussian1d}, i_in::Int)
    node.last_m_in[:] = m_in
    return
end

function forward_predict(node::ArgmaxFactor, m_in::AbstractMatrix{Gaussian1d})
    return forward_argmax(m_in)
end

function backward_message(node::ArgmaxFactor, y::Int, i_in::Int)
    @assert 1 <= y <= size(node.m_back, 1)

    backward_argmax(node.last_m_in, y, node.new_m_back; regularize=node.regularize)

    new_m_back = node.new_m_back
    m_back = selectdim(node.m_back, 2, i_in)

    @tullio m_back[i] = new_m_back[i]
    # @tullio m_back[i] = EMA(m_back[i], new_m_back[i], ifelse(m_back[i].ρ > 0, 0.8, 1.0)) # EMA
    return m_back
end

function reset_batch_caches(node::ArgmaxFactor)
    node.m_back .= Gaussian1d()
end


###
### Factor Graph
###
mutable struct FactorGraph
    layers::Vector{Factor}
    batch_size::Int
    device_array::UnionAll
end

# Only supports scalar labels for now (i.e., regression problems)
function create_factor_graph(layer_templates::Vector{<:Tuple}, batch_size::Int)
    layers = Vector{Factor}()

    @assert all(isa(i, Int) for i in layer_templates[1]) # Input size hint
    d_current = layer_templates[1]
    first_layer = true

    for (i, e) in enumerate(layer_templates[2:end])
        if e[1] == :Linear
            @assert isa(e[2], Integer) # Output size
            @assert length(d_current) == 1

            prior_σ2 = (length(e) >= 3 && isa(e[3], FloatType)) ? e[3] : nothing

            if first_layer
                first_layer = false
                push!(layers, FirstGaussianLinearLayerFactor(d_current[1], e[2], batch_size, prior_σ2))
            else
                push!(layers, GaussianLinearLayerFactor(d_current[1], e[2], batch_size, prior_σ2))
            end
            # Size of activation vector
            d_current = (e[2],)
        elseif e[1] == :Conv
            @assert isa(e[2], Int) # Num output features
            @assert isa(e[3], Int) # Kernel Size
            @assert length(d_current) == 3

            # Set padding, if supplied
            p = 0
            if (length(e) >= 4) && isa(e[4], Int)
                p = e[4]
            end

            if first_layer
                first_layer = false
                push!(layers, FirstConv2dFactor(d_current..., e[2], e[3], batch_size, p))
            else
                push!(layers, Conv2dFactor(d_current..., e[2], e[3], batch_size, p))
            end
            d_current = (2 * p + d_current[1] - e[3] + 1, 2 * p + d_current[2] - e[3] + 1, e[2])
        elseif e[1] == :Regression
            @assert e == layer_templates[end] # Last Layer
            @assert isa(e[2], FloatType) # β2
            @assert length(d_current) == 1

            push!(layers, RegressionFactor(d_current[1], e[2]))
        elseif e[1] == :Softmax
            @assert e == layer_templates[end] # Last Layer
            @assert length(d_current) == 1

            push!(layers, SoftmaxFactor(d_current[1]))
        elseif e[1] == :Argmax
            @assert e == layer_templates[end] # Last Layer
            @assert length(d_current) == 1

            regularize = (length(e) > 1 && e[2] == true)
            push!(layers, ArgmaxFactor(d_current[1], batch_size; regularize))
        elseif e[1] == :Flatten
            push!(layers, FlattenFactor())

            d_current = (prod(d_current),)
        elseif e[1] == :AvgPool
            @assert length(d_current) == 3

            push!(layers, AvgPool2dFactor(d_current...))
            d_current = (d_current[3],)
        elseif e[1] == :MaxPool
            @assert length(d_current) == 3
            @assert isa(e[2], Int) # MaxPool Size

            push!(layers, MaxPool2dFactor(d_current..., e[2]))
            d_current = (div(d_current[1], e[2]), div(d_current[2], e[2]), d_current[3])
        else
            @assert i > 1
            @assert e[1] == :LeakyReLU
            @assert isa(e[2], FloatType) # leak

            push!(layers, LeakyReLUFactor(d_current, batch_size, e[2]))
        end
    end
    return FactorGraph(layers, batch_size, Array)
end

function Adapt.adapt_structure(to, fg::FactorGraph)
    layers::AbstractVector{Factor} = [adapt(to, layer) for layer in fg.layers]
    return FactorGraph(layers, fg.batch_size, to)
end

function recompute_marginals(fg::FactorGraph)
    if has_cuda_lib() && fg.device_array == CuArray
        for layer in fg.layers
            if isa(layer, WeightFactor)
                recompute_marginal(layer)
            end
        end
    else
        Threads.@threads for layer in fg.layers
            if isa(layer, WeightFactor)
                recompute_marginal(layer)
            end
        end
    end
end

function forward_pass(fg::FactorGraph, x_in::AbstractArray{FloatType}, i_in::Int)
    m_forward = forward_message(fg.layers[1], x_in, i_in)
    for layer in fg.layers[2:end]
        m_forward = forward_message(layer, m_forward, i_in)
    end
    return
end

# Runs batch-wise prediction
function predict(fg::FactorGraph, X_in::AbstractArray{FloatType}; silent=false)
    # Create wrapper for output
    original_type = Base.typename(typeof(X_in)).wrapper
    preds = []

    n = size(X_in, ndims(X_in))
    num_batches = Int64(ceil(n / fg.batch_size))

    for b in optional_progress_bar(1:num_batches, silent)
        i1 = 1 + fg.batch_size * (b - 1)
        i2 = min(n, fg.batch_size * b)

        # Put X_i on GPU, if needed
        X_i = selectdim(X_in, ndims(X_in), i1:i2)

        if fg.device_array != Array
            # GPU Code - copying X_i to GPU unfortunately allocates CPU Memory...
            X_i = fg.device_array(X_i)

            m_forward = forward_predict(fg.layers[1], X_i)
            for layer in fg.layers[2:end]
                m_forward_new = forward_predict(layer, m_forward)

                free_if_CUDA!(m_forward)
                m_forward = m_forward_new
            end
            push!(preds, adapt(original_type, m_forward))

            free_if_CUDA!(X_i)
            free_if_CUDA!(m_forward)
        else
            # CPU Code
            m_forward = forward_predict(fg.layers[1], X_i)
            for layer in fg.layers[2:end]
                m_forward = forward_predict(layer, m_forward)
            end
            push!(preds, m_forward)
        end
    end

    return hcat(preds...)
end

function backward_pass(fg::FactorGraph, y::Union{Int,AbstractVector{FloatType}}, i_in::Int)
    m_back = backward_message(fg.layers[end], y, i_in)
    for layer in reverse(fg.layers[1:end-1])
        m_back = backward_message(layer, m_back, i_in)
    end
    return
end

function train_batch(fg::FactorGraph, X::AbstractArray{FloatType}, Y::Union{AbstractMatrix{FloatType},AbstractVector{Int}}; num_training_its::Int=10, rng::Union{Random.MersenneTwister,Nothing}=nothing)
    nd_X = ndims(X)
    nd_Y = ndims(Y)
    @assert size(X, nd_X) == size(Y, nd_Y)

    # Create shuffled order
    order = 1:size(Y, nd_Y)
    if isnothing(rng)
        order = Random.shuffle(order)
    else
        order = Random.shuffle(rng, order)
    end

    if nd_Y == 1
        # Classification
        for it in 1:num_training_its
            for i in order
                forward_pass(fg, selectdim(X, nd_X, i), i)
                backward_pass(fg, Y[i], i)
            end
            recompute_marginals(fg)
        end
    else
        # Regression
        for it in 1:num_training_its
            for i in order
                forward_pass(fg, selectdim(X, nd_X, i), i)
                backward_pass(fg, selectdim(Y, nd_Y, i), i)
            end
            recompute_marginals(fg)
        end
    end
end


function evaluate(fg::FactorGraph, X::AbstractArray{FloatType}, Y::Union{AbstractMatrix{FloatType},AbstractVector{Int}}; as_classification::Union{Nothing,Bool}=nothing, silent::Bool=false)
    if isa(fg.layers[end], RegressionFactor)
        if isnothing(as_classification) || as_classification == false
            preds = predict(fg, X; silent)
            mse = Statistics.mean((mean.(preds) .- Y) .^ 2)
            ll = sum(logpdf_normal.(Y, preds))

            println("MSE: $mse\nLog Likelihood: $ll")
            return ll
        else
            # Evaluation
            posterior_preds = forward_argmax(predict(fg, X; silent))

            # Now get the maximum for each category
            @tullio preds[j] := argmax(posterior_preds[:, j])
            @tullio labels[j] := argmax(Y[:, j])

            # Test correctness
            num_correct = sum(preds .== labels)
            println("Correct max prediction: $num_correct / $(size(Y,2)) ($(100 * num_correct / size(Y,2)) %)\n")
            return num_correct / size(Y, 2)
        end
    else
        # Always runs as classification
        @assert as_classification != false

        # Compute maximum
        posterior_preds = predict(fg, X; silent)
        @tullio preds[j] := argmax(posterior_preds[:, j])

        # Test correctness
        num_correct = sum(preds .== Y)
        println("Correct max prediction: $num_correct / $(length(Y)) ($(100 * num_correct / length(Y)) %)")
        return num_correct / length(Y)
    end
end


###
### Trainer - allows continued training with a 2nd "train" call
###
mutable struct Trainer{M_G<:AbstractMatrix{Gaussian1d},A_G<:AbstractArray{Gaussian1d},D_X<:AbstractArray{FloatType},D_Y<:Union{Matrix{FloatType},Vector{Int}}}
    fg::FactorGraph

    Ws::Vector{A_G}
    biases::Vector{M_G}
    weight_layers::Vector{WeightFactor}

    X::D_X
    Y::D_Y

    num_batches::Int
    current_epoch::Int

    rng::Random.MersenneTwister # For generating reproducible random data orders without influencing the global rng
end

function Trainer(fg::FactorGraph, X::AbstractArray{FloatType}, Y::Union{Matrix{FloatType},Vector{Int}})
    n = size(X, ndims(X))
    num_batches = Int64(ceil(n / fg.batch_size))

    # Store messages to weights from batches
    Ws = []
    biases = []
    weight_layers = Vector{WeightFactor}()

    for layer in fg.layers
        if isa(layer, WeightFactor)
            W_i = adapt(fg.device_array, NaturalGaussianTensor(size(layer.last_W_marginal)..., 1 + num_batches))
            bias_i = adapt(fg.device_array, NaturalGaussianTensor(size(layer.last_bias_marginal)..., 1 + num_batches))

            # Store the weight prior in the last index
            nd_Wi, nd_tW = ndims(W_i), ndims(layer.m_to_weights)
            selectdim(W_i, nd_Wi, size(W_i, nd_Wi)) .= selectdim(layer.m_to_weights, nd_tW, size(layer.m_to_weights, nd_tW))

            # Store the bias prior
            nd_bi, nd_tb = ndims(bias_i), ndims(layer.m_to_biases)
            selectdim(bias_i, nd_bi, size(bias_i, nd_bi)) .= selectdim(layer.m_to_biases, nd_tb, size(layer.m_to_biases, nd_tb))

            push!(Ws, W_i)
            push!(biases, bias_i)
            push!(weight_layers, layer)
        end
    end
    return Trainer(fg, [Ws...], [biases...], weight_layers, X, Y, num_batches, 0, Random.MersenneTwister(42))
end

# Run batched message passing: Iterate one batch, then keep only their joint likelihood and throw away individual messages.
function train(trainer::Trainer; num_epochs::Int=1, num_training_its::Int=2, silent=false)
    for _ in 1:num_epochs
        trainer.current_epoch += 1
        β_EMA = 0.9 + 0.1 * exp(-0.3 * (trainer.current_epoch - 1))
        if !silent
            println("--- Epoch $(trainer.current_epoch) --- ($(num_training_its) its, β=$(round(β_EMA, digits=2)))")
        end

        for b in optional_progress_bar(Random.shuffle(trainer.rng, 1:trainer.num_batches), silent)
            i1 = 1 + trainer.fg.batch_size * (b - 1)
            i2 = min(size(trainer.X, ndims(trainer.X)), trainer.fg.batch_size * b)

            # Set new priors
            for l_i in eachindex(trainer.Ws)
                W, bias, layer = trainer.Ws[l_i], trainer.biases[l_i], trainer.weight_layers[l_i]
                nd = ndims(W)
                nb = ndims(bias)

                # Product of all batch messages
                prior_W_ = prod(W, dims=nd)
                prior_W = selectdim(prior_W_, nd, 1)
                prior_bias_ = prod(bias, dims=nb)
                prior_bias = selectdim(prior_bias_, nb, 1)

                # Divide current batch out
                prior_W ./= selectdim(W, nd, b)
                prior_bias ./= selectdim(bias, nb, b)
                reset_for_batch(layer, prior_W, prior_bias)

                free_if_CUDA!.((prior_W_, prior_bias))
            end

            # Reset layer, if needed
            reset_batch_caches.(trainer.fg.layers)

            # Iterate the factor graph
            X_b = selectdim(trainer.X, ndims(trainer.X), i1:i2)
            Y_b = selectdim(trainer.Y, ndims(trainer.Y), i1:i2) # Regression / classification labels

            if trainer.fg.device_array != Array
                if ndims(Y_b) > 1
                    # Put X_b and Y_b on GPU
                    cu_X_b = trainer.fg.device_array(X_b)
                    cu_Y_b = trainer.fg.device_array(Y_b)

                    train_batch(trainer.fg, cu_X_b, cu_Y_b; num_training_its)
                    free_if_CUDA!(cu_X_b)
                    free_if_CUDA!(cu_Y_b)
                else
                    # For classification, the labels cannot go on GPU (because we are doing element-wise access with them)
                    cu_X_b = trainer.fg.device_array(X_b)

                    train_batch(trainer.fg, cu_X_b, Y_b; num_training_its)
                    free_if_CUDA!(cu_X_b)
                end
            else
                train_batch(trainer.fg, X_b, Y_b; num_training_its)
            end

            # Store weight results
            for l_i in eachindex(trainer.Ws)
                W, bias, layer = trainer.Ws[l_i], trainer.biases[l_i], trainer.weight_layers[l_i]
                store_likelihood_messages(layer, selectdim(W, ndims(W), b), selectdim(bias, ndims(bias), b); β_EMA)
            end
        end
    end
end

# Shortcut method that doesn't require an extra call to create a trainer
function train(fg::FactorGraph, X::AbstractArray{FloatType}, Y::Union{Matrix{FloatType},Vector{Int}}; num_epochs::Int=1, num_training_its::Int=10)
    trainer = Trainer(fg, X, Y)
    train(trainer; num_epochs, num_training_its)
end




###
### The below is some testing code that can be used for performance testing of individual layers.
### There also used to be correctness tests that compare the new implementation to the old, but by now the old implementation has been removed.
###

# Random.seed!(98)
# d_in, d_out, num_b = 784, 1000, 320
# a = randn(Float64, d_in)
# a_b = randn(Float64, d_in, num_b)
# b = abs.(randn(Float64, d_in))
# c = randn(Float64, d_out,)
# d = abs.(randn(Float64, d_out))

# e = randn(Float64, d_in)
# f = abs.(randn(Float64, d_in))
# g = randn(Float64, 10)
# g_b = randn(Float64, 10, num_b)
# h = randn(Float64, (d_in, num_b))
# i = randn(Float64, (26, 26, 100))
# j = abs.(randn(Float64, (26, 26, 100)))

# m_forward = Gaussian1d.(a, b);
# m_back = Gaussian1d.(c, d);
# m_back_leaky_relu = Gaussian1d.(e, f);
# m_back_conv = Gaussian1d.(i, j);


# ###
# ### Test GaussianLinearLayer
# ###
# l = GaussianLinearLayerFactor(d_in, d_out, num_b, nothing)
# prior_l = [l.m_to_weights[:, :, end]; reshape(l.m_to_biases[:, end], 1, :)]

# v_in = create_variables(d_in)
# send_message.(m_forward, add_factor.(v_in)) # forward message
# v_out = create_variables(d_out)
# W, prior_factor = create_weights(d_in, d_out, 1.0)
# send_message.(prior_l, prior_factor)
# l_old = GaussianLinearLayerFactor(; W=add_factor.(W), β=1.0, v_in=add_factor.(v_in), v_out=add_factor.(v_out))

# f1 = copy(forward_message(l, m_forward, 1))
# b1, w1 = backward_message(l, m_back, 1, testing=true)
# f1_1 = forward_message(l, m_forward, 1)

# f2 = forward_message(l_old)
# send_message.(m_back, add_factor.(v_out)) # send backward message
# b2, w2 = backward_message(l_old)
# f2_1 = forward_message(l_old)

# println("Forward GaussianLinearLayer: $(maximum(maxdiff.(f1, f2)))")
# println("Backward GaussianLinearLayer (inputs): $(maximum(maxdiff.(b1, b2)))")
# println("Backward GaussianLinearLayer (weights): $(maximum(maxdiff.(w1, w2)))")
# println("Forward2 GaussianLinearLayer: $(maximum(maxdiff.(f1_1, f2_1)))")

# ###
# ### Test FirstGaussianLinearLayer
# ###
# l2 = FirstGaussianLinearLayerFactor(d_in, d_out, num_b, nothing)
# prior_l2 = [l2.m_to_weights[:, :, end]; reshape(l2.m_to_biases[:, end], 1, :)]

# v = create_variables(d_out)
# v_factor = add_factor.(v)
# send_message.(m_back, v_factor) # send backward message
# W, prior_factor = create_weights(d_in, d_out, 1.0)
# send_message.(prior_l2, prior_factor)
# l2_old = FirstGaussianLinearLayerFactor(; W=add_factor.(W), x=a[:, 1], β=1.0, v_out=add_factor.(v))

# f1 = forward_message(l2, a[:, 1], 1)
# b1 = backward_message(l2, m_back[:, 1], 1; testing=true)

# f2 = forward_message(l2_old)
# b2 = backward_message(l2_old)
# println("\nForward FirstGaussianLinearLayer: $(maximum(maxdiff.(f1, f2)))")
# println("Backward FirstGaussianLinearLayer: $(maximum(maxdiff.(b1, b2)))")

# ###
# ### Test LeakyReLU
# ###
# l3 = LeakyReLUFactor((d_in,), d_in, 0.01)

# v_in = create_variables(d_in)
# send_message.(m_forward, add_factor.(v_in)) # forward message
# v_out = create_variables(d_in)
# l3_old = LeakyReLUFactor(0.01, add_factor.(v_in), add_factor.(v_out))

# f1 = copy(forward_message(l3, m_forward, 1))
# b1 = copy(backward_message(l3, m_back_leaky_relu, 1))
# f1_1 = copy(forward_message(l3, m_forward, 1))

# f2 = forward_message(l3_old)
# send_message.(m_back_leaky_relu, add_factor.(v_out)) # send backward message
# b2 = backward_message(l3_old)
# f2_1 = forward_message(l3_old)

# stack([b1[[g.ρ for g in b1].!=[g.ρ for g in b2]][:, 1], b2[[g.ρ for g in b1][:, 1].!=[g.ρ for g in b2]]])

# println("\nForward LeakyReLU: $(maximum(maxdiff.(f1, f2)))")
# println("Backward LeakyReLU: $(maximum(maxdiff.(b1, b2)))")
# println("Forward2 LeakyReLU: $(maximum(maxdiff.(f1_1, f2_1)))")
# ;

# ###
# ### Loss Functions
# ###
# l4 = RegressionFactor(d_in, 0.05^2)
# l5 = SoftmaxFactor(d_in)

# forward_predict(l4, stack([m_forward, m_forward]))
# forward_predict(l5, stack([m_forward, m_forward]))
# backward_message(l4, h[:, 1], 1)

# @tullio h2[j] := argmax((@view h[:, j]))
# forward_message(l5, m_forward, 1)
# backward_message(l5, h2[1], 1);


# ###
# ### Convolutions
# ###
# l6 = Conv2dFactor(28, 28, 1, 100, 3, num_b);
# forward_message(l6, reshape(m_forward, 28, 28, 1), 1);
# forward_predict(l6, reshape(m_forward, 28, 28, 1, 1));

# forward_message(l6, reshape(m_forward, 28, 28, 1), 1);
# backward_message(l6, m_back_conv, 1);

# l7 = FirstConv2dFactor(28, 28, 1, 100, 3, num_b);
# forward_message(l7, reshape(a, 28, 28, 1), 1);
# forward_predict(l7, reshape(a, 28, 28, 1, 1));

# forward_message(l7, reshape(a, 28, 28, 1), 1);
# backward_message(l7, m_back_conv, 1);

# function forward_backward(layer::Factor, m_forward::AbstractArray, m_backward::AbstractArray{Gaussian1d})
#     forward_message(layer, m_forward, 1)
#     backward_message(layer, m_backward, 1)
#     return
# end



# ###
# ### Performance measurements
# ###
# # Product Factor
# @btime forward_message($l, $m_forward, 1); # 793µs, 10KiB
# @btime backward_message($l, $m_back, 1); # 1.3ms, 11KiB

# # Sum Factor
# @btime forward_message($l2, $a, 1); # 204µs, 7KiB
# @btime backward_message($l2, $m_back, 1); # 519µs, 6KiB

# # LeakyReLU
# @btime forward_message($l3, $m_forward, 1); # 191µs, 12KiB
# @btime backward_message($l3, $m_back_leaky_relu, 1); # 193µs, 12KiB

# # Regression / Softmax
# forward_message(l5, m_forward, 1);
# @btime backward_message($l4, $h[:, 1], 1); # 5µs, 6KiB
# @btime backward_message($l5, $h2[1], 1); # 3.8ms, 2MiB

# # Conv Factor
# @btime forward_backward($l6, $(reshape(m_forward, 28, 28, 1)), $m_back_conv);
# @btime forward_backward($l7, $(reshape(a, 28, 28, 1)), $m_back_conv);
# @assert 1 == 1


# ###
# ### Performance Measurement with CUDA
# ###
# cu_l = adapt(CuArray, l)
# cu_l2 = adapt(CuArray, l2)
# cu_l3 = adapt(CuArray, l3)
# cu_l4 = adapt(CuArray, l4)
# cu_l5 = adapt(CuArray, l5)
# cu_l6 = adapt(CuArray, l6)
# cu_l7 = adapt(CuArray, l7)

# cu_m_forward = CuArray(m_forward)
# cu_m_back = CuArray(m_back)
# cu_a = CuArray(a)
# cu_a_b = CuArray(a_b)
# cu_g = CuArray(g)
# cu_g_b = CuArray(g_b)
# cu_m_back_leaky_relu = CuArray(m_back_leaky_relu)
# cu_m_back_conv = CuArray(m_back_conv)
# cu_h = CuArray(h)
# cu_h2 = CuArray(h2);

# println("\nCUDA Predictions")
# @btime CUDA.@sync forward_predict($cu_l, $reshape(cu_m_forward, :, 1)); # 1ms
# @btime CUDA.@sync forward_predict($cu_l2, $reshape(cu_a, :, 1));
# @btime CUDA.@sync forward_predict($cu_l3, $reshape(cu_m_forward, :, 1)); # 77µs
# @btime CUDA.@sync forward_predict($cu_l4, $reshape(cu_m_forward, :, 1)); # 18µs
# @btime CUDA.@sync forward_predict($cu_l5, $reshape(cu_m_forward, :, 1)); # 


# println("\nCUDA Measurements")
# # Product Factor
# @btime CUDA.@sync forward_message($cu_l, $cu_m_forward, 1); # 83µs, 19KiB
# @btime CUDA.@sync backward_message($cu_l, $cu_m_back, 1); # 120µs, 18KiB

# # Sum Factor
# @btime CUDA.@sync forward_message($cu_l2, $cu_a, 1); # 67µs, 13KiB
# @btime CUDA.@sync backward_message($cu_l2, $cu_m_back, 1); # 60µs, 6KiB

# # LeakyReLU
# @btime CUDA.@sync forward_message($cu_l3, $cu_m_forward, 1); # 134µs, 3KiB
# @btime CUDA.@sync backward_message($cu_l3, $cu_m_back_leaky_relu, 1); # 117µs, 3KiB

# # Regression / Softmax
# forward_message(cu_l5, cu_m_forward, 1);

# @btime CUDA.@sync backward_message($cu_l4, $cu_h[:, 1], 1); # 27µs, 4KiB
# CUDA.@allowscalar h2_1 = cu_h2[1];
# @btime CUDA.@sync backward_message($cu_l5, $h2_1, 1); # 3ms, 930KiB

# # Conv Factor - have to benchmark alternatingly to avoid correctness issues
# @btime CUDA.@sync forward_backward($cu_l6, $(reshape(cu_m_forward, 28, 28, 1)), $cu_m_back_conv);
# @btime CUDA.@sync forward_backward($cu_l7, $(reshape(cu_a, 28, 28, 1)), $cu_m_back_conv);
# @assert 1 == 1



# β2 = 0.05^2
# fg = create_factor_graph([
#         (d_in,),
#         (:Linear, d_out),
#         (:LeakyReLU, 0.01),
#         (:Linear, 10),
#         (:Regression, β2)
#     ],
#     num_b)

# forward_pass(fg, a, 1)
# backward_pass(fg, g, 1);

# fg2 = adapt(CuArray, fg);
# trainer1 = Trainer(fg, a_b, g_b)
# trainer2 = Trainer(fg2, a_b, g_b);

# println("\nNew Factor Graph:")
# @btime forward_pass($fg, $a, 1); # 760µs
# @btime backward_pass($fg, $g, 1); # 6.2ms

# @btime CUDA.@sync forward_pass($fg2, $cu_a, 1); # 460µs
# @btime CUDA.@sync backward_pass($fg2, $cu_g, 1); # 710µs

# @btime train_batch($fg, $a_b, $g_b; num_training_its=2); # 8.6s
# @btime CUDA.@sync train_batch($fg2, $cu_a_b, $cu_g_b; num_training_its=2); # 650ms

# @btime train($trainer1; num_epochs=1, num_training_its=2); # 9.4s
# @btime CUDA.@sync train($trainer2; num_epochs=1, num_training_its=2); # 9.4s

# predict(fg, a)
# ;
# train(fg, a, g)
# ;


# β2 = 0.05^2
# fg = create_factor_graph([(d_in, d_out, 0.01),
#         (:LeakyReLU, 0.01),
#         (d_out, 10, 0.01)],
#     num_b, β2)

# # predict(fg, a)
# a2, g2 = reshape(a, :, 1), reshape(g, :, 1)
# forward_pass(fg, a2)
# backward_pass(fg, g2);

# println("\nOld Factor Graph:")
# @btime forward_pass($fg, $a2)
# @btime backward_pass($fg, $g2)



# # Tullio Example 1:
# @btime x1 = prod(m_forward, dims=2)[:, 1] # 345ms
# @btime @tullio (*) x2[i] := m_forward[i, j] # 80ms
# @assert all(x1 .== x2)
# ;
