include("./gaussian.jl")
include("./messages_gaussian_mult.jl")
using Tullio
import NNlib
using QuadGK


###
### Leaky ReLU Factor
###
moments_relu_factor(m_in::Gaussian1d) = moments_relu_factor(mean(m_in), variance(m_in))
function moments_relu_factor(μ::FloatType, σ2::FloatType)
    σ = sqrt(σ2)
    m2_in = σ2 + μ^2
    x = μ / σ

    pdf_x = pdf_normal(x)
    cdf_x = cdf_normal(x)

    new_μ = σ * pdf_x + μ * cdf_x
    new_m2 = (σ * μ * pdf_x) + (m2_in * cdf_x)
    return new_μ, new_m2
end

function forward_leaky_relu_factor(m_in::Gaussian1d, leak::FloatType)
    relu_μ, relu_m2 = moments_relu_factor(m_in)

    new_μ = (1 - leak) * relu_μ + leak * mean(m_in)
    new_m2 = (1 - leak^2) * relu_m2 + (leak^2) * m2(m_in)

    return Gaussian1d_m2(; μ=new_μ, m2=new_m2)
end

function backward_leaky_relu_factor(m_back::Gaussian1d, leak::FloatType)
    μ_r1, m2_r1 = moments_relu_factor(m_back / (-leak))
    μ_r2, m2_r2 = moments_relu_factor(m_back)
    x = m_back.τ * sqrt(variance(m_back))

    m0 = (1 / leak) * cdf_normal(-x) + cdf_normal(x)
    m1 = (-1 / leak) * μ_r1 + μ_r2
    m2 = (1 / leak) * m2_r1 + m2_r2

    return Gaussian1d_m2(; μ=m1 / m0, m2=m2 / m0)
end

function forward_leaky_relu_factor(m_forward::Gaussian1d, m_back::Gaussian1d, leak::FloatType)
    # a > 0
    m0, m1, m2 = block52(m_forward, m_back)

    # a < 0
    # TODO: On page 39 of the report, there is an 1/leak factor. Adding it makes the results worse. Is the report wrong?
    m0_neg, m1_neg, m2_neg = block52(m_back * -1.0, m_forward * (-leak))
    m0 += m0_neg
    m1 -= m1_neg
    m2 += m2_neg

    # Compute result
    marginal_σ2 = (m2 / m0) - (m1 / m0)^2
    ρ = 1 / marginal_σ2
    τ = (m1 / m0) * ρ

    if (m0 < 1e-8) || isnan(ρ) || isnan(τ) || (ρ < m_forward.ρ + m_back.ρ)
        return forward_leaky_relu_factor(m_forward, leak)
    end

    return Gaussian1d(τ, ρ) / m_back
end

function backward_leaky_relu_factor(m_forward::Gaussian1d, m_back::Gaussian1d, leak::FloatType)
    # z > 0
    m0, m1, m2 = block52(m_forward, m_back)

    # z < 0
    m0_neg, m1_neg, m2_neg = block52(m_forward * -1.0, m_back / (-leak))
    m0 += m0_neg / leak
    m1 -= m1_neg / leak
    m2 += m2_neg / leak

    # Compute result
    marginal_σ2 = (m2 / m0) - (m1 / m0)^2
    ρ = max(m_forward.ρ + 1e-8, 1 / marginal_σ2)
    τ = (m1 / m0) * ρ

    if isinf(ρ) || isinf(τ) || isnan(ρ) || isnan(τ) || ((m_back.τ < 0) && (m_back.ρ < 2e-8))
        return Gaussian1d(0.0, 0.0)
    end

    return Gaussian1d(τ, ρ) / m_forward
end

function block52(g1::Gaussian1d, g2::Gaussian1d)
    μ_1, σ2_1 = mean(g1), variance(g1)
    μ_2, σ2_2 = mean(g2), variance(g2)

    marginal_μ = multiplication_mean(g1, g2)
    marginal_σ2 = multiplication_variance(g1, g2)
    m1_relu, m2_relu = moments_relu_factor(marginal_μ, marginal_σ2)

    norm = pdf_normal(μ_1, μ_2, σ2_1 + σ2_2)
    m0_out = norm * cdf_normal(marginal_μ / sqrt(marginal_σ2))
    m1_out = norm * m1_relu
    m2_out = norm * m2_relu
    return m0_out, m1_out, m2_out
end

###
### Convolutional Factor
###
function unfold_input(in::AbstractArray, kernel::AbstractArray{Gaussian1d,4}; padding::Int=0)
    k_h, k_w, k_f, k_f_out = size(kernel)

    if ndims(in) == 3
        padded_in = NNlib.pad_symmetric(in, (padding, padding, padding, padding, 0, 0))
        in_h, in_w, in_f = size(padded_in)
        in_dims = (in_h, in_w, in_f, 1, 1)
        cdims = NNlib.DenseConvDims(in_dims, (k_h, k_w, k_f, 1, k_f_out))

        col = similar(padded_in, NNlib.im2col_dims(cdims)[1:(end-1)])
        col_b = reshape(col, size(col)..., 1) # add a singleton dim at the end
        NNlib.unfold!(col_b, reshape(padded_in, in_dims), cdims)

        free_if_CUDA!(padded_in)
        return col, cdims
    elseif ndims(in) == 4
        padded_in = NNlib.pad_symmetric(in, (padding, padding, padding, padding, 0, 0, 0, 0))
        in_h, in_w, in_f, b = size(padded_in)
        in_dims = (in_h, in_w, in_f, 1, b)
        cdims = NNlib.DenseConvDims(in_dims, (k_h, k_w, k_f, 1, k_f_out))

        col = similar(padded_in, (NNlib.im2col_dims(cdims)[1:(end-1)]..., b))
        NNlib.unfold!(col, reshape(padded_in, in_dims), cdims)

        free_if_CUDA!(padded_in)
        return col, cdims
    else
        @assert false
    end
end

function unfold_input_pool(in::AbstractArray, pool_size::Tuple{Int,Int})
    k_h, k_w = pool_size

    # Pretend it's a batched operation with only one input and output feature
    if ndims(in) == 3
        in_h, in_w, in_f = size(in)
        in_dims = (in_h, in_w, 1, 1, in_f)
        cdims = NNlib.DenseConvDims(in_dims, (k_h, k_w, 1, 1, 1), stride=(k_h, k_w, 1))

        col = similar(in, (NNlib.im2col_dims(cdims)[1:(end-1)]..., in_f))
        NNlib.unfold!(col, reshape(in, in_dims), cdims)
        return col, cdims
    elseif ndims(in) == 4
        in_h, in_w, in_f, b = size(in)
        in_dims = (in_h, in_w, 1, 1, in_f * b)
        cdims = NNlib.DenseConvDims(in_dims, (k_h, k_w, 1, 1, 1), stride=(k_h, k_w, 1))

        col = similar(in, (NNlib.im2col_dims(cdims)[1:(end-1)]..., in_f * b))
        NNlib.unfold!(col, reshape(in, in_dims), cdims)
        return col, cdims
    else
    end
end

function forward_conv_factor(x_in::AbstractArray{T,4}, kernel::AbstractArray{Gaussian1d,4}, bias::AbstractVector{Gaussian1d}; padding::Int=0) where {T<:Union{FloatType,Gaussian1d}}
    f_h, f_w, f_f, b = size(x_in)
    k_h, k_w, k_f, k_f_out = size(kernel)

    @assert length(bias) == k_f_out
    @assert f_h >= k_h && f_w >= k_w && f_f == k_f

    col, _ = unfold_input(x_in, kernel; padding)
    out = forward_mult_batched(col, reshape2d(kernel), bias)

    free_if_CUDA!(col)
    return reshape(out, (2 * padding + f_h - k_h + 1, 2 * padding + f_w - k_w + 1, k_f_out, b))
end

function forward_conv_factor(x_in::AbstractArray{Gaussian1d,3}, kernel::AbstractArray{Gaussian1d,4}, marginal_bias::AbstractVector{Gaussian1d}, m_to_bias::AbstractVector{Gaussian1d}, out::AbstractArray{Gaussian1d,3}; padding::Int=0)
    f_h, f_w, f_f = size(x_in)
    k_h, k_w, k_f, k_f_out = size(kernel)
    o_h, o_w, o_f_out = size(out)

    @assert length(marginal_bias) == length(m_to_bias) == k_f_out
    @assert f_h >= k_h && f_w >= k_w && f_f == k_f
    @assert o_h == (2 * padding + f_h - k_h + 1) && o_w == (2 * padding + f_w - k_w + 1) && o_f_out == k_f_out

    col, _ = unfold_input(x_in, kernel; padding)
    forward_mult(col, reshape2d(kernel), marginal_bias, m_to_bias, reshape2d(out))
    free_if_CUDA!(col)
    return
end

function forward_conv_factor(x_in::AbstractArray{FloatType,3}, marginal_kernel::AbstractArray{Gaussian1d,4}, m_to_kernel::AbstractArray{Gaussian1d,4}, marginal_bias::AbstractVector{Gaussian1d}, m_to_bias::AbstractVector{Gaussian1d}, out::AbstractArray{Gaussian1d,3}; padding::Int=0)
    f_h, f_w, f_f = size(x_in)
    k_h, k_w, k_f, k_f_out = size(marginal_kernel)
    o_h, o_w, o_f_out = size(out)

    @assert size(marginal_kernel) == size(m_to_kernel)
    @assert f_h >= k_h && f_w >= k_w && f_f == k_f
    @assert o_h == (2 * padding + f_h - k_h + 1) && o_w == (2 * padding + f_w - k_w + 1) && o_f_out == k_f_out
    @assert length(marginal_bias) == length(m_to_bias) == k_f_out

    col, _ = unfold_input(x_in, marginal_kernel; padding)
    forward_mult(col, reshape2d(marginal_kernel), reshape2d(m_to_kernel), marginal_bias, m_to_bias, reshape2d(out))
    free_if_CUDA!(col)
    return
end

# Input: Messages
function backward_conv_factor(m_in::AbstractArray{Gaussian1d,3}, kernel::AbstractArray{Gaussian1d,4}, marginal_bias::AbstractVector{Gaussian1d}, m_to_bias::AbstractVector{Gaussian1d},
    m_pred::AbstractArray{Gaussian1d,3}, m_back::AbstractArray{Gaussian1d,3},
    out_back::AbstractArray{Gaussian1d,3}, out_kernel::AbstractArray{Gaussian1d,4}, out_bias::AbstractVector{Gaussian1d}; padding::Int=0)
    f_h, f_w, f_f = size(m_in)
    k_h, k_w, k_f, k_f_out = size(kernel)
    p_h, p_w, p_f_out = size(m_pred)

    @assert f_h >= k_h && f_w >= k_w && f_f == k_f
    @assert p_h == (2 * padding + f_h - k_h + 1) && p_w == (2 * padding + f_w - k_w + 1) && p_f_out == k_f_out
    @assert size(m_in) == size(out_back)
    @assert size(kernel) == size(out_kernel)
    @assert size(m_pred) == size(m_back)
    @assert isodd(k_h) && isodd(k_w)
    @assert length(marginal_bias) == length(m_to_bias) == length(out_bias) == k_f_out

    # Backward
    col, cdims = unfold_input(m_in, kernel; padding)
    out_col = similar(col)
    backward_mult(col, reshape2d(kernel), marginal_bias, m_to_bias,
        reshape2d(m_pred), reshape2d(m_back),
        out_col, reshape2d(out_kernel), out_bias)


    # Aggregate the backward messages to inputs
    # Because NNlib.fold! is imprecise, the compuation is run twice and the mean is taken
    # ...ρ
    @tullio temp[i, j] := out_col[i, j].ρ
    reshaped_temp = reshape(temp, size(temp)..., 1)

    ρ_back1 = similar(temp, size(m_in))
    ρ_back2 = similar(temp, size(m_in))
    NNlib.fold!(reshape(ρ_back1, size(ρ_back1)..., 1, 1), reshaped_temp, cdims)
    NNlib.fold!(reshape(ρ_back2, size(ρ_back2)..., 1, 1), reshaped_temp, cdims)

    # ...τ
    @tullio temp[i, j] = out_col[i, j].τ

    τ_back1 = similar(temp, size(m_in))
    τ_back2 = similar(temp, size(m_in))
    NNlib.fold!(reshape(τ_back1, size(τ_back1)..., 1, 1), reshaped_temp, cdims)
    NNlib.fold!(reshape(τ_back2, size(τ_back2)..., 1, 1), reshaped_temp, cdims)


    # ...results
    @assert size(ρ_back1) == size(τ_back2) == size(out_back)
    @tullio out_back[i, j, k] = Gaussian1d(
        τ_back2[i, j, k] + (τ_back1[i, j, k] - τ_back2[i, j, k]) / 2,
        ρ_back2[i, j, k] + (ρ_back1[i, j, k] - ρ_back2[i, j, k]) / 2
    )

    free_if_CUDA!.((col, out_col, temp, ρ_back1, ρ_back2, τ_back1, τ_back2))
    return
end

# Input: Scalars
function backward_conv_factor(x_in::AbstractArray{FloatType,3}, marginal_kernel::AbstractArray{Gaussian1d,4}, m_to_kernel::AbstractArray{Gaussian1d,4}, marginal_bias::AbstractVector{Gaussian1d}, m_to_bias::AbstractVector{Gaussian1d}, m_pred::AbstractArray{Gaussian1d,3}, m_back::AbstractArray{Gaussian1d,3}, out_kernel::AbstractArray{Gaussian1d,4}, out_bias::AbstractVector{Gaussian1d}; padding::Int=0)
    f_h, f_w, f_f = size(x_in)
    k_h, k_w, k_f, k_f_out = size(marginal_kernel)
    p_h, p_w, p_f_out = size(m_pred)

    @assert f_h >= k_h && f_w >= k_w && f_f == k_f
    @assert p_h == (2 * padding + f_h - k_h + 1) && p_w == (2 * padding + f_w - k_w + 1) && p_f_out == k_f_out
    @assert size(marginal_kernel) == size(out_kernel)
    @assert isodd(k_h) && isodd(k_w)
    @assert size(marginal_kernel) == size(m_to_kernel)

    # Backward
    col, _ = unfold_input(x_in, marginal_kernel; padding)
    backward_mult(col,
        reshape2d(marginal_kernel), reshape2d(m_to_kernel),
        marginal_bias, m_to_bias,
        reshape2d(m_pred), reshape2d(m_back),
        reshape2d(out_kernel), out_bias)
    free_if_CUDA!(col)
    return
end


###
### MaxPool - always applies stride=pool_size
###
# Batched operation (used for predict)
function forward_max_pool_factor(m_in::AbstractArray{Gaussian1d,4}, size_pool::Tuple{Int,Int})
    f_h, f_w, f_f, b = size(m_in)
    k_h, k_w = size_pool

    # @assert (f_h % k_h == 0) && (f_w % k_w == 0)
    o_h = div(f_h, k_h)
    o_w = div(f_w, k_w)
    o_f_out = f_f

    # Flatten input into (kernel_size_flat, :)
    col_nonflat, _ = unfold_input_pool(m_in, size_pool)
    col_permuted = permutedims(col_nonflat, (2, 1, 3))
    col = reshape(col_permuted, size(col_permuted, 1), :)
    max_weights = forward_argmax(col)

    # Compute output and reshape to correct size
    out = forward_mult_batched(max_weights, col)

    free_if_CUDA!.((col_nonflat, col_permuted, max_weights))
    return reshape(out, (o_h, o_w, o_f_out, b))
end

function forward_max_pool_factor(m_in::AbstractArray{Gaussian1d,3}, size_pool::Tuple{Int,Int}, out::AbstractArray{Gaussian1d,3})
    f_h, f_w, f_f = size(m_in)
    k_h, k_w = size_pool
    o_h, o_w, o_f_out = size(out)

    # @assert (f_h % k_h == 0) && (f_w % k_w == 0)
    @assert o_h == div(f_h, k_h) && o_w == div(f_w, k_w) && o_f_out == f_f

    # Flatten input into (kernel_size_flat, :)
    col_nonflat, _ = unfold_input_pool(m_in, size_pool)
    col_permuted = permutedims(col_nonflat, (2, 1, 3))
    col = reshape(col_permuted, size(col_permuted, 1), :)
    max_weights = forward_argmax(col)

    forward_mult_batched(max_weights, col; out=reshape(out, :))

    free_if_CUDA!.((col_nonflat, col_permuted, max_weights))
    return
end

function backward_max_pool_factor(m_in::AbstractArray{Gaussian1d,3}, size_pool::Tuple{Int,Int}, m_pred::AbstractArray{Gaussian1d,3}, m_back::AbstractArray{Gaussian1d,3}, out_back::AbstractArray{Gaussian1d,3})
    f_h, f_w, f_f = size(m_in)
    k_h, k_w = size_pool
    o_h, o_w, o_f_out = size(m_pred)

    # @assert (f_h % k_h == 0) && (f_w % k_w == 0)
    @assert o_h == div(f_h, k_h) && o_w == div(f_w, k_w) && o_f_out == f_f
    @assert size(m_back) == size(m_pred)
    @assert size(out_back) == size(m_in)

    # Flatten input into (kernel_size_flat, :)
    # TODO: Perhaps it is better to store col instead of m_in in the layer!
    col_nonflat, cdims = unfold_input_pool(m_in, size_pool)
    col_permuted = permutedims(col_nonflat, (2, 1, 3))
    col = reshape(col_permuted, size(col_permuted, 1), :)
    max_weights = forward_argmax(col)

    out_col = similar(col_permuted)
    backward_mult_batched(max_weights, col,
        reshape(m_pred, :), reshape(m_back, :),
        reshape(out_col, size(col_permuted, 1), :))


    # Reverse the reshape and permutedims
    out_unfolded = col_nonflat # renaming. TODO: Test if it is really only a reference
    permutedims!(out_unfolded, out_col, (2, 1, 3))

    # Now fold back into size(m_in)
    # ...ρ
    @tullio temp[i, j, k] := out_unfolded[i, j, k].ρ
    ρ_back = similar(temp, size(m_in))
    NNlib.fold!(reshape(ρ_back, (f_h, f_w, 1, 1, f_f)), temp, cdims)

    # ...τ
    @tullio temp[i, j, k] = out_unfolded[i, j, k].τ
    τ_back = similar(temp, size(m_in))
    NNlib.fold!(reshape(τ_back, (f_h, f_w, 1, 1, f_f)), temp, cdims)

    # ...results
    @tullio out_back[i, j, k] = Gaussian1d(τ_back[i, j, k], ρ_back[i, j, k])
    free_if_CUDA!.((col_nonflat, col_permuted, max_weights, out_col, temp, ρ_back, τ_back))
    return
end

###
### AvgPool
###
# CPU Version
function forward_avg_pool_factor(m_in::Array{Gaussian1d,3}, out::AbstractVector{Gaussian1d})
    @assert length(out) == size(m_in, 3)
    x = 1 / (size(m_in, 1) * size(m_in, 2))

    Threads.@threads for i in axes(m_in, 3)
        out[i] = forward_product(x, selectdim(m_in, 3, i))
    end
    return
end

# GPU Version
function forward_avg_pool_factor(m_in::AbstractArray{Gaussian1d,3}, out::AbstractVector{Gaussian1d})
    @assert length(out) == size(m_in, 3)
    x = 1 / (size(m_in, 1) * size(m_in, 2))
    forward_col_product(x, reshape(m_in, :, size(m_in, 3)), out)
    return
end

# First computes the message towards the factor, then writes the forward message that leaves the factor into "out".
function backward_avg_pool_factor(m_in::AbstractArray{Gaussian1d,3}, m_pred::AbstractVector{Gaussian1d}, m_back::AbstractVector{Gaussian1d}, out::AbstractArray{Gaussian1d})
    @assert length(m_pred) == length(m_back) == size(m_in, 3)
    @assert size(out) == size(m_in)

    # Compute Backward Sum elementwise
    x = FloatType(1 / (size(m_in, 1) * size(m_in, 2)))
    @tullio out[i, j, k] = backward_mult(x, m_in[i, j, k], m_pred[k], m_back[k])
    return
end


###
### Softmax Factor (forward only for prediction, backward only for training). See appendix of the Laplace-Redux paper or also https://arxiv.org/pdf/1703.00091 (worse).
###
# Computes deterministic softmax. The x will be overwritten with the result. Function mostly copied from NNlib
function softmax!(x::AbstractMatrix{FloatType})
    out = x # just synonym

    max_ = NNlib.fast_maximum(x; dims=1)
    if all(isfinite, max_)
        @tullio out[i, j] = exp(x[i, j] - max_[1, j])
    else
        _zero, _one, _inf = 0.0, 1.0, FloatType(inf)
        @tullio out[i, j] = ifelse(isequal(max_[1, j], _inf), ifelse(isequal(x[i, j], _inf), _one, _zero), exp(x[i, j] - max_[i, j]))
    end
    tmp = sum!(max_, out) # sum out column-wise into max_
    out ./= tmp
end

# Intended for batch-wise computation, but it would be easy to build a vectorized version (for one training example at a time)
function forward_softmax(m_in::AbstractMatrix{Gaussian1d})
    @tullio tau[i, j] := mean(m_in[i, j]) / sqrt(1 + π / (8 * m_in[i, j].ρ))

    out = tau # just synonym
    softmax!(out)
    return out
end

# Intended for example-wise computation (takes input vector belonging to one example). The number of samples is determined by the size of the samples matrix
function backward_softmax(m_in::Vector{Gaussian1d}, i_target::Int, out::Vector{Gaussian1d}, samples::Matrix{FloatType}, likelihoods::Matrix{FloatType})

    @tullio tau[i] := mean(m_in[i]) / sqrt(1 + π / (8 * m_in[i].ρ))

    # Compute maximum of all tau's (including the "distribution-tau" of the sampled variable)
    max_ = NNlib.fast_maximum(tau, dims=1)[1]
    @assert isfinite(max_)

    # Compute sum of all non-target and non-sampled distributions
    @tullio exp_tau[i] := exp(tau[i] - max_)
    sum_probs = sum(exp_tau)

    # Create the samples
    @inbounds Threads.@threads for i in eachindex(m_in)
        sample!((@view samples[:, i]), m_in[i])
    end

    # Compute likelihoods (expected softmax fo the target class)
    exp_target = exp_tau[i_target]
    @tullio likelihoods[i, j] = ifelse(j == i_target,
        exp(samples[i, j] - max_) / (sum_probs - exp_tau[j] + exp(samples[i, j] - max_)),
        exp_target / (sum_probs - exp_tau[j] + exp(samples[i, j] - max_))
    )

    # Find the moments
    m0, m1, m2 = zeros(FloatType, size(out)), zeros(FloatType, size(out)), zeros(FloatType, size(out))
    @inbounds Threads.@threads for (d, mi) in [(0, m0), (1, m1), (2, m2)]
        @tullio mi[j] = likelihoods[i, j] * samples[i, j]^d
    end

    @tullio out[i] = Gaussian1d_m2(; μ=(m1[i] / m0[i]), m2=(m2[i] / m0[i]))
    return
end

@inline function log_softmax_marginal_unnormalized(z::FloatType, max_::FloatType, sum_probs_i::FloatType, log_exp_target::FloatType, z_is_target::Bool, μ_f::FloatType, σ2_f::FloatType)
    log_exp_z = z - max_
    log_exp_target = z_is_target ? log_exp_z : log_exp_target

    log_likelihood = log_exp_target - log(sum_probs_i + exp(log_exp_z))
    return log_likelihood + logpdf_normal(z, μ_f, σ2_f)
end

# Integrals-based solution for backward softmax. Only slightly slower than 1000-sample sampling
function backward_softmax_integrals(m_in::AbstractVector{Gaussian1d}, i_target::Int, out::Vector{Gaussian1d})
    # TODO: Is it worth catching and stopping high-variance forward messages?

    # Necessary pre-work for computing the likelihood (aka expected softmax)
    @tullio tau[i] := mean(m_in[i]) / sqrt(1 + π / (8 * m_in[i].ρ))

    # Compute maximum of all tau's (including the "distribution-tau" of the sampled variable)
    max_ = NNlib.fast_maximum(tau, dims=1)[1]
    @assert isfinite(max_)

    # Compute sum of all non-target and non-sampled distributions
    @tullio exp_tau[i] := exp(tau[i] - max_)
    sum_probs = sum(exp_tau)

    # Find integration bounds of the target
    μ_target, σ_target = mean(m_in[i_target]), sqrt(variance(m_in[i_target]))
    left_target, right_target = μ_target - 7 * σ_target, μ_target + 7 * σ_target

    Threads.@threads for i in eachindex(m_in)
        # Define unnormalize density of the marginal
        μ_f, σ2_f = mean(m_in[i]), variance(m_in[i])
        sum_probs_i = sum_probs - exp_tau[i]
        log_exp_target = tau[i_target] - max_
        max_val = max_

        # Find integration bounds and make sure to integrate over the bounds of the target
        σ_f = sqrt(σ2_f)
        left, right = μ_f - 7 * σ_f, μ_f + 7 * σ_f

        # Extend the integration bounds to include the target, but only when the current variable has any chance of being bigger than the target
        if right > left_target
            left, right = min(left_target, left), max(right_target, right)
        end

        # Compute moments
        m0, error0 = quadgk(z -> exp(log_softmax_marginal_unnormalized(z, max_val, sum_probs_i, log_exp_target, (i == i_target), μ_f, σ2_f)), left, right)
        m1, error1 = quadgk(z -> z * exp(log_softmax_marginal_unnormalized(z, max_val, sum_probs_i, log_exp_target, (i == i_target), μ_f, σ2_f)), left, right)
        m2, error2 = quadgk(z -> z^2 * exp(log_softmax_marginal_unnormalized(z, max_val, sum_probs_i, log_exp_target, (i == i_target), μ_f, σ2_f)), left, right)

        if m0 < 1e-12
            out[i] = Gaussian1d(0.0, 0.0)
        else
            try
                ρ = m0 / (m2 - m1^2 / m0) # = 1 / (m2/m0 - (m1/m0)^2)
                τ = m1 / (m2 - m1^2 / m0) # = ρ * (m1/m0), but also = 1 / (m2/m1 - m1/m0). Would that be more stable?
                @assert isfinite(ρ)
                @assert isfinite(τ)
                out[i] = safe_division(Gaussian1d(τ, ρ), m_in[i]; tolerance=0.1)
            catch
                println("Failed to compute softmax!")
                println("m_in[i]: $(m_in[i])")
                println("m_in[target]: $(m_in[i_target])")
                println("left: $left, right: $right")
                println("left_target: $left_target, right_target: $right_target")
                println("sum_probs_i: $sum_probs_i")
                println("m0: $m0, error: $error0")
                println("m1: $m1, error: $error1")
                println("m2: $m2, error: $error2")
                println("\n\nAll Messages:\n$(m_in)")
                @assert false
            end
        end
    end
    return
end

# Solution for GPU: Just run the CPU-version of the code
@inline function backward_softmax_integrals(m_in::AbstractVector{Gaussian1d}, i_target::Int, out::AbstractVector{Gaussian1d})
    assert_well_defined.(m_in)

    cpu_in = Vector(m_in)
    cpu_out = Vector(out)
    backward_softmax_integrals(cpu_in, i_target, cpu_out)
    out[:] = cpu_out
    return
end


###
### Argmax Factor (alternative to Softmax)
###
# Compute c_i = p(x_i = max(x)) for each column of m_in using an approximation
function forward_argmax(m_in::AbstractMatrix{Gaussian1d})
    # Compute pairwise probabilities
    noise = Gaussian1d(μ=0.0, σ2=0.3)
    @tullio probs[i, j, k] := ifelse(
        i == j,
        1.0,
        cdf_normal(0.0, variable_sum(variable_difference(m_in[i, k], m_in[j, k]), noise)) # Prob that m_in[j] > m_in[i]
    )

    # Product and renormalize
    out = similar(probs, size(m_in))
    prod!(reshape(out, 1, size(out)...), probs)
    free_if_CUDA!(probs)

    sum_out = sum(out, dims=1)
    out ./= sum_out
    return out
end

# TODO: Add hyperparameters for the regularization and the output noise

# Compute a backward message that uses truncation to enforce that target is pairwise bigger than all other variables
function backward_argmax(m_in::AbstractVector{Gaussian1d}, i_target::Int, out::AbstractVector{Gaussian1d}; regularize::Bool=false)
    mask = [i != i_target for i in eachindex(m_in)]
    in_others = (@view m_in[mask])
    out_others = (@view out[mask])

    # Compute backward message from truncation
    m_target = (@ALLOW_SCALAR_IF_CUDA m_in[i_target])
    noise = Gaussian1d(; μ=0.0, σ2=0.3)
    @tullio m_forward[i] := variable_sum(variable_difference(m_target, in_others[i]), noise)
    # @tullio m_back[i] := leaky_truncated(m_forward[i]) / m_forward[i]
    @tullio m_back[i] := truncated(m_forward[i]) / m_forward[i]

    # Compute backward to target first
    @tullio out_others[i] = backward_mult(1.0, m_target, m_forward[i], m_back[i])
    out_prod = prod(out_others)
    if regularize
        out_prod = out_prod * Gaussian1d(; μ=1.0, σ2=0.1)
    end
    out .= out_prod # we only need to set out[i_target], but this is easier for CUDA

    # Now compute backward to all others
    @tullio out_others[i] = scale_ρ(backward_mult(-1.0, in_others[i], m_forward[i], m_back[i]), cdf_normal(0.0, m_forward[i] * -1.0))

    if regularize
        regularizer_term = Gaussian1d(; μ=-1.0, σ2=0.1)
        @tullio out_others[i] = out_others[i] * regularizer_term
    end

    free_if_CUDA!.((m_forward, m_back))
    return
end