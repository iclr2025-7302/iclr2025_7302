include("./gaussian.jl")
using Tullio

###
### This file contains forward/backward messages for "A * B + C".
### - A can be Float/Gaussian1d, but B and C must contain Gaussian1ds
### - Many backward messages take parameters such as "out_A", "out_B", "out_C"
### - If no out array is involved, the backward message will usually be computed towards B
###
# TODO: Perhaps merge operations for GPU/CPU and for "marginal" vs. "marginal + m_to". Concern: Maybe slower because of too many if statements?
# TODO: Refactor the parameters properly to make it clear that marginals are used for product factor-related code
# TODO: Refactor all internal functions (especially the backward functions), perhaps under a new name
# TODO: Rename factor into layer (and the first argument from node into layer)
# TODO: Refactor AvgPool to be similar to MaxPool

###
### Scalar x Scalar
###
@inline function forward_mult_μ(a::FloatType, b::Gaussian1d)
    return a * mean(b)
end

@inline function forward_mult_μ(a::Gaussian1d, b::Gaussian1d)
    return mean(a) * mean(b)
end

@inline function forward_mult_μ(a::FloatType, marginal_b::Gaussian1d, m_to_b::Gaussian1d)
    return a * division_mean(marginal_b, m_to_b)
end

@inline function forward_mult_σ2(a::FloatType, b::Gaussian1d)
    return a^2 / b.ρ
end

@inline function forward_mult_σ2(a::FloatType, marginal_b::Gaussian1d, m_to_b::Gaussian1d)
    return a^2 / division_ρ(marginal_b, m_to_b)
end

@inline function forward_mult_σ2(a::Gaussian1d, b::Gaussian1d)
    return m2(a) * m2(b) - (mean(a)^2 * mean(b)^2)
end

# The m_pred and m_back can be related to a vector' * vector product. Returns backward message towards b
@inline function backward_mult(a::FloatType, b::Gaussian1d, m_pred::Gaussian1d, m_back::Gaussian1d)
    @assert b.ρ >= 1e-8
    ρ_denom = (1 + m_back.ρ * (variance(m_pred) - a^2 * variance(b)))

    ρ = (a^2 * m_back.ρ) / ρ_denom
    τ = a * (m_back.τ - m_back.ρ * (mean(m_pred) - a * mean(b))) / ρ_denom
    return Gaussian1d(τ, ρ)
end

@inline function backward_mult(a::FloatType, marginal_b::Gaussian1d, m_to_b::Gaussian1d, m_pred::Gaussian1d, m_back::Gaussian1d)
    @assert division_ρ(marginal_b, m_to_b) >= 1e-8
    ρ_denom = (1 + m_back.ρ * (variance(m_pred) - a^2 * division_variance(marginal_b, m_to_b)))

    ρ = (a^2 * m_back.ρ) / ρ_denom
    τ = a * (m_back.τ - m_back.ρ * (mean(m_pred) - a * division_mean(marginal_b, m_to_b))) / ρ_denom
    return Gaussian1d(τ, ρ)
end

# Returns backward message towards b!
@inline function backward_mult(a::Gaussian1d, b::Gaussian1d, m_pred::Gaussian1d, m_back::Gaussian1d)
    mean_forward_prod = forward_mult_μ(a, b)
    variance_forward_prod = forward_mult_σ2(a, b)

    ρ_denom = (1 + m_back.ρ * (variance(m_pred) - variance_forward_prod))

    # Sum Factor backward
    ρ = m_back.ρ / ρ_denom
    τ = (m_back.τ - m_back.ρ * (mean(m_pred) - mean_forward_prod)) / ρ_denom

    # Product Factor backward
    ρ *= m2(a)
    τ *= mean(a)

    return Gaussian1d(τ, ρ)
end


###
### Vector x Scalar
###
function backward_mult(a::AbstractArray{Gaussian1d}, b::Gaussian1d, m_pred::AbstractArray{Gaussian1d}, m_back::AbstractArray{Gaussian1d}, out_a::AbstractVector{Gaussian1d})
    @assert size(a) == size(m_pred) == size(m_back)

    if all(m.ρ <= 2e-8 for m in m_pred) && all(m.ρ <= 2e-8 for m in m_back)
        return Gaussian1d(0.0, 1e-8)
    end

    τ_back = 0.0
    ρ_back = 0.0

    μ_b = mean(b)
    m2_b = m2(b)

    for i in eachindex(a)
        a_i = a[i]
        m_pred_i = m_pred[i]
        m_back_i = m_back[i]

        mean_forward_prod = μ_b * mean(a_i)
        variance_forward_prod = m2_b * m2(a_i) - (μ_b^2 * mean(a_i)^2)

        ρ_denom = (1 + m_back_i.ρ * (variance(m_pred_i) - variance_forward_prod))

        # Sum Factor backward
        ρ = m_back_i.ρ / ρ_denom
        τ = (m_back_i.τ - m_back_i.ρ * (mean(m_pred_i) - mean_forward_prod)) / ρ_denom

        # Product Factor backward towards b
        ρ_back += m2(a_i) * ρ
        τ_back += mean(a_i) * τ

        # Product Factor backward towards a[i]
        τ_back_ai = μ_b * τ
        ρ_back_ai = m2_b * ρ
        out_a[i] = Gaussian1d(τ_back_ai, ρ_back_ai)
    end
    return Gaussian1d(τ_back, ρ_back)
end


###
### Vector' x Vector
###
function forward_mult(a::AbstractArray{T}, b::AbstractArray{Gaussian1d}) where {T<:Union{FloatType,Gaussian1d}}
    @assert size(a) == size(b)
    μ = 0.0
    σ2 = 0.0

    @inbounds for i in eachindex(b)
        a_i, b_i = a[i], b[i]
        μ += forward_mult_μ(a_i, b_i)
        σ2 += forward_mult_σ2(a_i, b_i)
    end
    return Gaussian1d(; μ, σ2)
end

function forward_mult(a::AbstractArray{T}, b::AbstractArray{Gaussian1d}, c::Gaussian1d) where {T<:Union{FloatType,Gaussian1d}}
    @assert size(a) == size(b)
    μ = 0.0
    σ2 = 0.0

    @inbounds for i in eachindex(b)
        a_i, b_i = a[i], b[i]
        μ += forward_mult_μ(a_i, b_i)
        σ2 += forward_mult_σ2(a_i, b_i)
    end
    return Gaussian1d(; μ=(μ + mean(c)), σ2=(σ2 + variance(c)))
end

function forward_mult(a::AbstractArray{T}, b::AbstractArray{Gaussian1d}, marginal_c::Gaussian1d, m_to_c::Gaussian1d) where {T<:Union{FloatType,Gaussian1d}}
    @assert size(a) == size(b)
    μ = 0.0
    σ2 = 0.0

    @inbounds for i in eachindex(b)
        a_i, b_i = a[i], b[i]
        μ += forward_mult_μ(a_i, b_i)
        σ2 += forward_mult_σ2(a_i, b_i)
    end
    return Gaussian1d(; μ=(μ + division_mean(marginal_c, m_to_c)), σ2=(σ2 + division_variance(marginal_c, m_to_c)))
end

function forward_mult(a::AbstractArray{FloatType}, marginal_b::AbstractArray{Gaussian1d}, m_to_b::AbstractArray{Gaussian1d}, marginal_c::Gaussian1d, m_to_c::Gaussian1d)
    @assert size(a) == size(marginal_b) == size(m_to_b)
    μ = 0.0
    σ2 = 0.0

    @inbounds for i in eachindex(m_to_b)
        a_i, marginal_b_i, m_to_b_i = a[i], marginal_b[i], m_to_b[i]
        μ += forward_mult_μ(a_i, marginal_b_i, m_to_b_i)
        σ2 += forward_mult_σ2(a_i, marginal_b_i, m_to_b_i)
    end
    return Gaussian1d(; μ=(μ + division_mean(marginal_c, m_to_c)), σ2=(σ2 + division_variance(marginal_c, m_to_c)))
end

# Returns prod(b .* a)
function forward_product(a::T, b::AbstractArray{Gaussian1d}) where {T<:Union{FloatType,Gaussian1d}}
    @assert size(a) == size(b)
    μ = 0.0
    σ2 = 0.0

    @inbounds for i in eachindex(b)
        b_i = b[i]
        μ += forward_mult_μ(a, b_i)
        σ2 += forward_mult_σ2(a, b_i)
    end
    return Gaussian1d(; μ, σ2)
end

###
### Vector x Matrix
###
function forward_mult(a::AbstractVector{T}, B::AbstractMatrix{Gaussian1d}, c::AbstractVector{Gaussian1d}, out::AbstractVector{Gaussian1d}) where {T<:Union{FloatType,Gaussian1d}}
    d_in, d_out = size(B)
    @assert length(a) == d_in
    @assert length(out) == length(c) == d_out

    if !isCUDA(a)
        # Compute forward sum for each output separately
        Threads.@threads for j in eachindex(out)
            out[j] = forward_mult(a, (@view B[:, j]), c[j])
        end
    else
        # Compute a full a full product
        @tullio temp[i, j] := forward_mult_μ(a[i], B[i, j])
        _μ = sum(temp, dims=1)
        μ = @view _μ[1, :]

        @tullio temp[i, j] = forward_mult_σ2(a[i], B[i, j])
        _σ2 = sum(temp, dims=1)
        σ2 = @view _σ2[1, :]

        # Reduce to the result vector
        @tullio out[j] = Gaussian1d(;
            μ=μ[j] + mean(c[j]),
            σ2=σ2[j] + variance(c[j])
        )

        # Free CUDA memory
        free_if_CUDA!.((temp, _μ, _σ2))
    end
    return
end

function forward_mult(a::AbstractVector{T}, B::AbstractMatrix{Gaussian1d}, marginal_c::AbstractVector{Gaussian1d}, m_to_c::AbstractVector{Gaussian1d}, out::AbstractVector{Gaussian1d}) where {T<:Union{FloatType,Gaussian1d}}
    d_in, d_out = size(B)
    @assert length(a) == d_in
    @assert length(out) == length(marginal_c) == length(m_to_c) == d_out

    if !isCUDA(a)
        # Compute forward sum for each output separately
        Threads.@threads for j in eachindex(out)
            out[j] = forward_mult(a, (@view B[:, j]), marginal_c[j], m_to_c[j])
        end
    else
        # Compute a full a full product
        @tullio temp[i, j] := forward_mult_μ(a[i], B[i, j])
        _μ = sum(temp, dims=1)
        μ = @view _μ[1, :]

        @tullio temp[i, j] = forward_mult_σ2(a[i], B[i, j])
        _σ2 = sum(temp, dims=1)
        σ2 = @view _σ2[1, :]

        # Reduce to the result vector
        @tullio out[j] = Gaussian1d(;
            μ=μ[j] + division_mean(marginal_c[j], m_to_c[j]),
            σ2=σ2[j] + division_variance(marginal_c[j], m_to_c[j])
        )

        # Free CUDA memory
        free_if_CUDA!.((temp, _μ, _σ2))
    end
    return
end

# CPU Version
function forward_mult(a::AbstractVector{FloatType}, marginal_B::Matrix{Gaussian1d}, m_to_B::AbstractMatrix{Gaussian1d}, marginal_c::AbstractVector{Gaussian1d}, m_to_c::AbstractVector{Gaussian1d}, out::AbstractVector{Gaussian1d})
    d_in, d_out = size(marginal_B)
    @assert length(a) == d_in
    @assert size(m_to_B) == size(marginal_B)
    @assert length(out) == length(marginal_c) == length(m_to_c) == d_out

    # Compute forward sum for each output separately
    Threads.@threads for j in eachindex(out)
        out[j] = forward_mult(a, (@view marginal_B[:, j]), (@view m_to_B[:, j]), marginal_c[j], m_to_c[j])
    end
    return
end

# GPU Version
function forward_mult(a::AbstractVector{FloatType}, marginal_B::AbstractMatrix{Gaussian1d}, m_to_B::AbstractMatrix{Gaussian1d}, marginal_c::AbstractVector{Gaussian1d}, m_to_c::AbstractVector{Gaussian1d}, out::AbstractVector{Gaussian1d})
    d_in, d_out = size(marginal_B)
    @assert length(a) == d_in
    @assert size(m_to_B) == size(marginal_B)
    @assert length(out) == length(marginal_c) == length(m_to_c) == d_out

    # Compute a full a full product
    @tullio temp[i, j] := forward_mult_μ(a[i], marginal_B[i, j], m_to_B[i, j])
    _μ = sum(temp, dims=1)
    μ = @view _μ[1, :]

    @tullio temp[i, j] = forward_mult_σ2(a[i], marginal_B[i, j], m_to_B[i, j])
    _σ2 = sum(temp, dims=1)
    σ2 = @view _σ2[1, :]

    # Reduce to the result vector
    @tullio out[j] = Gaussian1d(;
        μ=μ[j] + division_mean(marginal_c[j], m_to_c[j]),
        σ2=σ2[j] + division_variance(marginal_c[j], m_to_c[j])
    )

    # Free CUDA memory
    free_if_CUDA!.((temp, _μ, _σ2))
    return
end

# Returns something similar to prod(b .* a, dims=1)
# TODO: Is this a _batched method? The AvgPool should be refactored either way...
function forward_col_product(a::FloatType, B::AbstractMatrix{Gaussian1d}, out::AbstractVector{Gaussian1d})
    d_in, d_out = size(B)
    @assert length(out) == d_out

    if !isCUDA(out)
        # Compute forward sum for each output separately
        Threads.@threads for j in eachindex(out)
            out[j] = forward_product(a, (@view B[:, j]))
        end
    else
        # Compute a full a full product
        @tullio temp[i, j] := forward_mult_μ(a, B[i, j])
        _μ = sum(temp, dims=1)
        μ = @view _μ[1, :]

        @tullio temp[i, j] = forward_mult_σ2(a, B[i, j])
        _σ2 = sum(temp, dims=1)
        σ2 = @view _σ2[1, :]

        # Reduce to the result vector
        @tullio out[j] = Gaussian1d(; μ=μ[j], σ2=σ2[j])

        # Free CUDA memory
        free_if_CUDA!.((temp, _μ, _σ2))
    end
    return
end

function backward_mult(a::AbstractVector{Gaussian1d}, B::AbstractMatrix{Gaussian1d}, marginal_c::AbstractVector{Gaussian1d}, m_to_c::AbstractVector{Gaussian1d},
    m_pred::AbstractVector{Gaussian1d}, m_back::AbstractVector{Gaussian1d},
    out_a::AbstractVector{Gaussian1d}, out_B::AbstractMatrix{Gaussian1d}, out_c::AbstractVector{Gaussian1d})

    d_in, d_out = size(B)
    @assert length(a) == length(out_a) == d_in
    @assert size(B) == size(out_B)
    @assert length(m_pred) == length(m_back) == length(marginal_c) == length(m_to_c) == length(out_c) == d_out

    if !isCUDA(a)
        # Back to inputs and non-bias weights
        Threads.@threads for i in eachindex(out_a)
            out_a[i] = backward_mult((@view B[i, :]), a[i], m_pred, m_back, (@view out_B[i, :]))
        end
    else
        # Back to inputs:
        @tullio temp_back[i, j] := backward_mult(B[i, j], a[i], m_pred[j], m_back[j])
        prod!(reshape(out_a, :, 1), temp_back) # sum out 2nd dimension
        free_if_CUDA!(temp_back)

        # Back to weights
        @tullio out_B[i, j] = backward_mult(a[i], B[i, j], m_pred[j], m_back[j])
    end

    # Back to biases
    @tullio out_c[j] = backward_mult(1.0, marginal_c[j], m_to_c[j], m_pred[j], m_back[j])
    return
end

function backward_mult(
    a::AbstractVector{FloatType},
    marginal_B::AbstractMatrix{Gaussian1d}, m_to_B::AbstractMatrix{Gaussian1d},
    marginal_c::AbstractVector{Gaussian1d}, m_to_c::AbstractVector{Gaussian1d},
    m_pred::AbstractVector{Gaussian1d}, m_back::AbstractVector{Gaussian1d},
    out_B::AbstractMatrix{Gaussian1d}, out_c::AbstractVector{Gaussian1d}
)
    d_in, d_out = size(marginal_B)
    @assert length(a) == d_in
    @assert size(m_to_B) == size(marginal_B) == size(out_B)
    @assert length(m_pred) == length(m_back) == length(marginal_c) == length(m_to_c) == length(out_c) == d_out

    # Compute Backward Sum elementwise
    @tullio out_B[i, j] = backward_mult(a[i], marginal_B[i, j], m_to_B[i, j], m_pred[j], m_back[j])

    # Compute backward to biases
    @tullio out_c[j] = backward_mult(1.0, marginal_c[j], m_to_c[j], m_pred[j], m_back[j])
    return
end


###
### Matrix x Matrix
###
function forward_mult(A::AbstractMatrix{T}, B::AbstractMatrix{Gaussian1d}, c::AbstractVector{Gaussian1d}; out::Union{Nothing,AbstractMatrix{Gaussian1d}}=nothing) where {T<:Union{FloatType,Gaussian1d}}
    d1 = size(A, 1)
    d2, d3 = size(B)
    @assert size(A) == (d1, d2)
    @assert length(c) == d3

    if isnothing(out)
        out = similar(B, (d1, d3))
    end
    @assert size(out) == (d1, d3)

    # Actual computation
    if !isCUDA(A)
        # Compute forward sum for each output separately
        Threads.@threads for ind in CartesianIndices(out)
            i, k = ind[1], ind[2]
            out[i, k] = forward_mult((@view A[i, :]), (@view B[:, k]), c[k])
        end
    else
        # Compute a full product
        @tullio temp[i, j, k] := forward_mult_μ(A[i, j], B[j, k])
        _μ = sum(temp, dims=2)
        μ = (@view _μ[:, 1, :])

        @tullio temp[i, j, k] = forward_mult_σ2(A[i, j], B[j, k])
        _σ2 = sum(temp, dims=2)
        σ2 = (@view _σ2[:, 1, :])

        # Reduce to the result vector
        @tullio out[i, k] = Gaussian1d(; μ=μ[i, k] + mean(c[k]), σ2=σ2[i, k] + variance(c[k]))

        free_if_CUDA!.((temp, _μ, _σ2))
    end
    return out
end

function forward_mult(A::AbstractMatrix{Gaussian1d}, B::AbstractMatrix{Gaussian1d}, marginal_c::AbstractVector{Gaussian1d}, m_to_c::AbstractVector{Gaussian1d}, out::AbstractMatrix{Gaussian1d})
    d1 = size(A, 1)
    d2, d3 = size(B)
    @assert size(A) == (d1, d2)
    @assert length(marginal_c) == length(m_to_c) == d3
    @assert size(out) == (d1, d3)

    if !isCUDA(A)
        # Compute forward sum for each output separately
        Threads.@threads for ind in CartesianIndices(out)
            i, k = ind[1], ind[2]
            out[i, k] = forward_mult((@view A[i, :]), (@view B[:, k]), marginal_c[k], m_to_c[k])
        end
    else
        # Compute a full product
        @tullio temp[i, j, k] := forward_mult_μ(A[i, j], B[j, k])
        _μ = sum(temp, dims=2)
        μ = (@view _μ[:, 1, :])

        @tullio temp[i, j, k] = forward_mult_σ2(A[i, j], B[j, k])
        _σ2 = sum(temp, dims=2)
        σ2 = (@view _σ2[:, 1, :])

        # Reduce to the result vector
        @tullio out[i, k] = Gaussian1d(; μ=μ[i, k] + division_mean(marginal_c[k], m_to_c[k]), σ2=σ2[i, k] + division_variance(marginal_c[k], m_to_c[k]))

        free_if_CUDA!.((temp, _μ, _σ2))
    end
    return out
end

function forward_mult(A::AbstractMatrix{FloatType}, marginal_B::AbstractMatrix{Gaussian1d}, m_to_B::AbstractMatrix{Gaussian1d}, marginal_c::AbstractVector{Gaussian1d}, m_to_c::AbstractVector{Gaussian1d}, out::AbstractMatrix{Gaussian1d})
    d1 = size(A, 1)
    d2, d3 = size(marginal_B)
    @assert size(A) == (d1, d2)
    @assert size(m_to_B) == size(marginal_B)
    @assert length(marginal_c) == length(m_to_c) == d3
    @assert size(out) == (d1, d3)

    if !isCUDA(A)
        # Compute forward sum for each output separately
        Threads.@threads for ind in CartesianIndices(out)
            i, k = ind[1], ind[2]
            out[i, k] = forward_mult((@view A[i, :]), (@view marginal_B[:, k]), (@view m_to_B[:, k]), marginal_c[k], m_to_c[k])
        end
    else
        # Compute a full product
        @tullio temp[i, j, k] := forward_mult_μ(A[i, j], marginal_B[j, k], m_to_B[j, k])
        _μ = sum(temp, dims=2)
        μ = (@view _μ[:, 1, :])

        @tullio temp[i, j, k] = forward_mult_σ2(A[i, j], marginal_B[j, k], m_to_B[j, k])
        _σ2 = sum(temp, dims=2)
        σ2 = (@view _σ2[:, 1, :])

        # Reduce to the result vector
        @tullio out[i, k] = Gaussian1d(; μ=μ[i, k] + division_mean(marginal_c[k], m_to_c[k]), σ2=σ2[i, k] + division_variance(marginal_c[k], m_to_c[k]))
        free_if_CUDA!.((temp, _μ, _σ2))
    end
    return out
end

function backward_mult(
    A::AbstractMatrix{Gaussian1d}, B::AbstractMatrix{Gaussian1d}, marginal_c::AbstractVector{Gaussian1d}, m_to_c::AbstractVector{Gaussian1d},
    m_pred::AbstractMatrix{Gaussian1d}, m_back::AbstractMatrix{Gaussian1d},
    out_A::AbstractMatrix{Gaussian1d}, out_B::AbstractMatrix{Gaussian1d}, out_c::AbstractVector{Gaussian1d}
)
    d1 = size(A, 1)
    d2, d3 = size(B)
    @assert size(A) == size(out_A) == (d1, d2)
    @assert size(B) == size(out_B)
    @assert length(marginal_c) == length(m_to_c) == length(out_c) == d3
    @assert size(m_pred) == size(m_back) == (d1, d3)

    # Backward to inputs
    @tullio backward_A[i, j, k] := backward_mult(B[j, k], A[i, j], m_pred[i, k], m_back[i, k])
    prod!(reshape(out_A, size(A)..., 1), backward_A) # multiply the 3rd dimension out

    # Backward to weights
    @tullio backward_B[i, j, k] := backward_mult(A[i, j], B[j, k], m_pred[i, k], m_back[i, k])
    prod!(reshape(out_B, 1, size(out_B)...), backward_B) # multiply the first dimension out

    # Backward to biases
    @tullio backward_c[i, k] := backward_mult(1.0, marginal_c[k], m_to_c[k], m_pred[i, k], m_back[i, k])
    prod!(reshape(out_c, 1, length(out_c)), backward_c) # multiply the first dimension out

    free_if_CUDA!.((backward_B, backward_c))
    return
end

function backward_mult(
    A::AbstractMatrix{FloatType},
    marginal_B::AbstractMatrix{Gaussian1d}, m_to_B::AbstractMatrix{Gaussian1d},
    marginal_c::AbstractVector{Gaussian1d}, m_to_c::AbstractVector{Gaussian1d},
    m_pred::AbstractMatrix{Gaussian1d}, m_back::AbstractMatrix{Gaussian1d},
    out_B::AbstractMatrix{Gaussian1d}, out_c::AbstractVector{Gaussian1d}
)
    d1 = size(A, 1)
    d2, d3 = size(marginal_B)
    @assert size(A) == (d1, d2)
    @assert size(m_to_B) == size(marginal_B) == size(out_B)
    @assert length(marginal_c) == length(m_to_c) == length(out_c) == d3
    @assert size(m_pred) == size(m_back) == (d1, d3)

    # Backward to weights
    @tullio backward_B[i, j, k] := backward_mult(A[i, j], marginal_B[j, k], m_to_B[j, k], m_pred[i, k], m_back[i, k])
    prod!(reshape(out_B, 1, size(out_B)...), backward_B) # multiply the first dimension out

    # Backward to biases
    @tullio backward_c[i, k] := backward_mult(1.0, marginal_c[k], m_to_c[k], m_pred[i, k], m_back[i, k])
    prod!(reshape(out_c, 1, length(out_c)), backward_c) # multiply the first dimension out

    free_if_CUDA!.((backward_B, backward_c))
    return
end

# Vector x Vector, but batched
function forward_mult_batched(A::AbstractMatrix{FloatType}, B::AbstractMatrix{Gaussian1d}; out::Union{Nothing,AbstractVector{Gaussian1d}}=nothing)
    d1, d2 = size(A)
    @assert size(B) == size(A)

    if isnothing(out)
        out = similar(B, d2)
    end
    @assert length(out) == d2

    # Actual computation
    if !isCUDA(A)
        # Compute forward sum for each output separately
        Threads.@threads for j in eachindex(out)
            out[j] = forward_mult((@view A[:, j]), (@view B[:, j]))
        end
    else
        # Compute a full product
        @tullio temp[i, j] := forward_mult_μ(A[i, j], B[i, j])
        _μ = sum(temp, dims=1)
        μ = (@view _μ[1, :])

        @tullio temp[i, j] = forward_mult_σ2(A[i, j], B[i, j])
        _σ2 = sum(temp, dims=1)
        σ2 = (@view _σ2[1, :])

        # Reduce to the result vector
        @tullio out[j] = Gaussian1d(; μ=μ[j], σ2=σ2[j])

        free_if_CUDA!.((temp, _μ, _σ2))
    end
    return out
end

function backward_mult_batched(
    A::AbstractMatrix{FloatType}, B::AbstractMatrix{Gaussian1d},
    m_pred::AbstractVector{Gaussian1d}, m_back::AbstractVector{Gaussian1d},
    out::AbstractMatrix{Gaussian1d}
)
    d1, d2 = size(A)
    @assert size(B) == size(out) == size(A)
    @assert length(m_pred) == length(m_back) == d2

    # Backward to weights
    @tullio out[i, j] = backward_mult(A[i, j], B[i, j], m_pred[j], m_back[j])

    # display(A)
    # display(B)
    # display([reshape(m_pred, 1, :); reshape(m_back, 1, :)])
    # display(out)
    return
end

###
### 3d Tensor x Matrix (batched matrix-matrix multiplication)
###
function forward_mult_batched(A::AbstractArray{T,3}, B::AbstractMatrix{Gaussian1d}, c::AbstractVector{Gaussian1d}; out::Union{Nothing,AbstractMatrix{Gaussian1d}}=nothing) where {T<:Union{FloatType,Gaussian1d}}
    d1, d2, d4 = size(A)
    d3 = size(B, 2)
    @assert size(B) == (d2, d3)
    @assert length(c) == d3

    if isnothing(out)
        out = similar(B, (d1, d3, d4))
    end
    @assert size(out) == (d1, d3, d4)

    # Actual computation
    if !isCUDA(A)
        # Compute forward sum for each output separately
        Threads.@threads for ind in CartesianIndices(out)
            i, k, l = ind[1], ind[2], ind[3]
            out[i, k, l] = forward_mult((@view A[i, :, l]), (@view B[:, k]), c[k])
        end
    else
        # Compute a full product
        @tullio temp[i, j, k, l] := forward_mult_μ(A[i, j, l], B[j, k])
        _μ = sum(temp, dims=2)
        μ = (@view _μ[:, 1, :, :])

        @tullio temp[i, j, k, l] = forward_mult_σ2(A[i, j, l], B[j, k])
        _σ2 = sum(temp, dims=2)
        σ2 = (@view _σ2[:, 1, :, :])

        # Reduce to the result vector
        @tullio out[i, k, l] = Gaussian1d(; μ=μ[i, k, l] + mean(c[k]), σ2=σ2[i, k, l] + variance(c[k]))
        free_if_CUDA!.((temp, _μ, _σ2))
    end
    return out
end