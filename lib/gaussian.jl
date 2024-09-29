include("utils.jl")
import Distributions: MvNormal, normpdf, normlogpdf, normcdf, normlogcdf, quantile
using LinearAlgebra
using Printf
using Tullio
import Statistics
using Polyester
import Random
import IrrationalConstants
using SpecialFunctions


###
### Library for 1d Gaussians
###
struct Gaussian1d
    τ::FloatType
    ρ::FloatType

    function Gaussian1d(τ::FloatType, ρ::FloatType)
        @assert ρ >= 0
        @assert !isnan(τ) && !isnan(ρ)
        @assert !isinf(τ) && !isinf(ρ)
        return new(τ, ρ)
    end
end

# Moment Parameter Constructors
Gaussian1d(; μ::FloatType, σ2::FloatType) = Gaussian1d(μ / σ2, FloatType(1) / σ2)
function Gaussian1d_m2(; μ::FloatType, m2::FloatType)
    σ2 = m2 - μ^2
    return Gaussian1d(μ / σ2, 1 / σ2)
end

# Treat Gaussian1d as scalar when doing broadcasting
Base.Broadcast.broadcastable(g::Gaussian1d) = Ref(g)

# The G(0, 0) distribution is the multiplicative neutral element
Base.one(::Gaussian1d) = Gaussian1d(0.0, 0.0)
Base.one(::Type{Gaussian1d}) = Gaussian1d(0.0, 0.0)
Gaussian1d() = Gaussian1d(FloatType(0), FloatType(0))

# Needed for NNlib.fold to work
function Gaussian1d(i::Int)
    @assert i == 0
    return Gaussian1d()
end

function GaussianTensor(; μ::AbstractArray{FloatType}, σ2::AbstractArray{FloatType})
    @assert size(μ) == size(σ2)

    τ = μ ./ σ2
    ρ = 1 ./ σ2

    return Gaussian1d.(τ, ρ)
end

function NaturalGaussianTensor(; τ::AbstractArray{FloatType}, ρ::AbstractArray{FloatType})
    @assert size(τ) == size(ρ)
    return Gaussian1d.(τ, ρ)
end

# Return Vector{Gaussian1d}
function NaturalGaussianTensor(n::Int)
    z = zeros(FloatType, n)
    return Gaussian1d.(z, z)
end

# Return Matrix{Gaussian1d}
function NaturalGaussianTensor(n::Int, m::Int)
    z = zeros(FloatType, n, m)
    return Gaussian1d.(z, z)
end

# Return Array{Gaussian1d,3}
function NaturalGaussianTensor(n::Int, m::Int, l::Int)
    z = zeros(FloatType, n, m, l)
    return Gaussian1d.(z, z)
end

# Return Array{Gaussian1d,4}
function NaturalGaussianTensor(n::Int, m::Int, l::Int, o::Int)
    z = zeros(FloatType, n, m, l, o)
    return Gaussian1d.(z, z)
end

# Return Array{Gaussian1d,5}
function NaturalGaussianTensor(n::Int, m::Int, l::Int, o::Int, p::Int)
    z = zeros(FloatType, n, m, l, o, p)
    return Gaussian1d.(z, z)
end


###
### Density Computations
###
function Base.:*(g1::Gaussian1d, g2::Gaussian1d)
    return (Gaussian1d(g1.τ + g2.τ, g1.ρ + g2.ρ))
end

# Divide one density by the other.
function Base.:/(g1::Gaussian1d, g2::Gaussian1d)
    @assert g1.ρ >= g2.ρ
    return Gaussian1d(g1.τ - g2.τ, g1.ρ - g2.ρ)
end

function safe_division(g1::Gaussian1d, g2::Gaussian1d; tolerance::FloatType=1e-8)
    try
        diff = g1.ρ - g2.ρ
        if g1.ρ - g2.ρ >= -tolerance
            diff = max(FloatType(0), diff)
        end
        return Gaussian1d(g1.τ - g2.τ, diff)
    catch
        println("Failed to divide the following two gaussians:")
        display(g1)
        display(g2)
        return (Gaussian1d(g1.τ - g2.τ, g1.ρ - g2.ρ))
    end
end

# β is the update rate
function EMA(g_old::Gaussian1d, g_new::Gaussian1d, β::FloatType)
    τ = (1 - β) * g_old.τ + β * g_new.τ
    ρ = (1 - β) * g_old.ρ + β * g_new.ρ
    return Gaussian1d(τ, ρ)
end

function truncated(g::Gaussian1d)
    sqrt_ρ = sqrt(g.ρ)
    t = g.τ / sqrt_ρ
    v_t = exp(logpdf_normal(t) - logcdf_normal(t))

    ρ_update = (1 - v_t * (v_t + t))

    ρ_back = g.ρ / ρ_update
    τ_back = (g.τ + v_t * sqrt_ρ) / ρ_update

    return Gaussian1d(τ_back, ρ_back)
end

# Cut off at the 90th quartile instead of at 0
# Alternative approach (which is not used here): https://math.stackexchange.com/questions/195911/calculation-of-the-covariance-of-gaussian-mixtures/195984#195984
function leaky_truncated(g::Gaussian1d; quantile::FloatType=0.5)
    cutoff = min(0, quantile_normal(quantile, mean(g), variance(g)))
    g = g - cutoff

    # Compute truncated Parameters
    sqrt_ρ = sqrt(g.ρ)
    t = g.τ / sqrt_ρ
    v_t = pdf_normal(t) / cdf_normal(t)

    ρ_update = (1 - v_t * (v_t + t))

    ρ_back = g.ρ / ρ_update
    τ_back = (g.τ + v_t * sqrt_ρ) / ρ_update

    return Gaussian1d(τ_back, ρ_back) + cutoff
end


###
### Linear Transformations of the Gaussian variables
###
function Base.:*(a::FloatType, g::Gaussian1d)
    return Gaussian1d(g.τ / a, g.ρ / a^2)
end

function Base.:*(g::Gaussian1d, a::FloatType)
    return Gaussian1d(g.τ / a, g.ρ / a^2)
end

function Base.:/(g::Gaussian1d, a::FloatType)
    return Gaussian1d(g.τ * a, g.ρ * a^2)
end

function Base.:+(g::Gaussian1d, a::FloatType)
    return Gaussian1d(g.τ + a * g.ρ, g.ρ)
end

function Base.:-(g::Gaussian1d, a::FloatType)
    return Gaussian1d(g.τ - a * g.ρ, g.ρ)
end


###
### Operations between Gaussian variables (not densities)
###
# For x1 ~ g1, x2 ~ g2, return the distribution of x1+x2
function variable_sum(g1::Gaussian1d, g2::Gaussian1d)
    return Gaussian1d(; μ=(mean(g1) + mean(g2)), σ2=(variance(g1) + variance(g2)))
end

# For x1 ~ g1, x2 ~ g2, return the distribution of x1-x2
function variable_difference(g1::Gaussian1d, g2::Gaussian1d)
    return Gaussian1d(; μ=(mean(g1) - mean(g2)), σ2=(variance(g1) + variance(g2)))
end


###
### Fused Operations (for performance reasons)
###
@inline function multiplication_mean(g1::Gaussian1d, g2::Gaussian1d)
    return (g1.τ + g2.τ) / (g1.ρ + g2.ρ)
end

@inline function multiplication_variance(g1::Gaussian1d, g2::Gaussian1d)
    return 1 / (g1.ρ + g2.ρ)
end

@inline function division_mean(g1::Gaussian1d, g2::Gaussian1d)
    @assert g1.ρ >= g2.ρ
    return (g1.τ - g2.τ) / (g1.ρ - g2.ρ)
end

@inline function division_variance(g1::Gaussian1d, g2::Gaussian1d)
    @assert g1.ρ >= g2.ρ
    return 1 / (g1.ρ - g2.ρ)
end

@inline function division_τ(g1::Gaussian1d, g2::Gaussian1d)
    @assert g1.ρ >= g2.ρ
    return g1.τ - g2.τ
end

@inline function division_ρ(g1::Gaussian1d, g2::Gaussian1d)
    @assert g1.ρ >= g2.ρ
    return g1.ρ - g2.ρ
end


###
### Compute Moment Parameters
###
@inline function mean(g::Gaussian1d)
    return (g.ρ == 0) ? FloatType(0) : g.τ / g.ρ
end

@inline function variance(g::Gaussian1d)
    return 1 / g.ρ
end

# Compute non-centered 2nd moment (which is "E[X^2]", or "σ2 + E[X]^2]")
function m2(g::Gaussian1d)
    return variance(g) + (mean(g)^2)
end


###
### Thresholding and debugging the values of Gaussian1d's
###
function min_ρ(g::Gaussian1d, min_ρ::FloatType)
    μ = mean(g)
    ρ = max(min_ρ, g.ρ)
    @assert isfinite(ρ)
    return Gaussian1d(μ * ρ, ρ)
end

function max_ρ(g::Gaussian1d, max_ρ::FloatType)
    μ = mean(g)
    ρ = min(max_ρ, g.ρ)
    @assert isfinite(ρ)
    return Gaussian1d(μ * ρ, ρ)
end

function scale_ρ(g::Gaussian1d, a::FloatType)
    μ = mean(g)
    ρ = g.ρ * a
    return Gaussian1d(μ * ρ, ρ)
end

function add_variance(g::Gaussian1d, β2::Float64)
    σ2_new = variance(g) + β2
    return Gaussian1d(; μ=mean(g), σ2=σ2_new)
end


function assert_well_defined(g::Gaussian1d)
    @assert isfinite(g.ρ) && isfinite(g.τ) && g.ρ > 0
    return
end

function maxdiff(g1::Gaussian1d, g2::Gaussian1d)
    return max(abs(g1.ρ - g2.ρ), abs(g1.τ - g2.τ))
end

function maxdiff(g1::AbstractArray{Gaussian1d}, g2::AbstractArray{Gaussian1d})
    @assert size(g1) == size(g2)
    return maximum(maxdiff.(g1, g2))
end

function Base.show(io::IO, g::Gaussian1d)
    if g.ρ > 0
        print(io, "Gaussian1d(μ=$(mean(g)), σ2=$(variance(g)))")
    else
        print(io, "Gaussian1d(NP:τ=$(g.τ), ρ=$(g.ρ))")
    end
end

function short_str(g::Gaussian1d)
    var = round(variance(g), digits=1)
    if var == 0.0
        var = @sprintf("%.1e", variance(g))
    end

    return "($(round(mean(g),digits=1)), $(var))"
end

function short_str(v::Vector{Gaussian1d})
    return join(short_str.(v), "\n")
end

function short_str(m::Matrix{Gaussian1d})
    return join([join(row, " ") for row in eachrow(short_str.(m))], "\n")
end



###
### Wrapped functions from Distributions.jl
###
function sample(g::Gaussian1d; n=1)
    μ, σ = mean(g), sqrt(variance(g))
    return μ .+ σ .* Random.randn(n)
end

# It would be slightly faster to use randn(length(out)), but then we have to allocate. Using randn!(...) is slower than the current solution
function sample!(out::AbstractVector{FloatType}, g::Gaussian1d)
    @batch for i in eachindex(out)
        out[i] = randn()
    end
    @assert out[1] != out[2] # just to make sure that the random numbers are different

    μ, σ = mean(g), sqrt(variance(g))
    out .= μ .+ σ .* out
    return
end

pdf_normal(x::FloatType) = pdf_normal(x, 0.0, 1.0)
pdf_normal(x::FloatType, g::Gaussian1d) = pdf_normal(x, mean(g), variance(g))
@inline function pdf_normal(x::FloatType, μ::FloatType, σ2::FloatType)
    @assert σ2 > 0
    if isinf(σ2)
        return 0.0
    end

    z = (x - μ) / sqrt(σ2)
    return exp(-abs2(z) / 2) * IrrationalConstants.invsqrt2π / sqrt(σ2)
end

logpdf_normal(x::FloatType) = logpdf_normal(x, 0.0, 1.0)
logpdf_normal(x::FloatType, g::Gaussian1d) = logpdf_normal(x, mean(g), variance(g))
function logpdf_normal(x::FloatType, μ::FloatType, σ2::FloatType)
    if σ2 == Inf
        @assert false
    end
    return normlogpdf(μ, sqrt(σ2), x)
end

cdf_normal(x::FloatType) = cdf_normal(x, 0.0, 1.0)
cdf_normal(x::FloatType, g::Gaussian1d) = cdf_normal(x, mean(g), variance(g))
function cdf_normal(x::FloatType, μ::FloatType, σ2::FloatType)
    @assert σ2 > 0
    z = (x - μ) / sqrt(σ2)
    return erfc(-z * IrrationalConstants.invsqrt2) / 2
end


logcdf_normal(x::FloatType) = logcdf_normal(x, 0.0, 1.0)
logcdf_normal(x::FloatType, g::Gaussian1d) = logcdf_normal(x, mean(g), variance(g))
function logcdf_normal(x::FloatType, μ::FloatType, σ2::FloatType)
    return normlogcdf(μ, sqrt(σ2), x)
end

quantile_normal(q::FloatType) = quantile_normal(q, 0.0, 1.0)
quantile_normal(q::FloatType, g::Gaussian1d) = quantile_normal(q, mean(g), variance(g))
function quantile_normal(q::FloatType, μ::FloatType, σ2::FloatType)
    return quantile(Normal(μ, sqrt(σ2)), q)
end




###
### Library for Multivariate Gaussian Distributions
###
# See also: https://github.com/philipphennig/ProbML_Apps/blob/main/07/gaussians.py
struct GaussianDist
    μ::Vector{FloatType}
    Σ::Matrix{FloatType}

    function GaussianDist(μ::Vector{FloatType}, Σ::AbstractMatrix{FloatType})
        @assert (length(μ) == size(Σ, 1)) && (length(μ) == size(Σ, 2))
        return new(μ, Symmetric(Σ))
    end
end

GaussianDist(dims::Int) = GaussianDist(zeros(FloatType, dims), FloatType(1.0) * I(dims))

# Unfortunately, the Distributions package does not have an implementation for the cdf of 
# a bivariate normal distribution. We instead use a C-implementation by John Burkhardt,
# see https://people.math.sc.edu/Burkardt/cpp_src/toms462/toms462.html.
# This is a C-translation of the original Fortran77 implementation by Thomas Donnelly in 1973
# which is based on ideas of Donald Owen dating back to 1956.
function mvcdf(dist::GaussianDist, argument::Vector{FloatType})
    if length(dist.μ) > 2
        error("cdf for three and more dimensional normal distributions is not implemented")
    end
    if length(dist.μ) == 1
        return cdf(Normal(dist.μ[1], dist.Σ[1, 1]), argument[0])
    elseif length(dist.μ) == 2
        marginal1 = FloatType(1) - cdf(Normal(dist.μ[1], sqrt(dist.Σ[1, 1])), argument[1])
        marginal2 = FloatType(1) - cdf(Normal(dist.μ[2], sqrt(dist.Σ[2, 2])), argument[2])
        a = (argument[1] - dist.μ[1]) / sqrt(dist.Σ[1, 1])
        b = (argument[2] - dist.μ[2]) / sqrt(dist.Σ[2, 2])
        correlation = dist.Σ[1, 2] / (sqrt(dist.Σ[1, 1] * dist.Σ[2, 2]))
        bivnor = FloatType(@ccall "Exploration/lib/bivnor.so".bivnor(a::Cdouble, b::Cdouble, correlation::Cdouble)::Cdouble)
        return FloatType(1) - marginal1 - marginal2 + bivnor
    end
end

###
### Code for Conditional Gaussians
###
# Assumption:
#   p(x) = g = N(...)
#   p(y | x) = N(y; Ax, Σ)
#   Now we want to find p(x | y), which is also Gaussian
function condition(g::GaussianDist, A::AbstractMatrix{FloatType}, Σ::AbstractMatrix{FloatType}, y::Vector{FloatType})
    @assert length(size(A)) == 2
    k, n = size(A)
    if n > k
        return _conditional_direct(g, A, Σ, y)
    else
        return _conditional_woodbury(g, A, Σ, y)
    end
end

# TODO: save the cholesky of Σ somewhere
function _conditional_direct(g::GaussianDist, A::AbstractMatrix{FloatType}, Σ::AbstractMatrix{FloatType}, y::Vector{FloatType})
    prior_Σ_cholesky = cholesky(Symmetric(g.Σ))
    Σ_cholesky = cholesky(Symmetric(Σ))

    posterior_prec = inv(prior_Σ_cholesky) .+ A * (Σ_cholesky \ A')
    posterior_Σ = inv(cholesky(Symmetric(posterior_prec)))

    posterior_μ = posterior_Σ * ((prior_Σ_cholesky \ g.μ) .+ A * (Σ_cholesky \ y))

    return GaussianDist(posterior_μ, posterior_Σ)
end

function _conditional_woodbury(g::GaussianDist, A::AbstractMatrix{FloatType}, Σ::AbstractMatrix{FloatType}, y::Vector{FloatType})
    Gram = A' * g.Σ * A .+ Σ
    cholesky_factor = cholesky(Symmetric(Gram))

    μ_new = g.μ + g.Σ * A * (cholesky_factor \ (y - A' * g.μ))
    Σ_new = g.Σ - g.Σ * A * (cholesky_factor \ (A' * g.Σ))

    return GaussianDist(μ_new, Σ_new)
end


###
### Utils
###
function sample(g::GaussianDist)
    # Add some epsilon
    Σ = g.Σ .+ (FloatType(1e-5) * I(length(g.μ)))
    return Vector{FloatType}(rand(MvNormal(g.μ, Symmetric(Σ))))
end

# By default, returns matrix where each row is one sample
function sample(v::Vector{GaussianDist})
    out = [sample(v[i]) for i in 1:length(v)]
    return hcat(out...)'
end

function get_std(g::GaussianDist)
    return diag(g.Σ) .^ 0.5
end

# TODO: Maybe more efficient to just implement the logpdf function directly here? Because I think the MvNormal constructor computes a Cholesky...
function logpdf(g::GaussianDist, x::Vector{FloatType})
    return Distributions.logpdf(MvNormal(g.μ, g.Σ), x)
end

function logpdf(v::Vector{GaussianDist}, X::Matrix{FloatType})
    @assert length(v) == size(X, 1)
    logpdfs = [logpdf(v[i], X[i, :]) for i in 1:length(v)]
    return sum(logpdfs)
end

###
### Some code for linear transformations (of the sampled variable)
###
function Base.:*(A::AbstractMatrix{FloatType}, g::GaussianDist)
    μ_new = A * g.μ
    Σ_new = A * g.Σ * A'
    return GaussianDist(μ_new, Σ_new)
end

function Base.:*(a::FloatType, g::GaussianDist)
    return (a * I(length(g.μ))) * g
end

function Base.:+(g::GaussianDist, b::Vector{FloatType})
    return GaussianDist(g.μ .+ b, g.Σ)
end

function Base.:+(b::Vector{FloatType}, g::GaussianDist)
    return g + b
end

function Base.:-(g::GaussianDist, b::Vector{FloatType})
    return GaussianDist(g.μ .- b, g.Σ)
end

function Base.:-(b::Vector{FloatType}, g::GaussianDist)
    return g - b
end