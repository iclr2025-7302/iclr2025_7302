using ProgressBars
using Tullio
const FloatType = Float64
import CalibrationErrors


###
### Other Utils
###
function optional_progress_bar(content, silent::Bool)
    return silent ? content : ProgressBar(content)
end

function transpose_permutedims(A::AbstractMatrix)
    return permutedims(A, (2, 1))
end

function reshape2d(A::AbstractArray)
    @assert ndims(A) > 1
    return reshape(A, :, size(A)[end])
end


###
### CUDA Utils
###
has_cuda_lib() = false
begin
    try
        using CUDA
        global has_cuda_lib() = true
    catch
    end
end

# Execute a statement only on nodes with a GPU
macro CUDA_RUN(statement)
    quote
        if has_cuda_lib()
            $(esc(statement))
        end
    end
end

# Execute a statement only on nodes without a GPU
macro NOT_CUDA_RUN(statement)
    quote
        if !has_cuda_lib()
            $(esc(statement))
        end
    end
end

macro ALLOW_SCALAR_IF_CUDA(statement)
    if @isdefined CUDA
        return quote
            if has_cuda_lib()
                if @isdefined CUDA
                    CUDA.@allowscalar $(esc(statement))
                else
                    $(esc(statement))
                end
            end
        end
    else
        return quote
            $(esc(statement))
        end
    end
end

@inline function isCUDA(a::AbstractArray)
    return has_cuda_lib() && Base.typename(typeof(a)).wrapper == CuArray
end

function free_if_CUDA!(a::AbstractArray)
    @CUDA_RUN begin
        if isCUDA(a)
            CUDA.unsafe_free!(a)
        end
    end
    return
end

###
### Evaluation Plots for Classification Data
###
function get_sorted_accuracies(probs::Matrix{FloatType}, labels::Vector{Int})
    # Order of uncertainties:
    order = reverse(sortperm(maximum(probs, dims=1)[1, :])) # highest certainty comes first
    probs_sorted = probs[:, order]
    @tullio preds_sorted[j] := argmax(probs_sorted[:, j])
    labels_sorted = labels[order]

    # Compute accuracy for each cutoff
    accuracies = cumsum(preds_sorted .== labels_sorted)
    accuracies = accuracies ./ (1:length(accuracies))
    accuracies = reverse(accuracies)

    # First entry is now sum over all (throwing away nothing), last entry means keeping only the highest certainty item
    x = (1:length(accuracies)) ./ length(accuracies)
    return x, accuracies
end

# Assumes the y-values are all >= 0 and belong to x-values that are equally-spaced from 0 to 1
function area_under_curve(y_values::Vector{Float64})
    n = length(y_values)
    h = 1.0 / (n - 1)  # The uniform spacing between x-values

    # Trapezoidal rule: sum the areas of the trapezoids
    area = h * (0.5 * y_values[1] + sum(y_values[2:end-1]) + 0.5 * y_values[end])
    return area
end

# Assumes the entropy of probs2 to be higher
function get_roc(probs1::Matrix{FloatType}, probs2::Matrix{FloatType})
    # Compute entropies
    @tullio entropy1[j] := probs1[i, j] * log(probs1[i, j])
    @tullio entropy2[j] := probs2[i, j] * log(probs2[i, j])

    n = length(entropy1)
    m = length(entropy2)

    # p = histogram(entropy1, label="MNIST", fillalpha=0.3)
    # histogram!(p, entropy2, label="Rotated MNIST", fillalpha=0.3)
    # display(p)

    # Generate merged array
    entropies = [entropy1; entropy2]
    nums = [zeros(n); ones(m)]

    order = sortperm(entropies)
    nums = nums[order]
    sums = cumsum(nums)

    # Compute roc points
    @tullio tp[i] := sums[i] / m
    @tullio fp[i] := (i - sums[i]) / n

    # Using the trapezoidal rule to compute AUC
    auc = sum((fp[2:end] - fp[1:end-1]) .* (tp[2:end] + tp[1:end-1])) / 2
    return (fp, tp), auc
end

"""
    topk_accuracy(y_pred::Matrix{Float64}, y_true::Vector{Int}, k::Int)

Calculate the top-k accuracy for predictions. y_pred should be a matrix of probabilities
where each column is a sample and each row is a class probability.
y_true should be a vector of true class labels (1-based indexing).
"""
function topk_accuracy(y_pred::Matrix{Float64}, y_true::Vector{Int}; k::Int=1)
    n_samples = size(y_pred, 2)
    correct = 0
    
    for i in 1:n_samples
        # Get indices of top k predictions for this sample
        topk_indices = partialsortperm(y_pred[:, i], 1:k, rev=true)
        # Check if true class is in top k
        if y_true[i] in topk_indices
            correct += 1
        end
    end
    
    return correct / n_samples
end

"""
    brier_score(y_pred::Matrix{Float64}, y_true::Vector{Int})

Calculate the Brier score for multiclass predictions.
Lower scores indicate better calibrated predictions (0 is perfect).
y_pred should be a matrix of probabilities where each column is a sample
and each row is a class probability.
y_true should be a vector of true class labels (1-based indexing).
"""
function brier_score(y_pred::Matrix{Float64}, y_true::Vector{Int})
    n_samples = size(y_pred, 2)
    n_classes = size(y_pred, 1)
    
    # Convert true labels to one-hot encoding
    y_true_onehot = zeros(n_classes, n_samples)
    for i in 1:n_samples
        y_true_onehot[y_true[i], i] = 1.0
    end
    
    # Calculate Brier score
    squared_diff = (y_pred .- y_true_onehot).^2
    mean_squared_diff = sum(squared_diff) / n_samples
    
    return mean_squared_diff
end

"""
    average_negative_log_likelihood(y_pred::Matrix{Float64}, y_true::Vector{Int})

Calculate the average negative log likelihood for predictions.
y_pred should be a matrix of probabilities where each column is a sample
and each row is a class probability.
y_true should be a vector of true class labels (1-based indexing).
"""
function average_negative_log_likelihood(y_pred::Matrix{Float64}, y_true::Vector{Int})
    n_samples = length(y_true)
    total_nll = 0.0
    
    for i in 1:n_samples
        # Get predicted probability for true class
        true_prob = y_pred[y_true[i], i]
        # Add negative log likelihood, with small epsilon to avoid log(0)
        total_nll -= log(max(true_prob, eps(Float64)))
    end
    
    # Return average NLL
    return total_nll / n_samples
end

struct EvalStats
    acc::FloatType
    top5_acc::FloatType
    ece::FloatType
    nll::FloatType
    brier::FloatType
    auroc::FloatType
    ood_auroc::FloatType
end

function get_eval_stats(probs::AbstractMatrix{FloatType}, labels::AbstractVector{Int}, probs_ood::AbstractMatrix{FloatType})
    acc = topk_accuracy(probs, labels)
    top5_acc = topk_accuracy(probs, labels, k=5)

    ece_tool = CalibrationErrors.ECE(CalibrationErrors.MedianVarianceBinning(20), CalibrationErrors.SqEuclidean())
    ece = ece_tool(CalibrationErrors.ColVecs(probs), labels)

    nll = average_negative_log_likelihood(probs, labels)

    brier = brier_score(probs, labels)

    sorted_accs = get_sorted_accuracies(probs, labels)
    auroc = area_under_curve(sorted_accs[2])

    ood_roc, ood_auroc = get_roc(probs, probs_ood)

    return EvalStats(acc, top5_acc, ece, nll, brier, auroc, ood_auroc)
end
