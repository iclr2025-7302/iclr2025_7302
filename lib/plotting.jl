include("./utils.jl")
include("./datasets.jl")
include("./gaussian.jl")
using Plots: plot, plot!, scatter, scatter!, xlabel!, ylabel!, histogram!, savefig, CURRENT_PLOT, RGBA, heatmap, histogram, title!, bar, bar!, font
import CalibrationErrors

function plot_posterior_preds_pretty(d::Dataset, X_preds::Matrix{FloatType}, posterior_preds::GaussianDist; ylims=(-1.5, 2), title=nothing, save_name=nothing, dpi=150, show_data::Bool=true)
    # The plots fail if μ gets too big. Thus, clamp it outside of ylims.
    μ_safe = clamp.(posterior_preds.μ, ylims[1] - 1, ylims[2] + 1)

    CURRENT_PLOT.nullableplot = nothing
    for i = 0:3
        ribbon = i * get_std(posterior_preds)
        # Adjust ribbons. If ribbon is invisible, it is of length <2. Visible ribbons can be between ]0, diff_ylims]
        ribbon_top = clamp.(posterior_preds.μ .+ ribbon, ylims[1] - 2, ylims[2] + 2) .- μ_safe
        ribbon_bottom = μ_safe .- clamp.(posterior_preds.μ .- ribbon, ylims[1] - 2, ylims[2] + 2)
        ribbon_safe = max.(ribbon_top, ribbon_bottom)
        plot!(
            X_preds[1, :],
            μ_safe,
            ribbon=ribbon_safe,
            fillalpha=0.1,
            linewidth=3,
            color=RGBA(1, 0, 0, i == 0 ? 1 : 0.4 - 0.1 * i),
            ylims=ylims,
            label=i == 0 ? "Prediction" : "$i x std",
            dpi=dpi
        )
    end

    # Plot data as scatter plot
    if show_data
        p = plot!(
            d.X_train[1, :],
            d.Y_train[1, :],
            seriestype=:scatter,
            legend=:bottomright,
            color=:orange,
            xtickfontsize=14,
            ytickfontsize=14,
            xguidefontsize=16,
            yguidefontsize=16,
            markersize=2,
            label="Data",
            dpi=dpi,
        )
    else
        # Only set plot config
        p = plot!(
            legend=:bottomright,
            xtickfontsize=14,
            ytickfontsize=14,
            xguidefontsize=16,
            yguidefontsize=16,
            dpi=dpi,
        )
    end

    if !isnothing(title)
        title!(title)
    end

    display(p)
    if !isnothing(save_name)
        savefig(p, save_name)
    end
    return p
end

function plot_posterior_preds(d::Dataset, X_preds::Matrix{FloatType}, posterior_preds::GaussianDist; ylims=(-1.5, 2), save_name=nothing)
    p = plot(
        d.X_train[1, :],
        d.Y_train[1, :],
        seriestype=:scatter,
        legend=false,
        color=:orange,
        xtickfontsize=14,
        ytickfontsize=14,
        xguidefontsize=16,
        yguidefontsize=16,
    )

    ribbon = 2 .* get_std(posterior_preds)


    # The plots fail if μ gets too big. Thus, clamp it outside of ylims.
    μ_safe = clamp.(posterior_preds.μ, ylims[1] - 1, ylims[2] + 1)

    # Adjust ribbons. If ribbon is invisible, it is of length <2. Visible ribbons can be between ]0, diff_ylims]
    ribbon_top = clamp.(posterior_preds.μ .+ ribbon, ylims[1] - 2, ylims[2] + 2) .- μ_safe
    ribbon_bottom = μ_safe .- clamp.(posterior_preds.μ .- ribbon, ylims[1] - 2, ylims[2] + 2)
    ribbon_safe = max.(ribbon_top, ribbon_bottom)

    plot!(
        X_preds[1, :],
        μ_safe,
        ribbon=ribbon_safe,
        fillalpha=0.2,
        linewidth=3,
        color=:red,
        ylims=ylims,
    )

    display(p)
    if !isnothing(save_name)
        savefig(p, save_name)
    end
end

# Identical function, but for 1d Gaussians
function plot_posterior_preds_pretty(d::Dataset, X_preds::Matrix{FloatType}, posterior_preds::Vector{Gaussian1d}; ylims=(-1.5, 2), title=nothing, save_name=nothing, show_data::Bool=true, dpi=150)
    posterior_preds_multivariate = GaussianDist(mean.(posterior_preds), Diagonal(variance.(posterior_preds)))
    plot_posterior_preds_pretty(d, X_preds, posterior_preds_multivariate; ylims, title, save_name, show_data, dpi)
end

function plot_posterior_preds(d::Dataset, X_preds::Matrix{FloatType}, posterior_preds::Vector{Gaussian1d}; ylims=(-1.5, 2), save_name=nothing)
    posterior_preds_multivariate = GaussianDist(mean.(posterior_preds), Diagonal(variance.(posterior_preds)))
    plot_posterior_preds(d, X_preds, posterior_preds_multivariate; ylims, save_name)
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

function plot_relative_calibration_curves(probs_torch::AbstractMatrix{FloatType}, labels_torch::AbstractVector{Int}, probs_mp::AbstractMatrix{FloatType}, labels_mp::AbstractVector{Int}; title=nothing, dpi=150)
    ece = CalibrationErrors.ECE(CalibrationErrors.MedianVarianceBinning(20), CalibrationErrors.SqEuclidean())
    ece_torch = ece(CalibrationErrors.ColVecs(probs_torch), labels_torch)
    ece_mp = ece(CalibrationErrors.ColVecs(probs_mp), labels_mp)

    sorted_accs_torch = get_sorted_accuracies(probs_torch, labels_torch)
    sorted_accs_mp = get_sorted_accuracies(probs_mp, labels_mp)

    auroc_torch = area_under_curve(sorted_accs_torch[2])
    auroc_mp = area_under_curve(sorted_accs_mp[2])

    p = plot(sorted_accs_torch..., label="PyTorch / SGD. ECE: $(round(ece_torch,digits=4)). Auroc: $(round(auroc_mp,digits=4))", xlabel="Percent discarded", ylabel="Accuracy", ylims=(0.96, 1.0001), dpi=dpi)
    plot!(p, sorted_accs_mp..., label="Message Passing. ECE: $(round(ece_mp,digits=4)). Auroc: $(round(auroc_torch,digits=4))")
    display(p)

    if !isnothing(title)
        @tullio preds[j] := argmax(posterior_preds[:, j])
        num_correct = sum(preds .== labels)
        savefig(p, "$(title)_$(num_correct).png")
    end
end

function plot_relative_calibration_curves(curves::Vector{Tuple{Matrix{FloatType},Vector{Int},String}}; save_name=nothing, dpi=150, colors=nothing, ylims=nothing, bigger_font=false)
    ece_tool = CalibrationErrors.ECE(CalibrationErrors.MedianVarianceBinning(20), CalibrationErrors.SqEuclidean())

    new_font = bigger_font ? font(13) : nothing
    p = plot(; xlabel="Proportion of discarded test data", ylabel="Accuracy on remaining test data", legendfont=new_font, guidefont=new_font)
    for (k, (preds, labels, label)) in enumerate(curves)
        ece = ece_tool(CalibrationErrors.ColVecs(preds), labels)
        sorted_accs = get_sorted_accuracies(preds, labels)

        # # Check for out-of-order tuples
        # sa1 = (@view sorted_accs[2][1:end-1])
        # sa2 = (@view sorted_accs[2][2:end])
        # @tullio bt[i] := ifelse(sa2[i] < sa1[i], 1, 0)
        # println("$(sum(bt)) / $(length(sa1))")

        auroc = area_under_curve(sorted_accs[2])

        label_text = bigger_font ? "$label" : "$label. ECE: $(round(ece,digits=4)). AUC: $(round(auroc,digits=4))"
        if isnothing(colors)
            plot!(p, sorted_accs..., lw=3, label=label_text, dpi=dpi, ylims=ylims)
        else
            plot!(p, sorted_accs..., lw=3, label=label_text, dpi=dpi, ylims=ylims, color=colors[k])
        end
    end
    display(p)

    if !isnothing(save_name)
        savefig(p, "$(save_name).png")
    end
end

function plot_calibration_curves(curves::Vector{Tuple{Matrix{FloatType},Vector{Int},String}}; n_bins::Int=20, save_name=nothing, dpi=150, colors=nothing)
    ece_tool = CalibrationErrors.ECE(CalibrationErrors.MedianVarianceBinning(20), CalibrationErrors.SqEuclidean())

    # Bin edges
    bin_edges = range(0.0, stop=1.0, length=n_bins + 1)

    p = plot()
    plot!(p, [0, 1], [0, 1], lw=1, line=:dash, label=nothing, color="black")

    for (k, (preds, labels, label)) in enumerate(curves)
        @tullio certainties[j] := maximum(preds[:, j])
        @tullio correct[j] := Int(argmax(preds[:, j]) == labels[j])

        # Initialize arrays to hold bin centers and accuracies
        bin_centers = Float64[]
        bin_accuracies = Float64[]
        bin_confidences = Float64[]

        for i in 1:n_bins
            # Find indices where certainties fall into the current bin
            bin_indices = findall(x -> bin_edges[i] <= x < bin_edges[i+1], certainties)

            if !isempty(bin_indices)
                # Calculate the average confidence (certainty) in this bin
                avg_confidence = Statistics.mean(certainties[bin_indices])
                # Calculate the accuracy (percentage of correct predictions) in this bin
                avg_accuracy = Statistics.mean(correct[bin_indices])

                # Store the results
                push!(bin_centers, avg_confidence)
                push!(bin_accuracies, avg_accuracy)
                push!(bin_confidences, length(bin_indices) / length(certainties)) # Normalized bin count
            end
        end

        # Plot the calibration curve
        ece = ece_tool(CalibrationErrors.ColVecs(preds), labels)
        if isnothing(colors)
            plot!(p, bin_centers, bin_accuracies, lw=3, label="$label. ECE: $(round(ece,digits=4))", xlabel="Predicted probability", ylabel="Proportion of correct predictions", xlim=(0, 1), ylim=(0, 1), dpi=dpi)
        else
            plot!(p, bin_centers, bin_accuracies, lw=3, label="$label. ECE: $(round(ece,digits=4))", xlabel="Predicted probability", ylabel="Proportion of correct predictions", xlim=(0, 1), ylim=(0, 1), dpi=dpi, color=colors[k])
        end

        # Optionally add a histogram of the predicted probabilities
        # bar!(p, bin_centers, bin_confidences, alpha=0.3, label="Histogram of max-class certainties", ylabel="Fraction of Samples")
        # bar!(p, bin_centers, bin_confidences, alpha=0.3, label="Histogram of max-class certainties")
    end
    display(p)

    if !isnothing(save_name)
        savefig(p, "$(save_name).png")
    end
end

function plot_calibration_scatter(curves::Vector{Tuple{Matrix{Float64},Vector{Int},String}}; n_bins::Int=20, save_name=nothing, dpi=150, colors=nothing, bigger_font=false)
    ece_tool = CalibrationErrors.ECE(CalibrationErrors.MedianVarianceBinning(20), CalibrationErrors.SqEuclidean())

    # Bin edges
    bin_edges = range(0.0, stop=1.0, length=n_bins + 1)

    new_font = bigger_font ? font(13) : nothing
    p = plot(; legendfont=new_font, guidefont=new_font)
    plot!(p, [0, 1], [0, 1], lw=1, line=:dash, label=nothing, color="black")

    for (k, (preds, labels, label)) in enumerate(curves)
        @tullio certainties[j] := maximum(preds[:, j])
        @tullio correct[j] := Int(argmax(preds[:, j]) == labels[j])

        # Initialize arrays to hold bin centers and accuracies
        bin_centers = Float64[]
        bin_accuracies = Float64[]
        bin_confidences = Float64[]

        for i in 1:n_bins
            # Find indices where certainties fall into the current bin
            bin_indices = findall(x -> bin_edges[i] <= x < bin_edges[i+1], certainties)

            if !isempty(bin_indices)
                # Calculate the average confidence (certainty) in this bin
                avg_confidence = Statistics.mean(certainties[bin_indices])
                # Calculate the accuracy (percentage of correct predictions) in this bin
                avg_accuracy = Statistics.mean(correct[bin_indices])

                # Store the results
                push!(bin_centers, avg_confidence)
                push!(bin_accuracies, avg_accuracy)
                push!(bin_confidences, length(bin_indices) / length(certainties)) # Normalized bin count
            end
        end

        # Plot the calibration curve as a scatter plot, with marker size scaled by bin_confidences
        ece = ece_tool(CalibrationErrors.ColVecs(preds), labels)
        marker_sizes = 5 .+ max.(0.0, 5 .* log.(100 .* bin_confidences)) # Logarithmic scaling of marker sizes

        label_text = bigger_font ? "$label" : "$label. ECE: $(round(ece,digits=4))"
        if isnothing(colors)
            scatter!(p, bin_centers, bin_accuracies, ms=marker_sizes, lw=2, label=label_text, xlabel="Predicted probability", ylabel="Proportion of correct predictions", xlim=(0, 1), ylim=(0, 1), dpi=dpi)
        else
            scatter!(p, bin_centers, bin_accuracies, ms=marker_sizes, lw=2, label=label_text, xlabel="Predicted probability", ylabel="Proportion of correct predictions", xlim=(0, 1), ylim=(0, 1), dpi=dpi, color=colors[k])
        end
    end
    display(p)

    if !isnothing(save_name)
        savefig(p, "$(save_name).png")
    end
end

# Assumes to get the predictions for two datasets, where "probs1" is in-distribution and "probs" is out-of-distribution.
function plot_ood_roc_curves(probs1_torch::AbstractMatrix{FloatType}, probs2_torch::AbstractMatrix{FloatType}, probs1_mp::AbstractMatrix{FloatType}, probs2_mp::AbstractMatrix{FloatType}; title=nothing, dpi=150)
    roc_torch, auc_torch = get_roc(probs1_torch, probs2_torch)
    roc_mp, auc_mp = get_roc(probs1_mp, probs2_mp)

    p = plot(roc_torch, label="PyTorch / SGD. AUC: $(round(auc_torch, digits=4))", xlabel="False Positive Rate", ylabel="True Positive Rate", xlims=(0, 1), ylims=(0, 1), dpi=dpi)
    plot!(p, roc_mp, label="Message Passing. AUC: $(round(auc_mp, digits=4))")
    plot!(p, [0, 1], [0, 1], line=:dash, color=:black, label="")
    display(p)

    if !isnothing(title)
        savefig(p, "ood_roc_$title.png")
    end
end

function plot_ood_roc_curves(curves::Vector{Tuple{Matrix{FloatType},Matrix{FloatType},String}}; save_name=nothing, dpi=150, colors=nothing, ylims=nothing, bigger_font=false)
    new_font = bigger_font ? font(13) : nothing
    p = plot(; xlabel="False positive rate", ylabel="True positive rate", legendfont=new_font, guidefont=new_font)
    plot!(p, [0, 1], [0, 1], line=:dash, color=:black, label="")
    for (k, (preds1, preds2, label)) in enumerate(curves)
        roc, auc = get_roc(preds1, preds2)

        label_text = bigger_font ? "$label" : "$label. AUC: $(round(auc, digits=4))"
        if isnothing(colors)
            plot!(p, roc, lw=3, label=label_text, dpi=dpi, ylims=ylims)
        else
            plot!(p, roc, lw=3, label=label_text, dpi=dpi, ylims=ylims, color=colors[k])
        end
    end
    display(p)

    if !isnothing(save_name)
        savefig(p, "$(save_name).png")
    end
end

function get_calibration_stats(probs::AbstractMatrix{FloatType}, labels::AbstractVector{Int}, probs_ood::AbstractMatrix{FloatType})
    ece_tool = CalibrationErrors.ECE(CalibrationErrors.MedianVarianceBinning(20), CalibrationErrors.SqEuclidean())
    ece = ece_tool(CalibrationErrors.ColVecs(probs), labels)

    sorted_accs = get_sorted_accuracies(probs, labels)
    auroc = area_under_curve(sorted_accs[2])

    ood_roc, ood_auroc = get_roc(probs, probs_ood)

    # Negative Log Likelihood
    @tullio a[j] := probs[labels[j], j]
    nll = -1 / length(a) * sum(log.(a))
    return ece, auroc, ood_auroc, nll
end