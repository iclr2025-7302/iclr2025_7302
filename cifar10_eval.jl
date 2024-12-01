include("lib/utils.jl")
@CUDA_RUN using CUDA, KernelAbstractions, CUDA.CUDAKernels
include("lib/datasets.jl")
include("lib/factor_graph.jl")
include("lib/plotting.jl")
using HDF5
import Images
using Statistics

base_dir = "./experiment_results/experiment_cifar10/"
method_names = ["AdamW", "IVON@mean", "IVON", "MP"]

preds = []
preds_svhn = []
labels = []
labels_svhn = []

for method_name in method_names
    if method_name == "IVON"
        pred_field = "preds@samples"
        file_prefix = "IVON"
    elseif method_name == "IVON@mean"
        pred_field = "preds@mean"
        file_prefix = "IVON"
    else
        pred_field = "preds"
        file_prefix = method_name
    end
    pred = FloatType.(h5read(base_dir * "$file_prefix.h5", pred_field))
    push!(preds, pred)
    label = Int64.(h5read(base_dir * "$file_prefix.h5", "labels")) .+ (method_name == "MP" ? 0 : 1)
    push!(labels, label)
    pred_svhn = FloatType.(h5read(base_dir * "$file_prefix" * "_svhn.h5", pred_field))
    push!(preds_svhn, pred_svhn)
    label_svhn = Int64.(h5read(base_dir * "$file_prefix" * "_svhn.h5", "labels")) .+ (method_name == "MP" ? 0 : 1)
    push!(labels_svhn, label_svhn)
end

latex_table = """
\\begin{table}[]
\\begin{tabular}{@{}lcccccc@{}}
\\toprule
        & Acc. \$\\uparrow\$ & Top-5 Acc. \$\\uparrow\$ & NLL \$\\downarrow\$ & ECE \$\\downarrow\$ & Brier \$\\downarrow\$ & OOD-AUROC \$\\uparrow\$ \\\\ \\midrule
"""

for (i, method_name) in enumerate(method_names)
    eval_stats = get_eval_stats(preds[i], labels[i], preds_svhn[i])
    latex_table *= "$method_name\t& $(round(eval_stats.acc, digits=3)) & $(round(eval_stats.top5_acc, digits=3)) & $(round(eval_stats.nll, digits=3)) & $(round(eval_stats.ece, digits=3)) & $(round(eval_stats.brier, digits=3)) & $(round(eval_stats.ood_auroc, digits=3)) \\\\\n"
end

latex_table *= """
\\bottomrule
\\end{tabular}
\\end{table}
"""

# Save latex table to file
open("cifar10_eval_latex_table.tex", "w") do f
    write(f, latex_table)
end


pred_label_desc_triplets = [(preds[i], labels[i], method_names[i]) for i in eachindex(method_names)];

plot_calibration_curves(pred_label_desc_triplets; n_bins=18, dpi=300, save_name="calibration_cifar10_conv890k");

plot_relative_calibration_curves(pred_label_desc_triplets; save_name="relative_calibration_cifar10_conv890k")

plot_ood_roc_curves([(preds[i], preds_svhn[i], method_names[i]) for i in eachindex(method_names)]; save_name="ood_roc_cifar10_conv890k")

