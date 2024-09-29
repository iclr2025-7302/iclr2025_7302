include("lib/utils.jl")
@CUDA_RUN using CUDA, KernelAbstractions, CUDA.CUDAKernels
include("lib/datasets.jl")
include("lib/factor_graph.jl")
include("lib/plotting.jl")
include("./exp_base.jl")
using HDF5
import Images


###
### A few MNIST demo plots:
###
Random.seed!(98)
d = MNIST()
for (num, i) in enumerate([7, 8, 9, 40002])
    label = d.Y_train[i] - 1
    img = transpose(d.X_train[:, :, 1, i])
    # display(Images.colorview(Images.Gray, img))
    # display(label)
    Images.save("mnist$(num)_$(label).png", Images.colorview(Images.Gray, img))
end



function load_h5(path::String)
    return h5open(path, "r") do file
        read(file["data"])
    end
end

###
### Load regression models
###
dir = "./models/experiment1"
dir_torch = "./models/experiment1_sgd_torch_norm"
model_r1 = "linear256_regression_640_98"
model_r2 = "linear256_regression_5120_98"
model_r3 = "linear256_regression_60000_98"
model_am1 = "linear256_classification_640_98"
model_am2 = "linear256_classification_5120_98"
model_am3 = "linear256_classification_60000_98"
model_tam1 = "linear256_classification_640"
model_tam2 = "linear256_classification_5120"
model_tam3 = "linear256_classification_60000"
model_tr1 = "linear256_regression_640"
model_tr2 = "linear256_regression_5120"
model_tr3 = "linear256_regression_60000"

preds_a1 = FloatType.(load_h5("$(dir)_old/class_probs_julia_$(model_r1).h5"))
preds_a2 = FloatType.(load_h5("$(dir)_old/class_probs_julia_$(model_r2).h5"))
preds_a3 = FloatType.(load_h5("$(dir)_old/class_probs_julia_$(model_r3).h5"))
preds_am1 = FloatType.(load_h5("$(dir)/class_probs_julia_$(model_am1).h5"))
preds_am2 = FloatType.(load_h5("$(dir)/class_probs_julia_$(model_am2).h5"))
preds_am3 = FloatType.(load_h5("$(dir)/class_probs_julia_$(model_am3).h5"))
preds_r1 = FloatType.(load_h5("$(dir)/class_probs_julia_$(model_r1).h5"))
preds_r2 = FloatType.(load_h5("$(dir)/class_probs_julia_$(model_r2).h5"))
preds_r3 = FloatType.(load_h5("$(dir)/class_probs_julia_$(model_r3).h5"))
preds_tsm1 = FloatType.(load_h5("$(dir_torch)/class_probs_$(model_tam1).h5"))
preds_tsm2 = FloatType.(load_h5("$(dir_torch)/class_probs_$(model_tam2).h5"))
preds_tsm3 = FloatType.(load_h5("$(dir_torch)/class_probs_$(model_tam3).h5"))
preds_tr1 = FloatType.(load_h5("$(dir_torch)/class_probs_$(model_tr1).h5"))
preds_tr2 = FloatType.(load_h5("$(dir_torch)/class_probs_$(model_tr2).h5"))
preds_tr3 = FloatType.(load_h5("$(dir_torch)/class_probs_$(model_tr3).h5"))

preds2_a1 = FloatType.(load_h5("$(dir)_old/class_probs2_julia_$(model_r1).h5"))
preds2_a2 = FloatType.(load_h5("$(dir)_old/class_probs2_julia_$(model_r2).h5"))
preds2_a3 = FloatType.(load_h5("$(dir)_old/class_probs2_julia_$(model_r3).h5"))
preds2_am1 = FloatType.(load_h5("$(dir)/class_probs2_julia_$(model_am1).h5"))
preds2_am2 = FloatType.(load_h5("$(dir)/class_probs2_julia_$(model_am2).h5"))
preds2_am3 = FloatType.(load_h5("$(dir)/class_probs2_julia_$(model_am3).h5"))
preds2_r1 = FloatType.(load_h5("$(dir)/class_probs2_julia_$(model_r1).h5"))
preds2_r2 = FloatType.(load_h5("$(dir)/class_probs2_julia_$(model_r2).h5"))
preds2_r3 = FloatType.(load_h5("$(dir)/class_probs2_julia_$(model_r3).h5"))
preds2_tsm1 = FloatType.(load_h5("$(dir_torch)/class_probs2_$(model_tam1).h5"))
preds2_tsm2 = FloatType.(load_h5("$(dir_torch)/class_probs2_$(model_tam2).h5"))
preds2_tsm3 = FloatType.(load_h5("$(dir_torch)/class_probs2_$(model_tam3).h5"))
preds2_tr1 = FloatType.(load_h5("$(dir_torch)/class_probs2_$(model_tr1).h5"))
preds2_tr2 = FloatType.(load_h5("$(dir_torch)/class_probs2_$(model_tr2).h5"))
preds2_tr3 = FloatType.(load_h5("$(dir_torch)/class_probs2_$(model_tr3).h5"))

labels_a1 = load_h5("$(dir)_old/labels_julia_$(model_r1).h5")
labels_a2 = load_h5("$(dir)_old/labels_julia_$(model_r2).h5")
labels_a3 = load_h5("$(dir)_old/labels_julia_$(model_r3).h5")
labels_am1 = load_h5("$(dir)/labels_julia_$(model_am1).h5")
labels_am2 = load_h5("$(dir)/labels_julia_$(model_am2).h5")
labels_am3 = load_h5("$(dir)/labels_julia_$(model_am3).h5")
labels_r1 = load_h5("$(dir)/labels_julia_$(model_r1).h5")
labels_r2 = load_h5("$(dir)/labels_julia_$(model_r2).h5")
labels_r3 = load_h5("$(dir)/labels_julia_$(model_r3).h5")
labels_tsm1 = load_h5("$(dir_torch)/labels_$(model_tam1).h5") .+ 1
labels_tsm2 = load_h5("$(dir_torch)/labels_$(model_tam2).h5") .+ 1
labels_tsm3 = load_h5("$(dir_torch)/labels_$(model_tam3).h5") .+ 1
labels_tr1 = load_h5("$(dir_torch)/labels_$(model_tr1).h5") .+ 1
labels_tr2 = load_h5("$(dir_torch)/labels_$(model_tr2).h5") .+ 1
labels_tr3 = load_h5("$(dir_torch)/labels_$(model_tr3).h5") .+ 1


###
### Load Conv models
###
model_lr1 = "lenet_regression_640_98"
model_lr2 = "lenet_regression_5120_98"
model_lr3 = "lenet_regression_60000_98"
model_lsm1 = "lenet_classification_640_98"
model_lsm2 = "lenet_classification_5120_98"
model_lsm3 = "lenet_classification_60000_98"
model_tlsm1 = "lenet_classification_640"
model_tlsm2 = "lenet_classification_5120"
model_tlsm3 = "lenet_classification_60000"
model_tlr1 = "lenet_regression_640"
model_tlr2 = "lenet_regression_5120"
model_tlr3 = "lenet_regression_60000"

preds_lr1 = FloatType.(load_h5("$(dir)/class_probs_julia_$(model_lr1).h5"))
preds_lr2 = FloatType.(load_h5("$(dir)/class_probs_julia_$(model_lr2).h5"))
preds_lr3 = FloatType.(load_h5("$(dir)/class_probs_julia_$(model_lr3).h5"))
preds_lsm1 = FloatType.(load_h5("$(dir)/class_probs_julia_$(model_lsm1).h5"))
preds_lsm2 = FloatType.(load_h5("$(dir)/class_probs_julia_$(model_lsm2).h5"))
preds_lsm3 = FloatType.(load_h5("$(dir)/class_probs_julia_$(model_lsm3).h5"))
preds_tlr1 = FloatType.(load_h5("$(dir)/class_probs_julia_$(model_lr1).h5"))
preds_tlr2 = FloatType.(load_h5("$(dir)/class_probs_julia_$(model_lr2).h5"))
preds_tlr3 = FloatType.(load_h5("$(dir)/class_probs_julia_$(model_lr3).h5"))
preds_tlsm1 = FloatType.(load_h5("$(dir_torch)/class_probs_$(model_tlsm1).h5"))
preds_tlsm2 = FloatType.(load_h5("$(dir_torch)/class_probs_$(model_tlsm2).h5"))
preds_tlsm3 = FloatType.(load_h5("$(dir_torch)/class_probs_$(model_tlsm3).h5"))
preds_tlr1 = FloatType.(load_h5("$(dir_torch)/class_probs_$(model_tlr1).h5"))
preds_tlr2 = FloatType.(load_h5("$(dir_torch)/class_probs_$(model_tlr2).h5"))
preds_tlr3 = FloatType.(load_h5("$(dir_torch)/class_probs_$(model_tlr3).h5"))

preds2_lr1 = FloatType.(load_h5("$(dir)/class_probs2_julia_$(model_lr1).h5"))
preds2_lr2 = FloatType.(load_h5("$(dir)/class_probs2_julia_$(model_lr2).h5"))
preds2_lr3 = FloatType.(load_h5("$(dir)/class_probs2_julia_$(model_lr3).h5"))
preds2_lsm1 = FloatType.(load_h5("$(dir)/class_probs2_julia_$(model_lsm1).h5"))
preds2_lsm2 = FloatType.(load_h5("$(dir)/class_probs2_julia_$(model_lsm2).h5"))
preds2_lsm3 = FloatType.(load_h5("$(dir)/class_probs2_julia_$(model_lsm3).h5"))
preds2_tlr1 = FloatType.(load_h5("$(dir)/class_probs2_julia_$(model_lr1).h5"))
preds2_tlr2 = FloatType.(load_h5("$(dir)/class_probs2_julia_$(model_lr2).h5"))
preds2_tlr3 = FloatType.(load_h5("$(dir)/class_probs2_julia_$(model_lr3).h5"))
preds2_tlsm1 = FloatType.(load_h5("$(dir_torch)/class_probs2_$(model_tlsm1).h5"))
preds2_tlsm2 = FloatType.(load_h5("$(dir_torch)/class_probs2_$(model_tlsm2).h5"))
preds2_tlsm3 = FloatType.(load_h5("$(dir_torch)/class_probs2_$(model_tlsm3).h5"))
preds2_tlr1 = FloatType.(load_h5("$(dir_torch)/class_probs2_$(model_tlr1).h5"))
preds2_tlr2 = FloatType.(load_h5("$(dir_torch)/class_probs2_$(model_tlr2).h5"))
preds2_tlr3 = FloatType.(load_h5("$(dir_torch)/class_probs2_$(model_tlr3).h5"))

labels_lr1 = load_h5("$(dir)/labels_julia_$(model_lr1).h5")
labels_lr2 = load_h5("$(dir)/labels_julia_$(model_lr2).h5")
labels_lr3 = load_h5("$(dir)/labels_julia_$(model_lr3).h5")
labels_lsm1 = load_h5("$(dir)/labels_julia_$(model_lsm1).h5")
labels_lsm2 = load_h5("$(dir)/labels_julia_$(model_lsm2).h5")
labels_lsm3 = load_h5("$(dir)/labels_julia_$(model_lsm3).h5")
labels_tlr1 = load_h5("$(dir)/labels_julia_$(model_lr1).h5")
labels_tlr2 = load_h5("$(dir)/labels_julia_$(model_lr2).h5")
labels_tlr3 = load_h5("$(dir)/labels_julia_$(model_lr3).h5")
labels_tlsm1 = load_h5("$(dir_torch)/labels_$(model_tlsm1).h5") .+ 1
labels_tlsm2 = load_h5("$(dir_torch)/labels_$(model_tlsm2).h5") .+ 1
labels_tlsm3 = load_h5("$(dir_torch)/labels_$(model_tlsm3).h5") .+ 1
labels_tlr1 = load_h5("$(dir_torch)/labels_$(model_tlr1).h5") .+ 1
labels_tlr2 = load_h5("$(dir_torch)/labels_$(model_tlr2).h5") .+ 1
labels_tlr3 = load_h5("$(dir_torch)/labels_$(model_tlr3).h5") .+ 1;


###
### Histogram
###
blue_colors = ["#0425d9", "#438ccc", "#b9eafa"]
orange_colors = ["#a16333", "#e07f1d", "#ffe6cc"]
green_colors = ["#005c05", "#359a44", "#7ef28d"]

# Insight: Using predictions obtained from regression with Argmax vs. with Softmax produces very different results. The Argmax is much closer to what is expected (img 2). Show together with img2
p = histogram(maximum(preds_a1, dims=1)[1, :], label="R-MP + SM, n=640", bins=range(0, 1.0001, length=101), fillalpha=1.0, fillcolor=blue_colors[1], xlabel="Probability of max-class", ylabel="Number of examples", ylims=(0, 2000), dpi=300)
histogram!(p, maximum(preds_a2, dims=1)[1, :], label="R-MP + SM, n=5120", bins=range(0, 1.0001, length=101), fillalpha=1.0, fillcolor=blue_colors[2])
histogram!(p, maximum(preds_a3, dims=1)[1, :], label="R-MP + SM, n=60000", bins=range(0, 1.0001, length=101), fillalpha=1.0, fillcolor=blue_colors[3])
histogram!(p, maximum(preds_r1, dims=1)[1, :], label="R-MP + AM, n=640", bins=range(0, 1.0001, length=101), fillcolor=orange_colors[1])
histogram!(p, maximum(preds_r2, dims=1)[1, :], label="R-MP + AM, n=5120", bins=range(0, 1.0001, length=101), fillcolor=orange_colors[2])
histogram!(p, maximum(preds_r3, dims=1)[1, :], label="R-MP + AM, n=60000", bins=range(0, 1.0001, length=101), fillalpha=0.5, fillcolor=orange_colors[3])
display(p);
savefig("hist1.png");

# Insight: AM-MP is a bit more certain than SGD, actually. Also used as a comparison image for the first distribution
p = histogram(maximum(preds_tsm2, dims=1)[1, :], label="SM-SGD, n=5120", bins=range(0, 1.0001, length=101), fillcolor=orange_colors[2], xlabel="Probability of max-class", ylabel="Number of examples", ylims=(0, 2000), dpi=300)
histogram!(p, maximum(preds_am2, dims=1)[1, :], label="AM-MP, n=5120", bins=range(0, 1.0001, length=101), fillalpha=1.0, fillcolor=blue_colors[2])
display(p);
savefig("hist2.png");


###
### Calibration Plot
###
# Insight: The ECE of R-SGD >> SM-SGD, but R-MP < AM-MP.
plot_calibration_curves([(preds_r2, labels_r2, "R-MP, n=5120"), (preds_am2, labels_am2, "AM-MP, n=5120"), (preds_tr2, labels_tr2, "R-SGD, n=5120"), (preds_tsm2, labels_tsm2, "SM-SGD, n=5120")]; n_bins=20, dpi=300, save_name="calibration1")

# Insight: The does not linearly grow with more data. Also describe the data distribution (most near 1)
plot_calibration_curves([(preds_r1, labels_r1, "R-MP, n=640"), (preds_r2, labels_r2, "R-MP, n=5120"), (preds_r3, labels_r3, "R-MP, n=60000")]; n_bins=20, colors=blue_colors, dpi=300, save_name="calibration2")


###
### Compare ECE Plot
###
# Insight: ECE is terrible for R-SGD + auroc is much worse than R-MP
plot_relative_calibration_curves([
        (preds_r2, labels_r2, "R-MP, n=5120"), (preds_r3, labels_r3, "R-MP, n=60000"),
        (preds_tr2, labels_tr2, "R-SGD, n=5120"), (preds_tr3, labels_tr3, "R-SGD, n=60000"),
    ]; ylims=(0.9, 1.0001), colors=[blue_colors[1], blue_colors[2], orange_colors[1], orange_colors[2], green_colors[2]], dpi=300, save_name="rocauc1")

# Insight: MP is not monotonously increasing, unlike SGD! But the ECE and AUROC are still better. Maybe print together with img 3
plot_relative_calibration_curves([
        (preds_am3, labels_am3, "AM-MP, n=60000"),
        (preds_tsm3, labels_tsm3, "SM-SGD, n=60000"),
        # (preds_lsm3, labels_lsm3, "LeNet Classification, n=60000"),
    ]; ylims=(0.99, 1.0001), colors=[blue_colors[2], orange_colors[2], green_colors[2]], dpi=300, save_name="rocauc2")

# Insight: Unlike Regression, the LeNet curves are actually monotonous. They are also even better
plot_relative_calibration_curves([
        (preds_r3, labels_r3, "R-MP, n=60000"),
        (preds_lr3, labels_lr3, "LeNet R-MP, n=60000"),
        (preds_lsm3, labels_lsm3, "LeNet AM-MP, n=60000"),
    ]; ylims=(0.99, 1.0001), colors=[blue_colors[2], orange_colors[2], green_colors[2]], dpi=300, save_name="rocauc3")

# Insight: Again, R-SGD is much worse. Print together with img 5
plot_relative_calibration_curves([
        (preds_lr1, labels_lr1, "LeNet R-MP, n=640"), (preds_lr2, labels_lr2, "LeNet R-MP, n=5120"), (preds_lr3, labels_lr3, "LeNet R-MP, n=60000"),
        (preds_tlr1, labels_tlr1, "LeNet R-SGD, n=640"), (preds_tlr2, labels_tlr2, "LeNet R-SGD, n=5120"), (preds_tlr3, labels_tlr3, "LeNet R-SGD, n=60000"),
    ]; ylims=(0.9, 1.0001), colors=[reverse(blue_colors)..., reverse(orange_colors)...], dpi=300, save_name="rocauc4")

# Insight: For 60k data, the SGD model is not much worse. But for less data, it is substantially worse!
plot_relative_calibration_curves([
        (preds_lsm1, labels_lsm1, "LeNet AM-MP, n=640"), (preds_lsm2, labels_lsm2, "LeNet AM-MP, n=5120"), (preds_lsm3, labels_lsm3, "LeNet AM-MP, n=60000"),
        (preds_tlsm1, labels_tlsm1, "LeNet SM-SGD, n=640"), (preds_tlsm2, labels_tlsm2, "LeNet SM-SGD, n=5120"), (preds_tlsm3, labels_tlsm3, "LeNet SM-SGD, n=60000"),
    ]; ylims=(0.9, 1.0001), colors=[reverse(blue_colors)..., reverse(orange_colors)...], dpi=300, save_name="rocauc5")


###
### OOD Recognition
###
# Insight: With little data, MP has much better OOD-Recognition than SGD. For lots of data, the advantage becomes smaller. Show together with plot2
plot_ood_roc_curves([
        (preds_am1, preds2_am1, "AM-MP, n=640"), (preds_am2, preds2_am2, "AM-MP, n=5120"), (preds_am3, preds2_am3, "AM-MP, n=60000"),
        # (preds_tsm1, preds2_tsm1, "Torch Softmax, n=640"), (preds_tsm2, preds2_tsm2, "Torch Softmax, n=5120"), (preds_tsm3, preds2_tsm3, "Torch Softmax, n=60000"),
    ], colors=[reverse(blue_colors)..., reverse(orange_colors)...], dpi=300, save_name="ood1")
plot_ood_roc_curves([
        # (preds_am1, preds2_am1, "Softmax, n=640"), (preds_am2, preds2_am2, "Softmax, n=5120"), (preds_am3, preds2_am3, "Softmax, n=60000"),
        (preds_tsm1, preds2_tsm1, "SM-SGD, n=640"), (preds_tsm2, preds2_tsm2, "SM-SGD, n=5120"), (preds_tsm3, preds2_tsm3, "SM-SGD, n=60000"),
    ], colors=[reverse(orange_colors)...], dpi=300, save_name="ood2")

# Insight: For LeNet, the advantage remains even for lots of data. The 60k SGD-ood is worse than the 640 MP-ood
plot_ood_roc_curves([
        (preds_lsm1, preds2_lsm1, "LeNet AM-MP, n=640"), (preds_lsm2, preds2_lsm2, "LeNet AM-MP, n=5120"), (preds_lsm3, preds2_lsm3, "LeNet AM-MP, n=60000"),
        (preds_tlsm1, preds2_tlsm1, "LeNet SM-SGD, n=640"), (preds_tlsm2, preds2_tlsm2, "LeNet SM-SGD, n=5120"), (preds_tlsm3, preds2_tlsm3, "LeNet SM-SGD, n=60000"),
    ], colors=[reverse(blue_colors)..., reverse(orange_colors)...], dpi=300, save_name="ood3")






###
### Paper Plots
###
# Insight: For LeNet, the advantage remains even for lots of data. The 60k SGD-ood is worse than the 640 MP-ood
plot_ood_roc_curves([
        (preds_lr1, preds2_lr1, "R-MP (LeNet), n=640"), (preds_lr2, preds2_lr2, "R-MP (LeNet), n=5120"), (preds_lr3, preds2_lr3, "R-MP (LeNet), n=60000"),
        (preds_tlsm1, preds2_tlsm1, "SM-SGD (LeNet), n=640"), (preds_tlsm2, preds2_tlsm2, "SM-SGD (LeNet), n=5120"), (preds_tlsm3, preds2_tlsm3, "SM-SGD (LeNet), n=60000"),
    ], colors=[reverse(blue_colors)..., reverse(orange_colors)...], bigger_font=true, dpi=300, save_name="paper_ood")
# plot_ood_roc_curves([
#         (preds_lr1, preds2_lr1, "LeNet R-MP, n=640"), (preds_lr3, preds2_lr3, "LeNet R-MP, n=60000"),
#         (preds_tlsm1, preds2_tlsm1, "LeNet SM-SGD, n=640"), (preds_tlsm3, preds2_tlsm3, "LeNet SM-SGD, n=60000"),
#     ], colors=[blue_colors[1], blue_colors[2], orange_colors[1], orange_colors[2]], dpi=300, save_name="paper_ood")


plot_relative_calibration_curves([
        (preds_lr1, labels_lr1, "R-MP (LeNet), n=640"), (preds_lr2, labels_lr2, "R-MP (LeNet), n=5120"), (preds_lr3, labels_lr3, "R-MP (LeNet), n=60000"),
        (preds_tlsm1, labels_tlsm1, "SM-SGD (LeNet), n=640"), (preds_tlsm2, labels_tlsm2, "SM-SGD (LeNet), n=5120"), (preds_tlsm3, labels_tlsm3, "SM-SGD (LeNet), n=60000"),
    ]; ylims=(0.9, 1.0001), colors=[reverse(blue_colors)..., reverse(orange_colors)...], bigger_font=true, dpi=300, save_name="paper_rocauc")

# plot_relative_calibration_curves([
#         (preds_r3, labels_r3, "R-MP (MLP), n=5120"), (preds_lr3, labels_lr3, "R-MP (LeNet), n=60000"),
#         (preds_tsm3, labels_tsm3, "SM-SGD (MLP), n=5120"), (preds_tlsm3, labels_tlsm3, "SM-SGD (LeNet), n=60000"),
#     ]; ylims=(0.96, 1.0001), colors=[blue_colors[2], blue_colors[1], orange_colors[2], orange_colors[1]], dpi=300, save_name="paper_rocauc2")

plot_calibration_scatter([(preds_r1, labels_r1, "R-MP, n=640"), (preds_am1, labels_am1, "AM-MP, n=640"), (preds_tsm1, labels_tsm1, "SM-SGD, n=640")]; n_bins=20, colors=[blue_colors[2], blue_colors[1], orange_colors[2]], bigger_font=true, dpi=300, save_name="paper_calibration")
