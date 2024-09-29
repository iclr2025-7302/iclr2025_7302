include("lib/utils.jl")
@CUDA_RUN using CUDA, KernelAbstractions, CUDA.CUDAKernels
include("lib/gaussian.jl")
include("lib/message_equations.jl")
include("lib/messages_gaussian_mult.jl")
include("lib/factor_graph.jl")
include("lib/datasets.jl")
using ProgressBars
using Statistics: mean as Stats_mean
using BenchmarkTools
using Tullio
using Adapt
import Random;



###
### Create inputs
###
Random.seed!(98)
d_in = 784
d_out = 1000
d_train = 100
d_conv = (28, 28, 4)
d_conv_back = (24, 24, 8)
d_maxpool_back = (14, 14, 4)
num_b = 10

As = [randn(Float64, d_in) for _ in 1:num_b]
Bs = [abs.(randn(Float64, d_in)) for _ in 1:num_b]
Cs = [randn(Float64, d_out) for _ in 1:num_b]
Ds = [abs.(randn(Float64, d_out)) for _ in 1:num_b];
Es = [randn(Float64, d_train) for _ in 1:num_b]
Fs = [abs.(randn(Float64, d_train)) for _ in 1:num_b];
Gs = [randn(Float64, d_conv) for _ in 1:num_b]
Hs = [abs.(randn(Float64, d_conv)) for _ in 1:num_b];
Is = [randn(Float64, d_conv_back) for _ in 1:num_b]
Js = [abs.(randn(Float64, d_conv_back)) for _ in 1:num_b];
Ks = [randn(Float64, d_maxpool_back) for _ in 1:num_b]
Ls = [abs.(randn(Float64, d_maxpool_back)) for _ in 1:num_b];
train_labels = [randn(Float64, d_train) for _ in 1:num_b]
train_labels_class = [argmax(train_labels[i]) for i in 1:num_b]

m_forward = [Gaussian1d.(As[i], Bs[i]) for i in 1:num_b];
m_backward = [Gaussian1d.(Cs[i], Ds[i]) for i in 1:num_b];
m_train = [Gaussian1d.(Es[i], Fs[i]) for i in 1:num_b];
m_conv_forward = [Gaussian1d.(Gs[i], Hs[i]) for i in 1:num_b];
m_conv_backward = [Gaussian1d.(Is[i], Js[i]) for i in 1:num_b];
m_maxpool_backward = [Gaussian1d.(Ks[i], Ls[i]) for i in 1:num_b];



###
### CPU (new)
###
# Layers with comparable "old" layers
l1 = FirstGaussianLinearLayerFactor(d_in, d_out, num_b, nothing);
l2 = GaussianLinearLayerFactor(d_in, d_out, num_b, nothing);
l3 = LeakyReLUFactor((d_in,), d_in, 0.01);

# Layers without
l4 = FirstConv2dFactor(d_conv..., d_conv_back[3], 5, num_b, 0);
l5 = Conv2dFactor(d_conv..., d_conv_back[3], 5, num_b, 0);
l6 = MaxPool2dFactor(d_conv..., 2);
l7 = RegressionFactor(d_train, 0.05^2);
l8 = SoftmaxFactor(d_train);
l9 = ArgmaxFactor(d_train, num_b; regularize=true);


function forward_backward(l::Factor, f, b)
    for i in range(1, length(f))
        forward_message(l, f[i], i)
        backward_message(l, b[i], i)
    end
end

for i in 1:3
    @btime forward_backward($l1, $As, $m_backward)
    @btime forward_backward($l2, $m_forward, $m_backward)
    @btime forward_backward($l3, $m_forward, $m_forward)
    @btime forward_backward($l4, $Gs, $m_conv_backward)
    @btime forward_backward($l5, $m_conv_forward, $m_conv_backward)
    @btime forward_backward($l6, $m_conv_forward, $m_maxpool_backward)
    @btime forward_backward($l7, $m_train, $train_labels)
    @btime forward_backward($l8, $m_train, $train_labels_class)
    @btime forward_backward($l9, $m_train, $train_labels_class)
    println("")
end

# 90.3 ms
# 147.8 ms
# 3.9 ms
# 40.8 ms
# 102.1 ms
# 10.6 ms
# 0.0 ms
# 12.5 ms
# 0.2 ms




###
### GPU (new)
###
cu_l1 = adapt(CuArray, l1)
cu_l2 = adapt(CuArray, l2)
cu_l3 = adapt(CuArray, l3)
cu_l4 = adapt(CuArray, l4)
cu_l5 = adapt(CuArray, l5)
cu_l6 = adapt(CuArray, l6)
cu_l7 = adapt(CuArray, l7)
cu_l8 = adapt(CuArray, l8)
cu_l9 = adapt(CuArray, l9);

cu_As = adapt.(CuArray, As)
cu_Gs = adapt.(CuArray, Gs)
cu_train_labels = adapt.(CuArray, train_labels)
cu_train_labels_class = adapt.(CuArray, train_labels_class);

cu_m_forward = adapt.(CuArray, m_forward)
cu_m_backward = adapt.(CuArray, m_backward)
cu_m_train = adapt.(CuArray, m_train)
cu_m_conv_forward = adapt.(CuArray, m_conv_forward)
cu_m_conv_backward = adapt.(CuArray, m_conv_backward)
cu_m_maxpool_backward = adapt.(CuArray, m_maxpool_backward);

for i in 1:3
    @btime CUDA.@sync forward_backward($cu_l1, $cu_As, $cu_m_backward)
    @btime CUDA.@sync forward_backward($cu_l2, $cu_m_forward, $cu_m_backward)
    @btime CUDA.@sync forward_backward($cu_l3, $cu_m_forward, $cu_m_forward)
    @btime CUDA.@sync forward_backward($cu_l4, $cu_Gs, $cu_m_conv_backward)
    @btime CUDA.@sync forward_backward($cu_l5, $cu_m_conv_forward, $cu_m_conv_backward)
    @btime CUDA.@sync forward_backward($cu_l6, $cu_m_conv_forward, $cu_m_maxpool_backward)
    @btime CUDA.@sync forward_backward($cu_l7, $cu_m_train, $cu_train_labels)
    @btime CUDA.@sync forward_backward($cu_l8, $cu_m_train, $cu_train_labels_class)
    @btime CUDA.@sync forward_backward($cu_l9, $cu_m_train, $cu_train_labels_class)
    println("")
end

# 13.0 ms
# 23.9 ms
# 3.1 ms
# 9.1 ms
# 15.5 ms
# 5.7 ms
# 0.1 ms
# 15.2 ms
# 2.7 ms



###
### Old Code - go to commit bc3432fcfefc855e62db0ac676703430b0ae488a for it to work ("Avoid breaking old code by changing Gaussian division") on May 6
###
v_l1 = create_variables(d_out);
v_factor_l1 = add_factor.(v_l1);
W, prior_factor = create_weights(d_in, d_out, 1.0)
l1 = FirstGaussianLinearLayerFactor(; W=add_factor.(W), x=As[1], β=1.0, v_out=add_factor.(v_l1));

function forward_backward(l::FirstGaussianLinearLayerFactor, f, b)
    for i in range(1, length(f))
        l.x = f[i]
        forward_message(l)

        send_message.(b[i], v_factor_l1) # send backward message
        backward_message(l)
    end
end

v_in_l2 = create_variables(d_in)
v_in_factor_l2 = add_factor.(v_in_l2)
v_out_l2 = create_variables(d_out)
v_out_factor_l2 = add_factor.(v_out_l2)
W, prior_factor = create_weights(d_in, d_out, 1.0)
l2 = GaussianLinearLayerFactor(; W=add_factor.(W), β=1.0, v_in=add_factor.(v_in_l2), v_out=add_factor.(v_out_l2))

function forward_backward(l::GaussianLinearLayerFactor, f, b)
    for i in range(1, length(f))
        send_message.(f[i], v_in_factor_l2) # forward message
        forward_message(l)
        send_message.(b[i], v_out_factor_l2) # send backward message
        backward_message(l)
    end
end

v_in_l3 = create_variables(d_in)
v_in_factor_l3 = add_factor.(v_in_l3)
v_out_l3 = create_variables(d_in)
v_out_factor_l3 = add_factor.(v_out_l3)
l3 = LeakyReLUFactor(0.01, add_factor.(v_in_l3), add_factor.(v_out_l3));

function forward_backward(l::LeakyReLUFactor, f, b)
    send_message.(b[1], v_out_factor_l3) # send backward message
    for i in range(1, length(f))
        send_message.(f[i], v_in_factor_l3) # forward message
        forward_message(l)
        send_message.(b[i], v_out_factor_l3) # send backward message
        backward_message(l)
    end
end


for i in 1:3
    @btime forward_backward($l1, $As, $m_backward)
    @btime forward_backward($l2, $m_forward, $m_backward)
    @btime forward_backward($l3, $m_forward, $m_forward)
    println("")
end

# 817.0 ms
# 871.0 ms
# 5.8 ms




###
###
### MNIST Experiments
###
###
include("lib/datasets.jl")
include("lib/factor_graph2.jl")
include("lib/plotting.jl")

# Do a warm-up run with 1 batch, then the actual training with 10 batches
for b in [1, 10]
    # Parameters
    σ2_prior = FloatType(1^2)
    β2 = FloatType(0.05^2)
    Random.seed!(98)

    # Generate data
    d = MNIST()
    batch_size = 320

    fg = create_factor_graph(
        [(784, 100, σ2_prior),
            (:LeakyReLU, 0.01),
            (100, 10, σ2_prior)],
        batch_size, β2
    )

    println("Training data size: $(size(d.X_train, 2))")
    nt = batch_size * b
    # nt = size(d.X_train, 2)
    println("Of which used: $nt")

    @time train(fg, d.X_train[:, 1:nt], d.Y_train[:, 1:nt]; num_epochs=1, num_training_its=2)
end
# 1297.848119 seconds (259.25 M allocations: 1.268 TiB, 2.95% gc time)


for b in [1, 10]
    # for nt in [80, 160, 320, 640, 1280, 2560, 5120, 10240]
    # Parameters
    Random.seed!(98)
    batch_size = 320

    # Prepare training
    @CUDA_RUN println("Using CUDA")
    println("Num Threads: $(Threads.nthreads())")

    d = as_regression_dataset(normalize_X!(MNIST()))
    fg = create_factor_graph([
            size(d.X_train)[1:end-1],
            (:Flatten,),
            (:Linear, 100),
            (:LeakyReLU, 0.1),
            (:Linear, 10),
            (:Regression, 0.01)
        ], batch_size
    )
    @CUDA_RUN fg = adapt(CuArray, fg)

    println("Training data size: $(size(d.X_train, ndims(d.X_train)))")
    nt = batch_size * b
    # nt = size(d.X_train, 4)
    println("Of which used: $nt")

    trainer = Trainer(fg, d.X_train[:, :, :, 1:nt], d.Y_train[:, 1:nt])
    # posterior_preds = forward_softmax(predict(fg, d.X_val))

    @time train(trainer; num_epochs=1, num_training_its=2, silent=false)
end
# CPU: 8.895834 seconds (1.21 M allocations: 110.834 MiB)
# GPU: 4.217470 seconds (17.06 M allocations: 500.218 MiB, 1.52% compilation time)



for b in [1, 10]
    # for nt in [80, 160, 320, 640, 1280, 2560, 5120, 10240]
    # Parameters
    Random.seed!(98)
    batch_size = 320

    # Prepare training
    @CUDA_RUN println("Using CUDA")
    println("Num Threads: $(Threads.nthreads())")

    d = as_regression_dataset(normalize_X!(MNIST()))
    fg = create_factor_graph([
            size(d.X_train)[1:end-1],
            (:Conv, 6, 5, 2),
            (:LeakyReLU, 0.1),
            (:MaxPool, 2),
            (:Conv, 16, 5),
            (:LeakyReLU, 0.1),
            (:MaxPool, 2),
            (:Flatten,),
            (:Linear, 120),
            (:LeakyReLU, 0.1),
            (:Linear, 84),
            (:LeakyReLU, 0.1),
            (:Linear, 10),
            (:Regression, 0.01)
        ], batch_size
    )
    @CUDA_RUN fg = adapt(CuArray, fg)

    println("Training data size: $(size(d.X_train, ndims(d.X_train)))")
    nt = batch_size * b
    # nt = size(d.X_train, 4)
    println("Of which used: $nt")

    trainer = Trainer(fg, d.X_train[:, :, :, 1:nt], d.Y_train[:, 1:nt])
    # posterior_preds = forward_softmax(predict(fg, d.X_val))

    println("Training:")
    @time begin
        steps = [1, 1, 2, 2]
        steps = [1]
        for (i, num_training_its) in ProgressBar(enumerate(steps))
            train(trainer; num_epochs=1, num_training_its, silent=true)
        end
    end

    println("Inference:")
    pred_inputs = b == 1 ? d.X_val : repeat(d.X_val, 1, 1, 1, 20)
    @time predict(fg, pred_inputs)
end
GC.gc();
CUDA.reclaim();

# CPU: 261.361915 seconds (77.03 M allocations: 221.565 GiB, 0.41% gc time)
# GPU: 96.3951366667 seconds (293.30 M allocations: 9.070 GiB, 4.29% gc time)
# CPU Inference (200k examples): 464.266515 seconds (2.26 G allocations: 286.147 GiB, 2.07% gc time, 0.00% compilation time)
# GPU Inference (200k examples): 72.352626 seconds (4.22 M allocations: 1.361 GiB, 0.66% gc time)


## Float32
# CPU: 438.923657 seconds (77.27 M allocations: 112.027 GiB, 0.43% gc time, 0.16% compilation time)
# GPU: 88.930622 seconds (285.51 M allocations: 8.761 GiB, 4.12% gc time)
# CPU Inference (200k examples): 392.692586 seconds (2.26 G allocations: 147.910 GiB, 1.39% gc time, 0.14% compilation time)
# GPU Inference (200k examples): 26.540144 seconds (4.18 M allocations: 763.430 MiB, 1.86% gc time, 0.06% compilation time)