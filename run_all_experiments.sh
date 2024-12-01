# Ensure conda is initialized for the current shell session
eval "$(conda shell.bash hook)"

conda activate mp-bnns-experiments

# CIFAR-10 experiments
julia --project=@bnn_mp cifar10_exp.jl
python cifar10_exp.py --optimizer=AdamW
python cifar10_exp.py --optimzier=IVON
julia --project=@bnn_mp cifar10_eval.jl

# synthetic data demonstration
julia --project=@bnn_mp toy_ood_uncertainty.jl