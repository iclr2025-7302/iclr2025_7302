from __future__ import print_function
import argparse
from pathlib import Path
import torch
import ivon
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import h5py
import os

# run with: python cifar10_exp.py --optimizer AdamW & python cifar10_exp.py --optimizer IVON


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, 3, padding=0),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, padding=0),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            # Head
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.model(x)


def train_epoch(args, model, device, train_loader, optimizer, epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for _ in range(args.train_samples):
            if args.optimizer == "IVON":
                with optimizer.sampled_params(train=True):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
            else:
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
        optimizer.step()


def train(args, model, optimizer,device, train_loader, no_epochs):
    model.train()
    scheduler = CosineAnnealingLR(optimizer, T_max=no_epochs)
    
    for epoch in tqdm(range(1, no_epochs + 1)):
        train_epoch(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()


def test(args, model, optimizer, device, test_loader, save_path=None):
    model.eval()
    test_loss = 0
    correct = 0

    all_preds = []
    if args.optimizer == "IVON":
        # when using IVON we evaluate the model at the means and via sampling
        all_preds_ivon = []
    targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            prob = F.softmax(output, dim=1)
            class_pred = prob.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += class_pred.eq(target.view_as(class_pred)).sum().item()
            all_preds.append(prob)
            targets.append(target)

            if args.optimizer == "IVON":
                sampled_probs = []
                for i in range(args.test_samples):
                    with optimizer.sampled_params():
                        sampled_logits = model(data)
                        sampled_probs.append(F.softmax(sampled_logits, dim=1))
                prob = torch.mean(torch.stack(sampled_probs), dim=0)
                all_preds_ivon.append(prob)

    test_loss /= len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    if args.optimizer == "IVON":
        with h5py.File(save_path, "w") as f:
            f.create_dataset("preds@mean", data=torch.cat(all_preds).cpu().numpy())
            f.create_dataset("preds@samples", data=torch.cat(all_preds_ivon).cpu().numpy())
            f.create_dataset("labels", data=torch.cat(targets).cpu().numpy())
    else:
        with h5py.File(save_path, "w") as f:
            f.create_dataset("preds", data=torch.cat(all_preds).cpu().numpy())
            f.create_dataset("labels", data=torch.cat(targets).cpu().numpy())


def get_appropriate_device(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    if use_cuda:
        return torch.device("cuda")
    elif use_mps:
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_training_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Example")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        metavar="N",
        help="optimizer (default: AdamW)",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=1,
        metavar="N",
        help="number of MC training samples for each training step(default: 1)",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=100,
        metavar="N",
        help="number of MC test samples (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        metavar="N",
        help="number of epochs to train (default: 30)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument("--normalize", action="store_true", default=True, help="")
    return parser


def get_save_directory(args):
    save_dir = Path("./experiment_results/experiment_cifar10")
    save_dir.mkdir(exist_ok=True, parents=True)
    return save_dir


def get_model_transform(args):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    if args.normalize:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )
    return transform


def main():
    model_name = "conv890k"

    parser = get_training_args()
    args = parser.parse_args()
    save_dir = get_save_directory(args)
    torch.manual_seed(args.seed)
    device = get_appropriate_device(args)

    model = Net().to(device)

    transform = get_model_transform(args)

    # Load dataset
    train_dataset = datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10("./data", train=False, transform=transform)

    # Set up data loaders
    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if device.type == "cuda":
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # set up optimizer
    optimizer = args.optimizer
    if optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters())
    elif optimizer == "IVON":
        optimizer = ivon.IVON(model.parameters(), lr=1e-1, ess=len(train_loader.dataset))
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    # Train
    train(args, model, optimizer, device, train_loader, no_epochs=args.epochs)

    # Eval
    test(
        args,
        model,
        optimizer,
        device,
        test_loader,
        save_path=save_dir / f"{args.optimizer}.h5",
    )

    # Also load Street View House Numbers for out-of-distribution testing
    dataset_svhn = datasets.SVHN(
        "./data", split="test", transform=transform, download=True
    )
    svhn_loader = torch.utils.data.DataLoader(dataset_svhn, **test_kwargs)
    test(
        args,
        model,
        optimizer,
        device,
        svhn_loader,
        save_path=save_dir / f"{args.optimizer}_svhn.h5",
    )

    if args.save_model:
        torch.save(
            model.state_dict(),
            save_dir / f"{args.optimizer}.pt",
        )


if __name__ == "__main__":
    main()
