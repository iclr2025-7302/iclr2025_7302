from __future__ import print_function
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import h5py

# python general_training_scripts/mnist.py & python general_training_scripts/mnist.py --softmax & python general_training_scripts/mnist.py --normalize & python general_training_scripts/mnist.py --softmax --normalize & python general_training_scripts/mnist.py --lenet & python general_training_scripts/mnist.py --softmax --lenet & python general_training_scripts/mnist.py --normalize --lenet & python general_training_scripts/mnist.py --softmax --normalize --lenet

model_name = "linear256"
directory = Path("./models/experiment1_sgd_torch")
USE_SOFTMAX = False # keep to false and use CLI params!
USE_LENET = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if USE_LENET:
            # LeNet-5
            self.conv1 = nn.Conv2d(1, 6, 5, 1, padding=2)
            self.conv2 = nn.Conv2d(6, 16, 5, 1)
            self.fc1 = nn.Linear(400, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
        else:
            # Linear256
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        if USE_LENET:
            x = self.conv1(x)
            x = F.leaky_relu(x, 0.1)
            x = F.max_pool2d(x, 2)
            x = self.conv2(x)
            x = F.leaky_relu(x, 0.1)
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.leaky_relu(x, 0.1)
            x = self.fc2(x)
            x = F.leaky_relu(x, 0.1)
            x = self.fc3(x)
            if USE_SOFTMAX:
                x = F.log_softmax(x, dim=1)
            return x
        else:
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.leaky_relu(x, 0.1)
            x = self.fc2(x)
            x = F.leaky_relu(x, 0.1)
            x = self.fc3(x)
            if USE_SOFTMAX:
                x = F.log_softmax(x, dim=1)
            return x



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if not USE_SOFTMAX:
            target = F.one_hot(target, num_classes=10).float()

        optimizer.zero_grad()
        output = model(data)
        if USE_SOFTMAX:
            loss = F.nll_loss(output, target)
        else:
            loss = F.mse_loss(output, target)
        loss.backward()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, addition="", save_labels=True):
    model.eval()
    test_loss = 0
    correct = 0

    all_preds = []
    targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            if not USE_SOFTMAX:
                target2 = F.one_hot(target, num_classes=10).float()

            output = model(data)
            if USE_SOFTMAX:
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            else:
                test_loss += F.mse_loss(output, target2, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if USE_SOFTMAX:
                all_preds.append(output)
            else:
                all_preds.append(torch.softmax(output, dim=1))
            targets.append(target)

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    with h5py.File(directory / f"class_probs{addition}.h5", 'w') as f:
        if USE_SOFTMAX:
            f.create_dataset('data', data=torch.exp(torch.cat(all_preds)).cpu().numpy())
        else:
            f.create_dataset('data', data=torch.cat(all_preds).cpu().numpy())
    
    if save_labels:
        with h5py.File(directory / f"labels{addition}.h5", 'w') as f:
            f.create_dataset('data', data=torch.cat(targets).cpu().numpy())



def main():
    global USE_SOFTMAX
    global USE_LENET
    global model_name
    global directory

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=90, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    # Own attributes
    parser.add_argument('--softmax', action='store_true', default=False,
                        help='')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='')
    parser.add_argument('--lenet', action='store_true', default=False,
                        help='')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()


    # Swich between models
    if args.lenet == True:
        USE_LENET = True
        model_name = "lenet"

    # Set USE_SOFTMAX appropriately
    assert USE_SOFTMAX == False
    if args.softmax:
        USE_SOFTMAX = True
        model_name += "_softmax"
    else:
        model_name += "_regression"

    # Use correct dir
    if args.normalize:
        directory = Path(str(directory) + "_norm")
    directory.mkdir(exist_ok=True)


    for n in [80, 160, 320, 640, 1280, 2560, 5120, 10240, 60000]:
        torch.manual_seed(args.seed)

        if use_cuda:
            device = torch.device("cuda")
        elif use_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        train_kwargs = {'batch_size': args.batch_size}
        test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True,
                        'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        transform=transforms.Compose([
            transforms.ToTensor(),
            ])
        if args.normalize:
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
        dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)
        dataset1 = Subset(dataset1, list(range(n)))

        train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


        # Also load Fashion MNIST
        dataset_fashion = datasets.FashionMNIST('../data', train=False,
                        transform=transform, download=True)
        fashion_loader = torch.utils.data.DataLoader(dataset_fashion, **test_kwargs)


        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        for epoch in tqdm(range(1, args.epochs + 1)):
            train(args, model, device, train_loader, optimizer, epoch)

        # Eval
        print("Number of Examples: ", n, "\n")
        test(model, device, test_loader, "_" + model_name + f"_{n}")
        test(model, device, fashion_loader, "2_" + model_name + f"_{n}", save_labels=False)

        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()