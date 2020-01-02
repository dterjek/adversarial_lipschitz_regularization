import argparse
from copy import deepcopy
import datetime
import os

from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import datasets, models
import torchvision.transforms as transforms



class ALR(object):
    """
    Commented lines were mostly used to treat invalid values that appear when one uses KLD as d_Y

    """
    def __init__(self, d_X, d_Y, lambda_lp, eps_min, eps_max, xi, ip, K):
        super(ALR, self).__init__()
        self.d_X = d_X
        self.d_Y = d_Y
        self.lambda_lp = lambda_lp
        self.eps_min = eps_min
        self.eps_max = eps_max
        if eps_min == eps_max:
            self.eps = lambda x: eps_min * torch.ones(x.size(0), 1, 1, 1, device=x.device)
        else:
            self.eps = lambda x: eps_min + (eps_max - eps_min) * torch.rand(x.size(0), 1, 1, 1, device=x.device)
        self.xi = xi
        self.ip = ip
        self.K = K

    def virtual_adversarial_direction(self, f, x):
        batch_size = x.size(0)
        f.zero_grad()
        normalize = lambda vector: F.normalize(vector.view(batch_size, -1, 1, 1), p=2, dim=1).view_as(x)
        d = torch.rand_like(x) - 0.5
        d = normalize(d)
        for _ in range(self.ip):
            d.requires_grad_()
            x_hat = torch.clamp(x + self.xi * d, min=-1, max=1)
            y = f(x)
            y_hat = f(x_hat)
            y_diff = d_Y(y, y_hat)
            # y_diff[y_diff != y_diff      ] = 0
            # y_diff[y_diff == float("inf")] = 0
            # y_diff[y_diff < 0            ] = 0
            # y_diff = y_diff[y_diff == y_diff].unsqueeze(1)
            # y_diff = y_diff[y_diff != float("inf")].unsqueeze(1)
            # y_diff = y_diff[y_diff >= 0].unsqueeze(1)
            y_diff = torch.mean(y_diff)
            y_diff.backward()
            # if torch.sum(d.grad != d.grad).item() > 0:
            #     print("nan in d.grad")
            d = normalize(d.grad).detach()
            # if torch.sum(d != d).item() > 0:
            #     print("nan in d")
            f.zero_grad()
        r_adv = normalize(d) * self.eps(x)
        r_adv[r_adv != r_adv] = 0
        r_adv[r_adv == float("inf")] = 0
        r_adv_mask = torch.clamp(
            torch.lt(torch.norm(r_adv.view(batch_size, -1, 1, 1), p=2, dim=1, keepdim=True), self.eps_min).float()
            +
            torch.gt(torch.norm(r_adv.view(batch_size, -1, 1, 1), p=2, dim=1, keepdim=True), self.eps_max).float(),
            min=0, max=1
        ).expand_as(x)
        r_adv = (1 - r_adv_mask) * r_adv + r_adv_mask * normalize(torch.rand_like(x) - 0.5)
        return r_adv
        
    def get_adversarial_perturbations(self, f, x):
        r_adv = self.virtual_adversarial_direction(f=deepcopy(f), x=x.detach())
        # if torch.sum(r_adv != r_adv).item() > 0:
        #     print("nans in r_adv")
        # if torch.sum(r_adv == float("inf")).item() > 0:
        #     print("infs in r_adv")
        x_hat = x + r_adv
        # if self.d_X(x, x_hat).min().item() < self.eps_min:
        #     print("min d_X < eps_min", self.d_X(x, x_hat).min().item())
        # if self.d_X(x, x_hat).max().item() > self.eps_max:
        #     print("min d_X > eps_max", self.d_X(x, x_hat).max().item())
        return x_hat

    def get_alp_loss(self, x, x_hat, y, y_hat):
        # y = f(x).detach()
        # y_hat = f(x_hat)
        y_diff = self.d_Y(y, y_hat)
        x_diff = self.d_X(x, x_hat)
        nan_count = torch.sum(y_diff != y_diff).item()
        # x_diff = x_diff[y_diff == y_diff].unsqueeze(1)
        # y_diff = y_diff[y_diff == y_diff].unsqueeze(1)
        # y_diff[y_diff != y_diff] = 0
        inf_count = torch.sum(y_diff == float("inf")).item()
        # x_diff = x_diff[y_diff != float("inf")].unsqueeze(1)
        # y_diff = y_diff[y_diff != float("inf")].unsqueeze(1)
        # y_diff[y_diff == float("inf")] = 0
        neg_count = torch.sum(y_diff < 0).item()
        # x_diff = x_diff[y_diff >= 0].unsqueeze(1)
        # y_diff = y_diff[y_diff >= 0].unsqueeze(1)
        # y_diff[y_diff < 0] = 0
        lip_ratio = y_diff / x_diff
        alp = torch.clamp(lip_ratio - self.K, min=0)
        nonzeros = torch.nonzero(alp)
        alp_count = nonzeros.size(0)
        alp_l1 = torch.mean(alp     )
        alp_l2 = torch.mean(alp ** 2)
        # alp_loss = self.lambda_lp * alp_l2
        alp_loss = self.lambda_lp * alp_l1
        return (
            alp_loss, lip_ratio, x_diff, y_diff, alp_l1, alp_l2, alp_count, nan_count, inf_count, neg_count
        )


def get_timestamp():
    system_time = str(datetime.datetime.now()).replace(" ", "_")
    run_name = system_time.replace(":", "_").replace(".", "_")
    return run_name


def loopy(iterable):
    while True:
        for x in iter(iterable):
            yield x


def get_dataloaders(batch_size):
    os.makedirs("/cache/data/cifar", exist_ok=True)
    dataset_train = datasets.CIFAR10(
        root="/cache/data/cifar10",
        download=True,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img * 2 - 1),
        ]),
        train=True
    )
    dataset_train_per_class = {
        c: torch.utils.data.Subset(
            dataset=dataset_train,
            indices=[i for i, data in enumerate(dataset_train) if data[1] == c]
        ) for c in range(10)
    }
    dataset_train_per_class_split = {
        c: torch.utils.data.random_split(
            dataset=dataset_train_per_class[c],
            lengths=[400, 100, 4500]
        ) for c in range(10)
    }
    dataset_labeled = torch.utils.data.ConcatDataset(
        datasets=[dataset_train_per_class_split[c][0] for c in range(10)]
    )
    dataset_valid = torch.utils.data.ConcatDataset(
        datasets=[dataset_train_per_class_split[c][1] for c in range(10)]
    )
    dataset_unlabeled = torch.utils.data.ConcatDataset(
        datasets=[dataset_train_per_class_split[c][2] for c in range(10)]
    )
    dataloader_train = loopy(torch.utils.data.DataLoader(
        dataset=dataset_labeled,
        batch_size=batch_size,
        shuffle=True
    ))
    dataloader_reg = loopy(torch.utils.data.DataLoader(
        dataset=dataset_unlabeled,
        batch_size=batch_size * 4,
        shuffle=True
    ))
    dataloader_valid = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=100,
        shuffle=False
    )
    dataset_test = datasets.CIFAR10(
        root="/cache/data/cifar10",
        download=True,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img * 2 - 1),
        ]),
        train=False
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=100,
        shuffle=False
    )
    return dataloader_train, dataloader_reg, dataloader_valid, dataloader_test


def accuracy(output, target, top_k=(1,)):
    max_k = max(top_k)
    batch_size = target.size(0)
    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    if len(res) == 1:
        res = res[0]
    return res


def entropy_loss(logits):
    return torch.distributions.categorical.Categorical(logits=logits).entropy().mean()


def kl_divergence(logits_p, logits_q):
    return torch.distributions.kl.kl_divergence(
        p=torch.distributions.categorical.Categorical(logits=logits_p),
        q=torch.distributions.categorical.Categorical(logits=logits_q)
    ).unsqueeze(1)


def js_divergence(logits_p, logits_q):
    p = torch.distributions.categorical.Categorical(logits=logits_p)
    q = torch.distributions.categorical.Categorical(logits=logits_q)
    m = torch.distributions.categorical.Categorical(probs=(p.probs + q.probs).detach() / 2)
    return (
        torch.distributions.kl.kl_divergence(p=p, q=m) + torch.distributions.kl.kl_divergence(p=q, q=m)
    ).unsqueeze(1) / 2


class Inception(nn.Module):
    def __init__(self, device):
        super(Inception, self).__init__()
        model = models.inception_v3(pretrained=True)
        self.f = nn.Sequential(
            nn.UpsamplingBilinear2d(size=299),
            model.Conv2d_1a_3x3,
            model.Conv2d_2a_3x3,
            model.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            model.Conv2d_3b_1x1,
            model.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            model.Mixed_5b,
            model.Mixed_5c,
            model.Mixed_5d,
            model.Mixed_6a,
            model.Mixed_6b,
            model.Mixed_6c,
            model.Mixed_6d,
            model.Mixed_6e,
            model.Mixed_7a,
            model.Mixed_7b,
            model.Mixed_7c,
            Flatten()
        ).cuda(device=device).eval()

    def forward(self, x):
        return self.f(x)


def inception_distance(x, x_hat):
    if not hasattr(inception_distance, "f"):
        inception_distance.f = Inception(x.device)
    with torch.no_grad():
        return torch.mean(
            (inception_distance.f(x) - inception_distance.f(x_hat)) ** 2,
            dim=1, keepdim=True
        )


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)


def conv_small():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=96),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=96),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=96),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=0.5, inplace=True),
        nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=192),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=192),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=192),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=0.5, inplace=True),
        nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=192),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, padding=0),
        nn.BatchNorm2d(num_features=192),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, padding=0),
        nn.BatchNorm2d(num_features=192),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.AdaptiveAvgPool2d(output_size=1),
        Flatten(),
        nn.Linear(in_features=192, out_features=10)
    )


def conv_large():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=128, ),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=128),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=128),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=0.5, inplace=True),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=256),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=256),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=256),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=0.5, inplace=True),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=512),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0),
        nn.BatchNorm2d(num_features=256),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0),
        nn.BatchNorm2d(num_features=128),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.AdaptiveAvgPool2d(output_size=1),
        Flatten(),
        nn.Linear(in_features=128, out_features=10)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir",                  default="/cache/logs/semisup/")
    parser.add_argument("--random_seed",  type=int,   default=0)
    parser.add_argument("--epochs",       type=int,   default=500)
    parser.add_argument("--iters",        type=int,   default=400)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--lambda_lp",    type=float, default=1)
    parser.add_argument("--lambda_em",    type=float, default=1)
    parser.add_argument("--eps_min",      type=float, default=0.1)
    parser.add_argument("--eps_max",      type=float, default=10)
    parser.add_argument("--xi",           type=float, default=10)
    parser.add_argument("--ip",           type=int,   default=1)
    parser.add_argument("--K",            type=float, default=0)
    parser.add_argument("--d_X",                      default="trivial", choices=["trivial", "l2", "inception", "msd"])
    parser.add_argument("--d_Y",                      default="kld",     choices=["kld", "jsd", "msd", "linf", "mad", "l2"])
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--cpu', dest='gpu', action='store_false')
    parser.set_defaults(gpu=True)
    args = parser.parse_args()
    print(args)

    # torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(args.random_seed)

    log_dir = args.log_dir + get_timestamp()
    os.makedirs(log_dir)
    print("Logging to:", log_dir)
    writer = SummaryWriter(logdir=log_dir)

    def upload(x):
        return x.cuda() if args.gpu else x

    model = conv_large()
    if args.gpu:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda i: 1 - i / (args.epochs * args.iters))

    dataloader_train, dataloader_reg, dataloader_valid, dataloader_test = get_dataloaders(batch_size=args.batch_size)

    if args.d_X == "trivial":
        d_X = lambda x, x_hat: torch.ones(x.size(0), 1, device=x.device)
    elif args.d_X == "l2":
        d_X = lambda x, x_hat: torch.norm((x - x_hat).view(x.size(0), -1), p=2, dim=1, keepdim=True)
    elif args.d_X == "inception":
        d_X = lambda x, x_hat: inception_distance(x, x_hat)
    elif args.d_X == "msd":
        d_X = lambda x, x_hat: torch.mean((x - x_hat).view(x.size(0), -1) ** 2, dim=1, keepdim=True)

    if args.d_Y == "kld":
        d_Y = lambda y, y_hat: kl_divergence(logits_p=y, logits_q=y_hat)
    elif args.d_Y == "jsd":
        d_Y = lambda y, y_hat: js_divergence(logits_p=y, logits_q=y_hat)
    elif args.d_Y == "msd":
        d_Y = lambda y, y_hat: torch.mean((y - y_hat) ** 2, dim=1, keepdim=True)
    elif args.d_Y == "linf":
        d_Y = lambda y, y_hat: torch.max(torch.abs(y - y_hat), dim=1, keepdim=True)[0]
    elif args.d_Y == "mad":
        d_Y = lambda y, y_hat: torch.mean(torch.abs(y - y_hat), dim=1, keepdim=True)
    elif args.d_Y == "l2":
        d_Y = lambda y, y_hat: torch.norm(y - y_hat, p=2, dim=1, keepdim=True)

    alr = ALR(d_X=d_X, d_Y=d_Y, lambda_lp=args.lambda_lp, eps_min=args.eps_min, eps_max=args.eps_max, xi=args.xi, ip=args.ip, K=args.K)

    lambda_em = args.lambda_em

    global_step = -1
    for epoch in range(args.epochs):
        t = tqdm(range(args.iters))
        model.train()
        for step in t:
            global_step += 1

            batch = next(dataloader_train)
            x, y = upload(batch[0]), upload(batch[1])
            x_reg = upload(next(dataloader_reg)[0])

            optimizer.zero_grad()

            x_reg_hat = alr.get_adversarial_perturbations(
                f=model, x=x_reg
            )

            y_hat = model(x)
            y_reg = model(x_reg)
            y_reg_hat = model(x_reg_hat)

            entropy = entropy_loss(logits=y_reg)
            
            nll_loss = F.nll_loss(F.log_softmax(y_hat, dim=1), y)
            acc = accuracy(F.softmax(y_hat.detach(), dim=1), y)

            alp_loss, lip_ratio, x_diff, y_diff, alp_l1, alp_l2, alp_count, nan_count, inf_count, neg_count = alp.get_alp_loss(
                x=x_reg, x_hat=x_reg_hat, y=y_reg.detach(), y_hat=y_reg_hat
            )

            loss = nll_loss + alp_loss + lambda_em * entropy

            loss.backward()

            optimizer.step()

            # scheduler.step()

            writer.add_scalar('train/loss'     , loss.item()       , global_step=global_step)
            writer.add_scalar('train/nll'      , nll_loss.item()   , global_step=global_step)
            writer.add_scalar('train/accuracy' , acc.item()        , global_step=global_step)
            writer.add_scalar('train/alp'      , alp_loss.item()   , global_step=global_step)
            writer.add_scalar('train/entropy'  , entropy.item()    , global_step=global_step)
            writer.add_scalar('train/lambda_lp', alr.lambda_lp     , global_step=global_step)
            writer.add_scalar('train/lambda_em', lambda_em         , global_step=global_step)
            writer.add_scalar('train/lr'       , scheduler.get_lr(), global_step=global_step)

            writer.add_scalar('lip_ratio/mean', torch.mean(lip_ratio).item(), global_step=global_step)
            writer.add_scalar('lip_ratio/std' , torch.std(lip_ratio).item() , global_step=global_step)
            writer.add_scalar('lip_ratio/min' , torch.min(lip_ratio).item() , global_step=global_step)
            writer.add_scalar('lip_ratio/max' , torch.max(lip_ratio).item() , global_step=global_step)
            writer.add_scalar('x_diff/mean'   , torch.mean(x_diff).item()   , global_step=global_step)
            writer.add_scalar('x_diff/std'    , torch.std(x_diff).item()    , global_step=global_step)
            writer.add_scalar('x_diff/min'    , torch.min(x_diff).item()    , global_step=global_step)
            writer.add_scalar('x_diff/max'    , torch.max(x_diff).item()    , global_step=global_step)
            writer.add_scalar('y_diff/mean'   , torch.mean(y_diff).item()   , global_step=global_step)
            writer.add_scalar('y_diff/std'    , torch.std(y_diff).item()    , global_step=global_step)
            writer.add_scalar('y_diff/min'    , torch.min(y_diff).item()    , global_step=global_step)
            writer.add_scalar('y_diff/max'    , torch.max(y_diff).item()    , global_step=global_step)
            writer.add_scalar('alp/alp_l1'    , alp_l1.item()               , global_step=global_step)
            writer.add_scalar('alp/alp_l2'    , alp_l2.item()               , global_step=global_step)
            writer.add_scalar('alp/count'     , alp_count                   , global_step=global_step)
            writer.add_scalar('alp/nan_count' , nan_count                   , global_step=global_step)
            writer.add_scalar('alp/inf_count' , inf_count                   , global_step=global_step)
            writer.add_scalar('alp/neg_count' , neg_count                   , global_step=global_step)

            desc = f"Epoch {epoch}/{args.epochs} [loss: {loss.item():.2f}] [acc: {acc.item():.2f}%]"
            t.set_description(desc)

        valid_acc = []
        model.eval()
        for step, batch in enumerate(iter(dataloader_valid)):
            x, y = upload(batch[0]), upload(batch[1])
            with torch.no_grad():
                y_hat = F.softmax(model(x), dim=1)
            valid_acc += [accuracy(y_hat, y)]
        valid_acc = torch.stack(valid_acc).mean()
        print(f"Validation accuracy: {valid_acc.item()}")
        writer.add_scalar('validation/accuracy', valid_acc.item(), global_step=global_step)

    test_acc = []
    model.eval()
    for step, batch in enumerate(iter(dataloader_test)):
        x, y = upload(batch[0]), upload(batch[1])
        with torch.no_grad():
            y_hat = F.softmax(model(x), dim=1)
        test_acc += [accuracy(y_hat, y)]
    test_acc = torch.stack(test_acc).mean()
    print(f"Test accuracy: {test_acc.item()}")
    writer.add_scalar('test/accuracy', test_acc.item(), global_step=global_step)


