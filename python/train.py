import os
import copy

import numpy as np
import onnx
import onnxoptimizer
from onnxsim import simplify

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

C1, C2 = 8, 16
FC_IN = C2 * 6 * 6
BATCH, SEED = 128, 15
MIN_EPOCHS, MAX_EPOCHS, TARGET_ACC = 100, 200, 0.9
ROOT, OUT = "fashion", "export"


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, C1, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(C1)
        self.conv2 = nn.Conv2d(C1, C2, 3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(C2)
        self.fc = nn.Linear(FC_IN, 10)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.bn1(self.conv1(x)), 0.1), 2)
        x = F.max_pool2d(F.leaky_relu(self.bn2(self.conv2(x)), 0.1), 2)
        return self.fc(x.flatten(1))


@torch.no_grad()
def logits_of(net, x, chunk=1000):
    return torch.cat([net(x[i:i + chunk]) for i in range(0, len(x), chunk)])


def train(net, dev, train_loader, test_loader):
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)
    best_acc, best_state = 0.0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            F.cross_entropy(net(xb.to(dev)), yb.to(dev)).backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            hit = sum((net(xb.to(dev)).argmax(1).cpu() == yb).sum().item()
                      for xb, yb in test_loader)
        acc = hit / len(test_loader.dataset)
        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(net.state_dict())
        print(f"epoch {epoch:3d}  test acc {acc:.4f}  best {best_acc:.4f}")
        if epoch >= MIN_EPOCHS and acc >= TARGET_ACC:
            break
    net.load_state_dict(best_state)


def main():
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)

    train_raw = datasets.FashionMNIST(ROOT, train=True, download=True)
    test_raw = datasets.FashionMNIST(ROOT, train=False, download=True)

    px = train_raw.data.to(torch.float32).div_(255.0)
    mean = float(np.float32(px.mean().item()))
    std = float(np.float32(px.std().item()))
    print(f"mean={mean!r} std={std!r}")

    norm = transforms.Normalize((mean,), (std,))
    train_loader = DataLoader(
        datasets.FashionMNIST(ROOT, train=True, transform=transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm
        ])),
        batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(
        datasets.FashionMNIST(ROOT, train=False, transform=transforms.Compose([
            transforms.ToTensor(), norm
        ])), batch_size=1000)

    def prep(u8):
        x = u8.to(torch.float32).div_(255.0)
        return x.sub_(mean).div_(std).unsqueeze(1)

    x_all = torch.cat([prep(train_raw.data), prep(test_raw.data)])
    y_all = torch.cat([train_raw.targets, test_raw.targets]).to(torch.uint8)

    # 1. train
    net = Net().to(dev)
    train(net, dev, train_loader, test_loader)
    net = net.cpu().eval()
    ref = logits_of(net, x_all)
    print(f"test acc {(ref[60000:].argmax(1) == y_all[60000:].long()).float().mean():.4f}")

    # 2. export models
    dummy = torch.zeros(1, 1, 28, 28)
    torch.jit.trace(net, dummy).save(f"{OUT}/model.pt")
    torch.onnx.export(net, dummy, f"{OUT}/model.onnx",
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
                      opset_version=13, dynamo=False)
    m, ok = simplify(onnx.load(f"{OUT}/model.onnx"))
    assert ok, "onnxsim could not validate the simplified model"
    m = onnxoptimizer.optimize(m, onnxoptimizer.get_fuse_and_elimination_passes())
    onnx.save(m, f"{OUT}/model.onnx")
    print("onnx nodes:", [n.op_type for n in m.graph.node])

    net.conv1 = fuse_conv_bn_eval(net.conv1, net.bn1)
    net.conv2 = fuse_conv_bn_eval(net.conv2, net.bn2)
    net.bn1, net.bn2 = nn.Identity(), nn.Identity()
    err = (logits_of(net, x_all[:2000]) - ref[:2000]).abs().max()
    assert err < 1e-4, f"conv+bn fusion changed the outputs: {err}"
    print(f"conv+bn fusion max diff {err:.2e}")

    tensors = [net.conv1.weight, net.conv1.bias, net.conv2.weight, net.conv2.bias,
               net.fc.weight, net.fc.bias]
    blob = [t.detach().numpy().astype("<f4").ravel() for t in tensors]
    np.concatenate(blob).tofile(f"{OUT}/model.bin")

    # 3. save data
    x_all.numpy().astype("<f4", copy=False).tofile(f"{OUT}/images.bin")
    ref.numpy().astype("<f4").tofile(f"{OUT}/ref_logits.bin")
    print(f"wrote {OUT}/: {sorted(os.listdir(OUT))}")


if __name__ == "__main__":
    main()
