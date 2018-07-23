from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.optim import lr_scheduler
from torch.autograd import Variable

def squash(x, dim):
    # if primarycaps: (B, 6*6*32, 8) --(axis=2)--> (B, 6*6*32, 8)
    # if digitcaps: (B, 1, 10, 16, 1) --(axis=3)--> (B, 1, 10, 16, 1)
    lengths2 = x.pow(2).sum(dim=dim,keepdim=True)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths)
    return x

class PrimaryCapsLayer(nn.Module):
    def __init__(self, device,input_channels, output_channel, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channel * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_channel = output_channel
        self.output_dim = output_dim
        self.device = device

    def forward(self, input):
        out = self.conv(input)
        B, C, H, W = out.size()
        out = out.view(B, self.output_channel, self.output_dim, H, W)
        # (B, 32, 8, 6, 6) --> (B, 32, 6, 6, 8) 
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        # (B, 32, 6, 6, 8) --> (B, 32*6*6, 8)
        out = out.view(out.size(0), -1, out.size(4))
        ui = squash(out, dim=2)
        return ui

class CapsLayer(nn.Module):
    def __init__(self, device,input_caps, input_dim, output_caps, output_dim, n_iterations):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(1, input_caps, output_caps, output_dim, input_dim))
        self.reset_parameters()
        self.n_iterations = n_iterations
        self.device = device

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, input):
        b_i = nn.Parameter(torch.zeros((1, self.input_caps, self.output_caps, 1, 1)))
        b_i = b_i.to(self.device)
        # input:(B, 1152,8 ) --> u_i:(B,1152,1,8,1)
        u_i = input.unsqueeze(2).unsqueeze(4)
        # u_i:(B,1152,1,8,1) --> u_i:(B,1152,10,8,1)
        u_i = u_i.expand(-1,-1,10,-1,-1)
        # u_i:(B,1152,10,8,1) --> u_ji = (B,1152,10,16,1)
        u_ji = torch.matmul(self.weights,u_i)

        for i in range(self.n_iterations):
            # c_i: (B,1152,10,1,1)
            c_i = F.softmax(b_i,dim=2)
            # c_imu_ji :(B,1152,10,16,1)
            c_imu_ji = c_i*u_ji
            # s_j: (B, 1, 10, 16, 1)
            s_j = c_imu_ji.sum(dim=1,keepdim=True)
            # v_j: (B, 1, 10, 16, 1)
            v_j = squash(s_j,dim=3)
            # b_i: (B, 1152, 10, 16,1 )
            b_i = b_i + u_ji*v_j
        
        # return (B, 10, 16)
        return v_j.squeeze()

class CapsNet(nn.Module):
    def __init__(self,device, routing_iterations, n_classes=10):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primaryCaps = PrimaryCapsLayer(device, 256, 32, 8, kernel_size=9, stride=2)  # outputs 6*6
        self.num_primaryCaps = 32 * 6 * 6
        self.digitCaps = CapsLayer(device, self.num_primaryCaps, 8, n_classes, 16, routing_iterations)
        self.device = device

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.primaryCaps(x)
        x = self.digitCaps(x)
        probs = x.pow(2).sum(dim=2).sqrt()
        return x, probs

class ReconstructionNet(nn.Module):
    def __init__(self, device ,n_dim=16, n_classes=10):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.device = device

    def forward(self, x, target):
        mask = Variable(torch.zeros((x.size()[0], self.n_classes)), requires_grad=False)
        mask = mask.to(self.device)
        mask.scatter_(1, target.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        x = x * mask
        x = x.view(-1, self.n_dim * self.n_classes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class CapsNetWithReconstruction(nn.Module):
    def __init__(self, device, capsnet, reconstruction_net):
        super(CapsNetWithReconstruction, self).__init__()
        self.capsnet = capsnet
        self.reconstruction_net = reconstruction_net
        self.device = device

    def forward(self, x, target):
        x, probs = self.capsnet(x)
        reconstruction = self.reconstruction_net(x, target)
        return reconstruction, probs


class MarginLoss(nn.Module):
    def __init__(self, device, m_pos, m_neg, lambda_):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_
        self.device = device

    def forward(self, lengths, targets, size_average=True):
        t = torch.zeros(lengths.size()).long()
        t = t.to(self.device)
        t = t.scatter_(1, targets.data.view(-1, 1), 1)
        targets = Variable(t)
        losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + self.lambda_ * (1. - targets.float()) * F.relu(lengths - self.m_neg).pow(2)
        return losses.mean() if size_average else losses.sum()


if __name__ == '__main__':
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable
    import argparse

    parser =  argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--lt', type=str, default='3', help='1, 3, 5')
    parser.add_argument('--if_reconstruct', type=bool, default=True, help='True, False')
    gpu_ids = parser.parse_args().gpu_ids
    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    # set gpu ids
    if len(gpu_ids) > 0:
        use_cuda = True
        torch.cuda.set_device(gpu_ids[0])
        device = torch.device('cuda:{}'.format(gpu_ids[0]))
    else:
        use_cuda = False
        device = torch.device('cpu')
    
    batch_size = 128
    epoch = 300
    lr = 0.001
    routr_lt = int(parser.parse_args().lt)
    if_reconstruct = parser.parse_args().if_reconstruct

    print("#########################################")
    print("batch size:{0}".format(batch_size))
    print("lr:{0}".format(lr))
    print("routr_lt:{0}".format(routr_lt))
    print("if_reconstruct:"+str(if_reconstruct))
    print("#########################################")

    seed = 3
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)

    kwargs = {}
    if use_cuda:
        kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(2), transforms.RandomCrop(28),
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=batch_size, shuffle=False, **kwargs)

    model = CapsNet(device, routr_lt)

    if if_reconstruct:
        reconstruction_model = ReconstructionNet(device,16, 10)
        reconstruction_alpha = 0.0005
        model = CapsNetWithReconstruction(device, model, reconstruction_model)

    if use_cuda:
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)

    loss_fn = MarginLoss(device, 0.9, 0.1, 0.5)


    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target, requires_grad=False)
            optimizer.zero_grad()
            if if_reconstruct:
                output, probs = model(data, target)
                reconstruction_loss = F.mse_loss(output, data.view(-1, 784))
                margin_loss = loss_fn(probs, target)
                loss = reconstruction_alpha * reconstruction_loss + margin_loss
            else:
                output, probs = model(data)
                loss = loss_fn(probs, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if use_cuda:
                data, target = data.to(device), target.to(device)
            data, target = Variable(data, volatile=True), Variable(target)

            if if_reconstruct:
                output, probs = model(data, target)
                reconstruction_loss = F.mse_loss(output, data.view(-1, 784), size_average=False).data[0]
                test_loss += loss_fn(probs, target, size_average=False).data[0]
                test_loss += reconstruction_alpha * reconstruction_loss
            else:
                output, probs = model(data)
                test_loss += loss_fn(probs, target, size_average=False).data[0]

            pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return test_loss


    for epoch in range(1, epoch + 1):
        train(epoch)
        test_loss = test()
        scheduler.step(test_loss)
