import torch

class ScaledClippedRelu(torch.nn.Module):
    def __init__(self, min_val, max_val, factor):
        super(ScaledClippedRelu, self).__init__()
        self.min_bound = min_val
        self.max_bound = max_val
        self.denomiator = factor

    def forward(self, x):
        divid_y = x // self.denomiator
        clipped_y1 = torch.max(divid_y, torch.ones_like(divid_y) * self.min_bound)
        clipped_y2 = torch.min(clipped_y1, torch.ones_like(clipped_y1) * self.max_bound)
        return clipped_y2


class ClippedRelu(torch.nn.Module):
    def __init__(self, min_val, max_val):
        super(ClippedRelu, self).__init__()
        self.min_bound = min_val
        self.max_bound = max_val

    def forward(self, x):
        clipped_y1 = torch.max(x, torch.ones_like(x) * self.min_bound)
        clipped_y2 = torch.min(clipped_y1, torch.ones_like(clipped_y1) * self.max_bound)
        return clipped_y2


class MyNnue(torch.nn.Module):

    def __init__(self, intput_dim, halfkp):
        super(MyNnue, self).__init__()
        self.input_dim = intput_dim
        self.halfkp = halfkp
        self.linear1 = torch.nn.Linear(self.input_dim, self.halfkp, True)
        self.linear2 = torch.nn.Linear(512, 32, True)
        self.linear3 = torch.nn.Linear(32, 32, True)
        self.linear4 = torch.nn.Linear(32, 1, True)
        self.clipped_relu = ClippedRelu(0, 127)
        self.scale_clip_relu = ScaledClippedRelu(0, 127, 64)

    def forward(self, x):
        x1 = x[:, 0:self.input_dim]                     # first half
        x2 = x[:, self.input_dim:(self.input_dim * 2)]  # second half
        y11 = self.clipped_relu.forward(self.linear1(x1))
        y12 = self.clipped_relu.forward(self.linear1(x2))
        y1 = torch.cat((y11, y12), 1)
        y2 = self.scale_clip_relu.forward(self.linear2(y1))
        y3 = self.scale_clip_relu.forward(self.linear3(y2))
        y4 = self.linear4(y3)
        return y4


if __name__ == '__main__':
    mynet = MyNnue(41024, 256)
