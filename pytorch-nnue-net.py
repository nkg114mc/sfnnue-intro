import numpy as np
import torch
import pandas as pd

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

    def __init__(self, halfkp):
        super(MyNnue, self).__init__()
        self.halfkp = halfkp
        self.linear1 = torch.nn.Linear(self.halfkp, 256, True)
        self.linear2 = torch.nn.Linear(512, 32, True)
        self.linear3 = torch.nn.Linear(32, 32, True)
        self.linear4 = torch.nn.Linear(32, 1, True)
        self.clipped_relu = ClippedRelu(0, 127)
        self.scale_clip_relu = ScaledClippedRelu(0, 127, 64)

    def forward(self, x):
        x1 = x[:, 0:self.halfkp] # first half
        x2 = x[:, self.halfkp:(self.halfkp * 2)]  # second half
        y11 = self.clipped_relu.forward(self.linear1(x1))
        y12 = self.clipped_relu.forward(self.linear1(x2))
        y1 = torch.cat((y11, y12), 1)
        y2 = self.scale_clip_relu.forward(self.linear2(y1))
        y3 = self.scale_clip_relu.forward(self.linear3(y2))
        y4 = self.linear4(y3)
        print("y4=", y4[0, :])
        return y4

    def read_tensor_from_csv(self, csv_fn, dtype):
        df = pd.read_csv(csv_fn, header=None).T
        print('csv shape: ', df.shape)
        ts = torch.tensor(df.values, dtype=dtype)
        return ts

    def read_from_file(self, folder_name):

        init_w0 = self.read_tensor_from_csv(folder_name + 'w0.csv', torch.float)#torch.int16)
        init_b0 = torch.squeeze(self.read_tensor_from_csv(folder_name + 'b0.csv', torch.float))#torch.int16)
        init_w1 = self.read_tensor_from_csv(folder_name + 'w1.csv', torch.float)#torch.int8)
        init_b1 = torch.squeeze(self.read_tensor_from_csv(folder_name + 'b1.csv', torch.float))#torch.int32)
        init_w2 = self.read_tensor_from_csv(folder_name + 'w2.csv', torch.float)#torch.int8)
        init_b2 = torch.squeeze(self.read_tensor_from_csv(folder_name + 'b2.csv', torch.float))#torch.int32)
        init_w3 = self.read_tensor_from_csv(folder_name + 'w3.csv', torch.float)#torch.int8)
        init_b3 = torch.squeeze(self.read_tensor_from_csv(folder_name + 'b3.csv', torch.float), 1)#torch.int32)

        self.linear1.weight.data = init_w0
        self.linear1.bias.data = init_b0
        self.linear2.weight.data = init_w1
        self.linear2.bias.data = init_b1
        self.linear3.weight.data = init_w2
        self.linear3.bias.data = init_b2
        self.linear4.weight.data = init_w3
        self.linear4.bias.data = init_b3


def run_forward():

    mynet = MyNnue(41024)

    print(mynet.linear1.weight.shape)
    print(mynet.linear1.bias.shape)
    print(mynet.linear2.weight.shape)
    print(mynet.linear2.bias.shape)
    print(mynet.linear3.weight.shape)
    print(mynet.linear3.bias.shape)
    print(mynet.linear4.weight.shape)
    print(mynet.linear4.bias.shape)

    mynet.read_from_file('/Users/chaom/workplace/aboutchess/nnchess/Stockfish-NNUE-2020-07-18/src/eval/csv_params/')

    print(mynet.linear1.weight.shape)
    print(mynet.linear1.bias.shape)
    print(mynet.linear2.weight.shape)
    print(mynet.linear2.bias.shape)
    print(mynet.linear3.weight.shape)
    print(mynet.linear3.bias.shape)
    print(mynet.linear4.weight.shape)
    print(mynet.linear4.bias.shape)

    #x = torch.randn(10, 82048)#, dtype=torch.int32)
    #y = mynet.forward(x)
    #print(y)

    df2 = pd.read_csv('/Users/chaom/workplace/aboutchess/nnchess/Stockfish-NNUE-2020-07-18/src/eval/feat100.csv', header=None)
    print(df2.shape)
    ts2 = torch.tensor(df2.values, dtype=torch.float)
    x2 = ts2[:, 0:82048]
    y2 = ts2[:, 82048:82049]

    print('x2=', x2)

    print("====")
    print(x2.shape)
    print(y2.shape)
    y2_pred = mynet.forward(x2)

    print('y2 = ', y2)
    print('y2_pred = ', y2_pred)

    #for i in range(0, 64):
    #    j = i * 641 + 35
    #    print(mynet.linear1.weight[:, j:(j + 1)])


'''
def run_torch_train():

    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    layers_dim = [2, 20, 15, 2]

    print(X)
    print(y)

    model = mynn()

    y_truth_onehot = np.zeros([len(y), 2])
    for j in range(0, len(y)):
        crrIdx = y[j]
        y_truth_onehot[j][crrIdx] = 1.0

    learning_rate = 0.01
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    for t in range(2000):
        # Forward pass: Compute predicted y by passing x to the model
        xtsr = torch.from_numpy(X).float()
        #print(xtsr)
        y_pred = model.forward(xtsr)
        print(y_pred)

        # Compute and print loss
        loss = criterion(y_pred, torch.from_numpy(y_truth_onehot).float())
        print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # Plot the decision boundary
    plot_decision_boundary(lambda x: model.predict(torch.from_numpy(x).float()), X, y)
    plt.title("Decision Boundary for my NN")
    plt.show()
'''

if __name__ == '__main__':
    run_forward()
