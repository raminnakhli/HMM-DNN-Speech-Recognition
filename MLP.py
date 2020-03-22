import torch
from torch import nn
from torch.autograd import Variable


class NeuralNetwork(nn.Module):

    def __init__(self, feature_size, class_count):
        super(NeuralNetwork, self).__init__()

        mid1_neuron = 40
        mid2_neuron = 30
        mid3_neuron = 20

        self.layer1 = nn.Sequential(
            nn.Linear(feature_size, mid1_neuron)
        )

        self.layer1_post = nn.Sequential(
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(mid1_neuron, mid2_neuron)
        )

        self.layer2_post = nn.Sequential(
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(mid2_neuron, mid3_neuron)
        )

        self.layer3_post = nn.Sequential(
            nn.LeakyReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(mid3_neuron, class_count)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer1_post(out)
        out = self.layer2(out)
        out = self.layer2_post(out)
        out = self.layer3(out)
        out = self.layer3_post(out)
        out = self.layer4(out)
        return out


class MLP:

    def info(self):
        print('trained: ', self.trained)

    def __init__(self, feature_size, class_count):
        self.net = NeuralNetwork(feature_size, class_count)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.RMSprop(self.net.parameters())
        self.epoch_count = 20
        self.trained = False

    def train(self, data, label, epoch=None):
        self.trained = True
        self.net.train()
        number_of_epuchs = self.epoch_count if epoch is None else epoch
        for epoch in range(number_of_epuchs):

            for i, (batch_data, batch_label) in enumerate(zip(data, label)):
                batch_data = batch_data.reshape(1, -1)

                batch_data = Variable(torch.from_numpy(batch_data)).float()
                batch_label = Variable(torch.from_numpy(batch_label)).long()

                self.optimizer.zero_grad()
                score = self.net(batch_data)

                loss = self.loss_function(score, batch_label)

                loss.backward()
                self.optimizer.step()

    def log_probablity(self, data):
        self.net.eval()

        data = Variable(torch.from_numpy(data)).float()

        scores = self.net(data)
        softmax_module = nn.LogSoftmax()
        prob = softmax_module(scores)
        return prob.data.numpy()

    def predict(self, data):
        self.net.eval()

        data = Variable(torch.from_numpy(data)).float()

        scores = self.net(data)
        _, predicted = torch.max(scores.data, 1)
        return predicted
