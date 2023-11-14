# Imports
import torch
import torch.nn as nn
import torch.nn.functional as functional


class GCNArgs:
    def __init__(self):

        # The hidden layer sizes as a list
        # For example, three hidden layers of 64, 128, 64 would be placed here
        self.hidden_sizes = []

        # The number of classes to classify into
        self.num_classes = 0

        # The ratio of the raw data given to test data
        self.test_ratio = 0

        # The number of epochs to train for
        self.num_epochs = 0

        # The learning rate of the model
        self.learning_rate = 0

        # The model id
        self.model_id = 0


class GCN(nn.Module):
    def __init__(self, x_size, a_hat, args: GCNArgs, bias=True):  # X_size = num features
        super(GCN, self).__init__()

        # Configure parameters
        self.A_hat = torch.tensor(a_hat, requires_grad=False).float()

        # Handle the creation of the weights and biases
        self.weights: list[nn.parameter.Parameter] = [nn.parameter.Parameter() for _ in args.hidden_sizes]
        self.biases: list[nn.parameter.Parameter] = [nn.parameter.Parameter() for _ in args.hidden_sizes]
        all_sizes = [x_size, *args.hidden_sizes, args.num_classes, ]
        for w in range(0, len(args.hidden_sizes)):

            # weights
            self.weights[w] = nn.parameter.Parameter(torch.FloatTensor(all_sizes[w], all_sizes[w + 1]))
            var = 2. / (self.weights[w].size(1) + self.weights[w].size(0))
            self.weights[w].data.normal_(0, var)

            # biases
            if bias:
                self.biases[w] = nn.parameter.Parameter(torch.FloatTensor(all_sizes[w + 1]))
                self.biases[w].data.normal_(0, var)

        self.args = args
        self.has_bias = bias
        self.fc1 = nn.Linear(args.hidden_sizes[-1], args.num_classes)  # final layer

    def forward(self, vals):  # Variable layer architecture

        # Loop through the layers
        for li in range(len(self.args.hidden_sizes)):
            vals = torch.mm(vals, self.weights[li])
            if self.has_bias:
                vals = (vals + self.biases[li])
            vals = functional.relu(torch.mm(self.A_hat, vals))

        # Return the forward of the final layer
        return self.fc1(vals)
