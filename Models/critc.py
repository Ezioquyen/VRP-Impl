import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.base_models import Encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AttentionCritic(nn.Module):
    def __init__(self, static_size, dynamic_size, hidden_size):
        super(AttentionCritic, self).__init__()
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.static_encoder = Encoder(static_size, hidden_size)
        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, hy, static, dynamic):
        static_hidden = self.static_encoder(static)
        dynamic_ld = dynamic[:, 1].unsqueeze(1)
        dynamic_hidden = self.dynamic_encoder(dynamic_ld.expand_as(dynamic))
        batch_size, hidden_size, seq = static_hidden.size()

        hidden = hy.unsqueeze(2).expand(batch_size, hidden_size, seq)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)
        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        prob = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        return static_hidden, prob


class StateCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()
        self.attention = AttentionCritic(static_size, dynamic_size, hidden_size)
        self.dense = nn.Linear(128, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.relu = torch.relu
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):
        batch, _, _ = static.size()
        hy = torch.zeros(batch, 128, device=device)
        for i in range(3):
            e, logit = self.attention(hy, static, dynamic)
            prob = F.softmax(logit, dim=-1)
            hy = torch.matmul(prob, e.transpose(1,2)).squeeze(1)
        output = self.dense(hy)
        output = self.relu(output)
        output = self.linear(output)
        return output


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):

        output = F.relu(self.fc1(input.unsqueeze(1)))
        output = F.relu(self.fc2(output)).squeeze(2)
        output = self.fc3(output).sum(dim=2)
        return output
