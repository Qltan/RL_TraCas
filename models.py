import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('current device: ', device)


class RLActor(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, input_size, hidden_size, output_size, state_size):
        super(RLActor, self).__init__()

        self.a1 = nn.Conv1d(input_size, hidden_size, kernel_size=1).to(device)  # conv1d_1
        self.a2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1).to(device)
        self.a3 = nn.Conv1d(hidden_size, 2, kernel_size=1).to(device)  # conv1d_1
        self.a4 = nn.Linear(state_size * 2, output_size).to(device)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, state):
        dynamic_hidden = self.a1(state)
        project_hidden = self.a2(dynamic_hidden)
        mid_hidden = self.a3(project_hidden)
        result_hidden = mid_hidden.view(mid_hidden.size(0), -1)
        result = self.a4(result_hidden)

        return result


class RLCritic(nn.Module):
    def __init__(self, input_size, hidden_size, linear_size):
        super(RLCritic, self).__init__()

        self.c1 = nn.Conv1d(input_size, hidden_size, kernel_size=1).to(device)
        self.c2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1).to(device)
        self.c3 = nn.Conv1d(hidden_size, 20, kernel_size=1).to(device)
        self.c4 = nn.Linear(20 * linear_size, 1).to(device)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, state):
        hidden = self.c1(state)
        output = F.relu(self.c2(hidden))
        output = F.relu(self.c3(output))
        output = output.view(output.size(0), -1)
        output = self.c4(output)
        return output


class RLCritic1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RLCritic1, self).__init__()
        self.c1 = nn.Conv1d(input_size, hidden_size, kernel_size=1).to(device)
        self.c2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1).to(device)
        self.c3 = nn.Conv1d(hidden_size, 20, kernel_size=1).to(device)
        self.c4 = nn.Conv1d(20, 1, kernel_size=1).to(device)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, state):
        output = F.relu(self.c1(state))
        output = F.relu(self.c2(output))
        output = F.relu(self.c3(output))
        # output = output.sum(dim=2)
        output = self.c4(output).sum()
        return output
