from torch import nn


class SimpleRNN(nn.Module):

    def __init__(self, rnn_type, input_size, hidden_size, num_layers):
        super(SimpleRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Initialise the correct RNN layer depending on what we.
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, dropout=(0 if num_layers == 1 else 0.05), num_layers=num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, dropout=(0 if num_layers == 1 else 0.05), num_layers=num_layers, batch_first=True)
        else:
            raise(ValueError('Incorrect choice of RNN supplied'))
        self.out = nn.Linear(hidden_size, 1)  # Linear layer is output of model

    def forward(self, x, h_state):
        # Define our forward pass, we take some input sequence and an initial hidden state.
        r_out, h_state = self.rnn(x, h_state)

        final_y = self.out(r_out[:, -1, :])  # Return only the last output of RNN.

        return final_y, h_state