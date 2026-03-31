# =========================
# IMPORTS
# =========================
import torch
import torch.nn as nn


# =========================
# LSTM REGRESSION MODEL
# =========================
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """
        Args:
            input_size (int): number of features
            hidden_size (int): number of hidden units
            num_layers (int): number of LSTM layers
            dropout (float): dropout rate
        """
        super(LSTMRegressor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # Mutiple LSTM Layers 
        )

        # Fully connected layer (output)
        self.fc = nn.Linear(hidden_size, 1)  # regression → 1 output

    def forward(self, x):
        """
        x shape: (batch_size, time_steps, input_size)
        """

        # Initialize hidden state (h0) and cell state (c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass through LSTM
        # out = All hidden states (batch, time_steps, hidden_size)
        # hn = Final hidden state (num_layers, batch, hidden_size)
        # cn = Final hidden state (num_layers, batch, hidden_size)

        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Take last time step output
        out = out[:, -1, :]   # (batch_size, hidden_size)

        # Pass through fully connected layer
        out = self.fc(out)    # (batch_size, 1)

        return out
