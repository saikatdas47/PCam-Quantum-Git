"""Exact real-valued statevector simulation of the declared RY/CNOT circuit."""
import torch


def expectations(inputs, weights, cnot):
    """Wire zero is the most significant bit; all states remain real."""
    q = inputs.shape[-1]
    state = inputs.new_zeros((len(inputs), 2 ** q))
    state[:, 0] = 1

    def ry(state, angle, wire):
        shaped = state.reshape(len(inputs), 2 ** wire, 2, 2 ** (q-wire-1))
        a, b = shaped[:, :, 0, :], shaped[:, :, 1, :]
        c = torch.cos(angle / 2).reshape(-1, 1, 1)
        s = torch.sin(angle / 2).reshape(-1, 1, 1)
        return torch.stack((c*a-s*b, s*a+c*b), dim=2).reshape(len(inputs), -1)

    for wire in range(q):
        state = ry(state, inputs[:, wire], wire)
    index = torch.arange(2 ** q, device=inputs.device)
    for layer in range(2):
        for wire in range(q):
            state = ry(state, weights[layer, wire], wire)
        if cnot:
            for wire in range(q-1):
                control = 1 << (q-wire-1)
                target = 1 << (q-wire-2)
                permutation = index ^ (((index & control) != 0).long() * target)
                state = state[:, permutation]
    signs = torch.stack([1-2*((index >> (q-wire-1)) & 1) for wire in range(q)], dim=1)
    return state.square() @ signs.to(inputs.dtype)


class TwoLayerRYHead(torch.nn.Module):
    """Two RY layers, identical off/on parameterization, affine Z readout."""
    def __init__(self, dimensions, cnot):
        super().__init__()
        if dimensions not in (4, 8):
            raise ValueError('Only declared 4/8-dimensional circuits are supported')
        self.dimensions = dimensions
        self.cnot = bool(cnot)
        self.weights = torch.nn.Parameter(torch.randn(2, dimensions, dtype=torch.float64)*0.1)
        self.readout = torch.nn.Linear(dimensions, 1, dtype=torch.float64)

    def forward(self, inputs):
        if inputs.dtype != torch.float64 or inputs.device.type != 'cpu':
            raise ValueError('Matched head training uses CPU float64')
        return self.readout(expectations(inputs, self.weights, self.cnot)).squeeze(-1)


class MLP(torch.nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(inputs, hidden, dtype=torch.float64), torch.nn.Tanh(), torch.nn.Linear(hidden, 1, dtype=torch.float64))
    def forward(self, inputs):
        return self.net(inputs).squeeze(-1)
