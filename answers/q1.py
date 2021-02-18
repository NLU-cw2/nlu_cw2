import torch
import torch.nn as nn


def q1a():
    """
    Describe what happens when self.bidirectional is set to True.
    What is the difference between final_hidden_states and final_cell_states?

    When bidirectional is True, the LSTM is constructed with two LSTM submodules.
    These submodules have the same size.
    One of them takes the original input and the other one takes the input in the reversed order.
    The outputs from these two submodules will be concatenated together for further prediction.

    cell_states is responsible for tracing history, while hidden_states is responsible for generating output.

    """

    batch_size = 4
    seq_len = 5
    emb_dim = 6
    h_dim = 7

    packed_source_embeddings = torch.randn(seq_len, batch_size, emb_dim)

    for bidirectional in (False, True):
        print(f'\n\ntesting bidirectional={bidirectional}')
        lstm = nn.LSTM(input_size=emb_dim, hidden_size=h_dim, bidirectional=bidirectional)
        (hidden_initial, context_initial) = torch.zeros(2 if bidirectional else 1, batch_size, h_dim), torch.zeros(2 if bidirectional else 1, batch_size, h_dim)
        packed_outputs, (final_hidden_states, final_cell_states) = lstm(packed_source_embeddings, (hidden_initial, context_initial))

        print('\nlstm parameters sizes:')
        for name, parameters in lstm.named_parameters():
            print(name, parameters.shape)

        print(f'\npacked_outputs:{packed_outputs.shape}, final_hidden_states:{final_hidden_states.shape}, final_cell_states:{final_cell_states.shape}')

        if not bidirectional:
            print(packed_outputs[-1] == final_hidden_states[0])
        else:
            print(packed_outputs[-1, :, :7] == final_hidden_states[0])
            print(packed_outputs[0, :, 7:] == final_hidden_states[1])


if __name__ == '__main__':
    q1a()
