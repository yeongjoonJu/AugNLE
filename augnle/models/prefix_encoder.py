import torch


class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.prefix_len, config.hidden_size) # two tasks
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.hidden_size),
                torch.nn.Dropout(config.prefix_dropout),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.prefix_len, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def weight_initialization(self, embed_vector):
        embed_vector = embed_vector.expand(self.embedding.weight.shape[0], -1)
        self.embedding.weight.data.copy_(embed_vector)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.dropout(self.embedding(prefix))
            prompt_output = self.trans(prefix_tokens)
        else:
            prompt_output = self.dropout(self.embedding(prefix))
        return prompt_output