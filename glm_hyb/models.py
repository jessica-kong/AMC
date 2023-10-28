import torch


# class PrefixEncoder(torch.nn.Module):
#     """
#     The torch.nn model to encode the prefix
#     Input shape: (batch-size, prefix-length)
#     Output shape: (batch-size, prefix-length, 2*layers*hidden)
#     """
#
#     def __init__(self, config, prompt_ids, embedding_layer: torch.nn.Module):
#         super().__init__()
#         prompt_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
#
#         with torch.no_grad():
#             self.embedded: torch.Tensor = embedding_layer(prompt_ids)
#         embedding_layer.zero_grad()
#
#         self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
#         self.linear.weight.data.fill_(1.0)
#         self.linear.bias.data.fill_(0.0)
#
#     def forward(self, prefix: torch.Tensor):
#         batch_size = prefix.size(0)
#         init_embedded = self.embedded.repeat(batch_size, 1, 1).to(prefix.device)
#         init_embedded = init_embedded.float()
#         past_key_values = self.linear(init_embedded)
#         return past_key_values
#

class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config, prompt_ids, embedding_layer: torch.nn.Module):
        super().__init__()
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            self.embedded: torch.Tensor = embedding_layer(prompt_ids)
        embedding_layer.zero_grad()

        kv_size = config.num_layers * config.kv_channels * config.multi_query_group_num * 2
        self.linear = torch.nn.Linear(config.hidden_size, kv_size)
        self.linear.weight.data.fill_(1.0)
        self.linear.bias.data.fill_(0.0)
        # self.tanh = torch.nn.Tanh()
        # self.linear2 = torch.nn.Linear(kv_size, config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        batch_size = prefix.size(0)
        init_embedded = self.embedded.repeat(batch_size, 1, 1).to(prefix.device)
        init_embedded = init_embedded.float()
        past_key_values = self.linear(init_embedded)
        # past_key_values = self.tanh(past_key_values)
        # past_key_values = self.linear2(past_key_values)

        return past_key_values
