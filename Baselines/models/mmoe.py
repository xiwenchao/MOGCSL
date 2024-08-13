import torch
from .layers import EmbeddingLayer, MultiLayerPerceptron


class MMoEModel(torch.nn.Module):
    def __init__(self, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, dropout):
        super().__init__()
        # self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.task_num = task_num
        self.embed_dim = embed_dim
        self.expert_num = expert_num

        self.expert = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(expert_num)])
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout, output_layer=True) for i in range(task_num)])
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.embed_dim, expert_num), torch.nn.Softmax(dim=1)) for i in range(task_num)])

    def forward(self, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        # numerical_emb = self.numerical_layer(numerical_x)
        gate_value = [self.gate[i](numerical_x).unsqueeze(1) for i in range(self.task_num)]
        fea = torch.cat([self.expert[i](numerical_x).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        
        results = [self.tower[i](task_fea[i]).squeeze(1) for i in range(self.task_num)]   # remove sigmoid
        return results