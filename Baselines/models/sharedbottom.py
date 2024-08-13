import torch
from .layers import EmbeddingLayer, MultiLayerPerceptron


class SharedBottomModel(torch.nn.Module):
    """
    A pytorch implementation of Shared-Bottom Model.
    """

    def __init__(self, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, dropout):
        super().__init__()
        # self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.task_num = task_num

        self.bottom = MultiLayerPerceptron(embed_dim, bottom_mlp_dims, dropout, output_layer=False)
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout, output_layer=True) for i in range(task_num)])

    def forward(self, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        fea = self.bottom(numerical_x)

        results = [self.tower[i](fea) for i in range(self.task_num)]
        return results