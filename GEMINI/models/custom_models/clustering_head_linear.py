from torch import nn
from source.runner_parameters import RunnerParameters

def get_model(input_shape=2, num_clusters=5, **kwargs):
	return nn.Sequential(nn.Linear(input_shape, num_clusters), nn.Softmax(dim=-1))
