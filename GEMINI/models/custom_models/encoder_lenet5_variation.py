from torch import nn


def get_model(in_channels=1,out_features=84,**kwargs):
    return nn.Sequential(            
            nn.Conv1d(in_channels=in_channels, out_channels=6, kernel_size=49, stride=1),  # 500 - 2*24 =  452
            nn.Tanh(),
            nn.AvgPool1d(kernel_size=2),  # 452/2 = 226
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=49, stride=1), # 226 - 2*24 = 178
            nn.Tanh(),
            nn.AvgPool1d(kernel_size=2), # 178/2 = 89
            nn.Conv1d(in_channels=16, out_channels=120, kernel_size=89, stride=1), # Aggregate rest
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(in_features=120, out_features=out_features),
            nn.Tanh(),
        )