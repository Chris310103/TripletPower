import torch
import torch.nn as nn

class CNN_Best(nn.Module):
    """
    The same backbone as the one used in the triplet network, ported from TensorFlow to PyTorch.
    """
    def __init__(self, input_channels: int = 1, input_length: int = 700, embed_size: int = 256):
        super().__init__()

        self.input_channels=input_channels
        self.input_length=input_length
        self.output_dim=embed_size

        self.feature_extractor=nn.Sequential(
            # block 1
            nn.ConstantPad1d((4,5),0),
            nn.Conv1d(input_channels, 64, kernel_size=11, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2),

            # block 2
            nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=5),
            nn.ReLU(inplace=True),  
            nn.AvgPool1d(kernel_size=2, stride=2),

            # block 3
            nn.Conv1d(128, 256, kernel_size=11, stride=1, padding=5),
            nn.ReLU(inplace=True),  
            nn.AvgPool1d(kernel_size=2, stride=2),

            # block 4
            nn.Conv1d(256, 512, kernel_size=11, stride=1, padding=5),
            nn.ReLU(inplace=True),  
            nn.AvgPool1d(kernel_size=2, stride=2),

            # block 5
            nn.Conv1d(512, 512, kernel_size=11, stride=1, padding=5),
            nn.ReLU(inplace=True),  
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            dummy_input=torch.zeros(1, self.input_channels, self.input_length)
            dummy_output=self.feature_extractor(dummy_input)
            flatten_size=dummy_output.view(1, -1).size(1)

            self.flatten_size=flatten_size

        self.embed_head=nn.Sequential(
            nn.Flatten(),
            # first layer
            nn.Linear(flatten_size, 4096),
            nn.ReLU(inplace=True),

            # second layer
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            # third layer
            nn.Linear(4096, self.output_dim),
            nn.ReLU(inplace=True),
        )

        # keras-stile weight initialization
        self.apply(self.init_keras_weights)

    def init_keras_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def to_channels_first(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 3:
                raise ValueError(
                    f"Expected 3D input, got {tuple(x.shape)}"
                )
    
            if x.shape[-1] == self.input_channels:
                return x.transpose(1, 2)
    
            if x.shape[1] == self.input_channels:
                return x
    
            raise ValueError(
                f"Unexpected input shape: {tuple(x.shape)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.to_channels_first(x)

        h=self.feature_extractor(x)
        h=h.transpose(1, 2)

        y=self.embed_head(h)

        return y

def build_cnn_best(input_shape, emb_size=256, classification=False):
    input_length=input_shape[0]
    input_channels=input_shape[1]

    if classification:
        raise NotImplementedError( "classification=True is not implemented yet." )
    else:
        model=CNN_Best(input_channels, input_length, embed_size=emb_size)

    return model

