import torch.nn as nn

class ImageClassifier(nn.Module):

    def __init__(self,
                input_size,
                output_size):
        self.input_size = input_size
        self.output_size = output_size

        super().__init__()


    def forward(self, x):
        pass

    