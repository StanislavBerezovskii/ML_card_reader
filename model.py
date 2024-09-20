import timm
import torch.nn as nn


class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()
        # defining all model parts
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # creating the classifier
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        # connecting the model parts and returning the output
        x = self.features(x)
        output = self.classifier(x)
        return output


model = SimpleCardClassifier(num_classes=53)
