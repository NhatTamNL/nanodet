import torch
import numpy as np

# Load the PyTorch model
model = torch.load('workspace/nanodet-plus-m_416/model_best/nanodet_model_best.pth')

# Function to convert the PyTorch model to Darknet format
def convert_pytorch_to_darknet(pytorch_model, cfg_file, weights_file):
    with open(cfg_file, 'w') as cfg:
        # Write the .cfg file based on your PyTorch model architecture
        # This part needs to be customized based on your model's architecture
        pass
    
    # Convert weights and save to .weights file
    with open(weights_file, 'wb') as f:
        # Example: Assuming model is a state_dict containing 'state_dict' key
        state_dict = pytorch_model['state_dict']
        
        for key, value in state_dict.items():
            if 'num_batches_tracked' in key:
                continue
            if isinstance(value, torch.nn.parameter.Parameter):
                value = value.detach().cpu().numpy()
            else:
                value = value.cpu().numpy()
            f.write(value.tobytes())

# Example usage
convert_pytorch_to_darknet(model, 'model_nano.cfg', 'model_nano.weights')
