import torch
import numpy as np

# Add necessary numpy globals to the safe list
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray])

# Load the tensor or model with weights_only=True
file_path = 'vectors\syntax_knn.pt'
data = torch.load(file_path, weights_only=False)

# Convert data to string based on its type
if isinstance(data, torch.Tensor):
    data_str = str(data)
elif isinstance(data, dict):
    data_str = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    data_str = str(data_str)
elif isinstance(data, (list, tuple)):
    data_str = [item.tolist() if isinstance(item, torch.Tensor) else item for item in data]
    data_str = str(data_str)
else:
    data_str = str(data)

file_name = 'syntax_knn.txt'

# Write the string to a txt file
with open(file_name, 'w') as file:
    file.write(data_str)

print("Data has been written to " + file_name)