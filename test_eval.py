import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import utils
import numpy as np
# Define a simple mock model
class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.linear = nn.Linear(1, 2)  # A simple linear layer to produce two outputs

    def forward(self, t):
        # Assume t is a 1D tensor with the same length as the input data
        return self.linear(t.unsqueeze(-1))

# Create the input data tensors
real_t = torch.tensor([0, 2, 6], dtype=torch.float32)
real_y = torch.tensor([0.1, 2, 2], dtype=torch.float32)
real_d = torch.tensor([0.2, 3, 1], dtype=torch.float32)

# Create a DataLoader using the input data
dataset = TensorDataset(real_t, real_y, real_d)  # Assume each element of real_t, real_y, real_d corresponds to a data point
dataloader = DataLoader(dataset, batch_size=3, shuffle=False)  # Set batch_size to the number of data points

# Create the mock model
model = MockModel()


@torch.no_grad()
def estimate_counterfactuals(model, dataloader, use_treatment=True):
    model.eval()  # Set the model to evaluation mode
    all_counterfactual_differences = []

    for real_t, real_y, real_d in dataloader:
        
        if use_treatment:
            t_original = real_t
            batch_counterfactual_differences = []

            # Get the original predictions
            out_original = model(t_original)
            
            for t_value in range(7):  # t values ranging from 0 to 6
                t = torch.full_like(t_original, fill_value=t_value)  # Create a tensor with the new t value
                out_intervene = model(t)
                
                
                # Compute the differences from the original predictions
                diff_y = out_original[:, 0] - out_intervene[:, 0]
                diff_d = out_original[:, 1] - out_intervene[:, 1]
                
                # Store the t value difference and the prediction differences
                t_diff = t_value - t_original
                differences_dict = {
                    "t_diff": t_diff.cpu().numpy(),
                    "diff_y": diff_y.cpu().numpy(),
                    "diff_d": diff_d.cpu().numpy()
                }
                batch_counterfactual_differences.append(differences_dict)
            
            all_counterfactual_differences.append(batch_counterfactual_differences)
        else:
            raise ValueError("The use_treatment argument should be True for counterfactual estimation.")
    
    return all_counterfactual_differences  # Returns a list of lists of tuples with t difference and prediction differences
from collections import defaultdict

def organize_counterfactuals(all_counterfactual_differences):
    organized_counterfactuals = defaultdict(lambda: {"diff_y": [], "diff_d": []})

    for batch_counterfactual_differences in all_counterfactual_differences:
        for differences_dict in batch_counterfactual_differences:
            t_diff_values = differences_dict['t_diff']
            diff_y_values = differences_dict['diff_y']
            diff_d_values = differences_dict['diff_d']
            
            # Assuming all arrays have the same length
            for i in range(len(t_diff_values)):
                t_diff = t_diff_values[i]
                key = f't_diff_{int(t_diff)}'
                organized_counterfactuals[key]["diff_y"].append(diff_y_values[i])
                organized_counterfactuals[key]["diff_d"].append(diff_d_values[i])
    # Convert lists to numpy arrays for consistency
    for key, value in organized_counterfactuals.items():
        organized_counterfactuals[key]["diff_y"] = np.array(value["diff_y"])
        organized_counterfactuals[key]["diff_d"] = np.array(value["diff_d"])

    return organized_counterfactuals

# Call the estimate_counterfactuals function
counterfactual_differences = estimate_counterfactuals(model, dataloader)
organized_counterfactuals = organize_counterfactuals(counterfactual_differences)
# Print the results
for key, value in counterfactual_differences.items():
    print(f'{key}: {value}')