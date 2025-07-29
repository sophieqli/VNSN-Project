import torch, os

def load_hidden(hidden_dir):
    return torch.load(os.path.join(hidden_dir, 'hidden.pt'))

def build_steering_vectors(data_dir, layers, save_dir):
    hidden = load_hidden(data_dir)
    os.makedirs(save_dir, exist_ok=True)

    for layer in layers:
        # layer_data = hidden.get(layer, {})  # dict idx -> {"states": Tensor, "pcs_score": float}
        # if not layer_data:
        #     print(f"Warning: no data for layer {layer}")
        #     continue
        
        if layer < 0 or layer >= len(hidden):
            print(f"Warning: layer {layer} is out of bounds")
            continue
            
        layer_data = hidden[layer]  # dict idx -> {"states": Tensor, "pcs_score": float}
        if not isinstance(layer_data, dict) or len(layer_data) == 0:
            print(f"Warning: no data for layer {layer}")
            continue

        # Extract the 'states' tensor from each example
        states_list = [entry["states"] for entry in layer_data.values()]
        # Now concatenate into one big (num_total_steps, hidden_dim) tensor
        all_steps = torch.cat(states_list, dim=0)

        # Split into “transition” vs “reflection” halves
        half = all_steps.size(0) // 2
        transition = all_steps[:half]
        reflection = all_steps[half:]

        # Compute SEAL‐style steer vector
        steer_vec = transition.mean(dim=0) - reflection.mean(dim=0)

        out_path = os.path.join(save_dir, f'layer_{layer}_steer_vec.pt')
        torch.save(steer_vec, out_path)
        print(f"Saved steering vector for layer {layer} to {out_path}")
