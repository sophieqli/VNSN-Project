import torch, os

def load_hidden(hidden_dir):
    return torch.load(os.path.join(hidden_dir, 'hidden.pt'))

def build_steering_vectors(data_dir, layers, save_dir):
    hidden = load_hidden(data_dir)
    os.makedirs(save_dir, exist_ok=True)
    for layer in layers:
        layer_data = hidden[layer]  # dict idx->Tensor(num_steps,hidden)
        all_steps = torch.cat([v for v in layer_data.values()], dim=0)
        # naive: first half as 'transition', second as 'reflection'
        half = all_steps.size(0)//2
        transition = all_steps[:half]
        reflection = all_steps[half:]
        steer_vec = transition.mean(dim=0) - reflection.mean(dim=0)
        torch.save(steer_vec, os.path.join(save_dir, f'layer_{layer}_steer_vec.pt'))
        print(f'Saved steering vector for layer {layer}')