# %%
import argparse
import dotenv
dotenv.load_dotenv("../.env")

import torch
import re
import json
import random
from messages import eval_messages
import messages
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import gc
import os
import utils

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate steering effects on model reasoning")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model to evaluate")
parser.add_argument("--n_examples", type=int, default=50,
                    help="Number of examples to use for evaluation")
parser.add_argument("--max_tokens", type=int, default=1000,
                    help="Maximum number of tokens to generate")
parser.add_argument("--load_in_8bit", type=bool, default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--seed", type=int, default=42, 
                    help="Random seed")
parser.add_argument("--only_viz", action="store_true",
                    help="Only visualize the results")
args = parser.parse_args()

# %%
def get_label_counts(thinking_process, original_response, labels):
    # Get annotated version using chat function
    annotated_response = utils.process_batch_annotations([thinking_process])[0]
    
    # Get token positions for each label using get_label_positions
    label_positions = utils.get_label_positions(annotated_response, original_response, tokenizer)
    
    # Initialize token counts for each label
    label_counts = {label: 0 for label in labels}
    
    # Calculate total tokens and per-label tokens
    total_tokens = 0
    for label, positions in label_positions.items():
        if label in labels:  # Only count if it's one of our target labels
            for start, end in positions:
                label_counts[label] += (end - start)
                total_tokens += (end - start)
    
    # Convert to fractions
    label_fractions = {
        label: count / total_tokens if total_tokens > 0 else 0 
        for label, count in label_counts.items()
    }
            
    return label_fractions, annotated_response

def generate_and_analyze(model, tokenizer, message, feature_vectors, model_steering_config, label, labels, steer_mode="none"):
    input_ids = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    steer_positive = True if steer_mode == "positive" else False

    output_ids = utils.custom_generate_steering(
        model,
        tokenizer,
        input_ids,
        max_new_tokens=args.max_tokens,
        label=label if steer_mode != "none" else "none",
        feature_vectors=feature_vectors if steer_mode != "none" else None,
        steering_config=utils.steering_config[model_name],
        steer_positive=steer_positive if steer_mode != "none" else None,
    )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract thinking process
    think_start = response.index("<think>") + len("<think>")
    try:
        think_end = response.index("</think>")
    except ValueError:
        think_end = len(response)
    thinking_process = response[think_start:think_end].strip()
    
    label_fractions, annotated_response = get_label_counts(thinking_process, response, labels)
    
    return {
        "response": response,
        "thinking_process": thinking_process,
        "label_fractions": label_fractions,
        "annotated_response": annotated_response
    }

def plot_label_statistics(results, model_name, ax=None, show_legend=True, show_ylabel=True, xtick_rotation=0):
    # Get model identifier for file naming
    model_id = model_name.split('/')[-1].lower()

    # Use white background
    plt.style.use('seaborn-v0_8-white')
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
        save_fig = True
    else:
        save_fig = False
        
    labels_list = list(results.keys())
    x = np.arange(len(labels_list))
    width = 0.25
    
    # Calculate means as before
    original_means = []
    positive_means = []
    negative_means = []
    
    for label in labels_list:
        orig_fracs = [ex["original"]["label_fractions"].get(label, 0) for ex in results[label]]
        pos_fracs = [ex["positive"]["label_fractions"].get(label, 0) for ex in results[label]]
        neg_fracs = [ex["negative"]["label_fractions"].get(label, 0) for ex in results[label]]
        
        original_means.append(np.mean(orig_fracs))
        positive_means.append(np.mean(pos_fracs))
        negative_means.append(np.mean(neg_fracs))
    
    # Plot bars with black edges
    ax.bar(x - width, original_means, width, label='Original', color='#2E86C1', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x, positive_means, width, label='Positive Steering', color='#27AE60', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x + width, negative_means, width, label='Negative Steering', color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add percentage labels on top of bars
    def add_labels(positions, values):
        for pos, val in zip(positions, values):
            # Position text slightly above the bar
            y_pos = val + 0.02 * max(original_means + positive_means + negative_means)
            ax.text(pos, y_pos, f'{val*100:.0f}%', ha='center', va='bottom', fontsize=16)

    if ax is None:
        add_labels(x - width, original_means)
        add_labels(x, positive_means)
        add_labels(x + width, negative_means)

    # Improve styling with larger font sizes and bold title
    if show_ylabel:
        ax.set_ylabel('Average Sentence Fraction (%)', fontsize=18, labelpad=10)
    ax.set_title(model_name.split('/')[-1], fontsize=18, pad=20, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace('-', '\n') for label in labels_list], rotation=xtick_rotation, fontsize=18, ha="center")
    ax.tick_params(axis='y', labelsize=16)
    
    # Convert y-axis to percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
    
    # Add grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Customize legend with larger font
    if show_legend:
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=18, loc="upper left")
    
    # Show all spines (lines around the plot)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Adjust y-axis limit to make space for labels
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], current_ylim[1] * 1.05)

    if save_fig:
        # Create figures directory if it doesn't exist
        os.makedirs('results/figures', exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'results/figures/steering_results_{model_id}.pdf', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def plot_combined_statistics(all_model_names):
    # Create figures directory if it doesn't exist
    os.makedirs('results/figures', exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    
    for i, model_name in enumerate(all_model_names):
        model_id = model_name.split('/')[-1].lower()
        try:
            with open(f'results/vars/steering_evaluation_results_{model_id}.json') as f:
                results = json.load(f)
        except FileNotFoundError:
            print(f"Results file for {model_name} not found, skipping.")
            continue
            
        show_legend = (i == 0)
        show_ylabel = (i == 0)
        plot_label_statistics(results, model_name, ax=axes[i], show_legend=show_legend, show_ylabel=show_ylabel, xtick_rotation=45)
        
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent title overlap
    plt.savefig('results/figures/steering_results_deepseek-r1-distill-all.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# %% Parameters
only_viz = args.only_viz
n_examples = args.n_examples
random.seed(args.seed)
model_name = args.model
model_id = model_name.split('/')[-1].lower()

# %% Create data directory if it doesn't exist
os.makedirs('results/vars', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# %%
# Load model and vectors
if not only_viz:
    print(f"Loading model {model_name}...")
    model, tokenizer, feature_vectors = utils.load_model_and_vectors(compute_features=True, model_name=model_name, load_in_8bit=args.load_in_8bit)

# %% Randomly sample evaluation examples
if not only_viz:
    eval_indices = random.sample(range(len(eval_messages)), n_examples)

    # Store results
    labels = list(list(utils.steering_config.values())[0].keys())
    results = {label: [] for label in labels}

    # Evaluate each label
    for label in labels:
        for idx in tqdm(eval_indices, desc=f"Processing examples for {label}"):
            message = eval_messages[idx]

            # Only proceed if original version has >5% of the target label
            example_results = {
                "original": generate_and_analyze(model, tokenizer, message, feature_vectors, utils.steering_config[model_name], label, labels, "none"),
                "positive": generate_and_analyze(model, tokenizer, message, feature_vectors, utils.steering_config[model_name], label, labels, "positive"),
                "negative": generate_and_analyze(model, tokenizer, message, feature_vectors, utils.steering_config[model_name], label, labels, "negative")
            }
            
            results[label].append(example_results)

    # Save results
    with open(f'results/vars/steering_evaluation_results_{model_id}.json', 'w') as f:
        json.dump(results, f, indent=2)

# %% Plot statistics
# results = json.load(open(f'results/vars/steering_evaluation_results_{model_id}.json'))
# plot_label_statistics(results, model_name)

# %% Plot combined statistics for all models
all_model_names = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
]
plot_combined_statistics(all_model_names)

# %%
