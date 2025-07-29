import torch
import torch.nn.functional as F

# Custom generation with pruning, no nnsight
def custom_gen_pruning(model, tokenizer, input_ids, scorer, max_new_tokens=200, k=5):
    device = input_ids.device
    generated = input_ids.clone()

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Get logits for current sequence
            outputs = model(input_ids=generated)
            logits = outputs.logits[:, -1, :]  # shape: (1, vocab_size)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k, dim=-1)  # (1, k)

            topk_tokens = topk_indices[0]  # (k,)
            topk_probs_ = topk_probs[0]
            tokens_and_probs = [
                (tokenizer.decode([tok]), prob.item())
                for tok, prob in zip(topk_tokens, topk_probs_)
            ]

            if step % 25 == 1:
              print(f"Step {step+1} top-{k} candidates:")
              for token_str, prob in tokens_and_probs:
                  print(f"  Token: '{token_str}' | Probability: {prob:.4f}")

            repeated_input = generated.repeat(k, 1)  # (k, seq_len)
            candidate_tokens = topk_tokens.unsqueeze(1)  # (k, 1)
            candidate_seqs = torch.cat([repeated_input, candidate_tokens], dim=1)  # (k, seq_len+1)

            # forward pass all k candidates
            candidate_outputs = model(input_ids=candidate_seqs, output_hidden_states=True, return_dict=True)
            # grab hidden states from the last layer (tuple of layers, take last one)
            hidden_states = candidate_outputs.hidden_states[-1][:, -1, :]  # (k, hidden_dim)
            logits = candidate_outputs.logits[:, -1, :]  # (k, vocab_size)

            #print("hiddens: ", hidden_states)
            # Score each candidate
            efficiency_scores = []
            for i in range(k):
                score, _ = scorer.drift_penalty(hidden_states[i])
                efficiency_scores.append(-1 * score)

            best_idx = torch.tensor(efficiency_scores).argmax()
            best_token = topk_tokens[best_idx].unsqueeze(0).unsqueeze(0)  # (1, 1)
            scorer.update_cache(hidden_states[best_idx])

            generated = torch.cat([generated, best_token.to(device)], dim=1)

            if best_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)
