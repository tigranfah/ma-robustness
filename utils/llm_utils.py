import torch


def compute_next_token_acc(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            preds = logits.argmax(dim=-1)
            mask = attention_mask[:, 1:].bool()
            correct += (preds == labels)[mask].sum().item()
            total += mask.sum().item()
    return correct / total


def compute_adv_acc(model, attacker, loader, device):
    model.eval()
    correct = total = 0
    for input_ids, attention_mask in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        adv_embeds = attacker.attack(input_ids, attention_mask)
        with torch.no_grad():
            outputs = model(inputs_embeds=adv_embeds, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        preds = logits.argmax(dim=-1)
        mask = attention_mask[:, 1:].bool()
        correct += (preds == labels)[mask].sum().item()
        total += mask.sum().item()
    return correct / total


def compute_avg_max_activation(model, loader, device):
    model.eval()
    total_max = 0.0
    n = 0
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            for hidden in outputs.hidden_states[1:]:
                token_maxes = hidden.abs().max(dim=-1).values
                total_max += token_maxes.mean().item()
                n += 1
    return total_max / n
