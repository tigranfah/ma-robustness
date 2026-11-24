from datasets import load_dataset
import torch


def get_pile_loaders(tokenizer, max_batches, batch_size, seq_len):
    dataset = load_dataset(
        "EleutherAI/the_pile_deduplicated",
        split="train",
        streaming=True,
    )

    train_input_ids = []
    train_attention_masks = []
    test_input_ids = []
    test_attention_masks = []

    for i, example in enumerate(dataset):
        ids = tokenizer(
            example["text"],
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        if i < max_batches * batch_size:
            train_input_ids.append(ids["input_ids"].squeeze(0))
            train_attention_masks.append(ids["attention_mask"].squeeze(0))
        elif i < 2 * max_batches * batch_size:
            test_input_ids.append(ids["input_ids"].squeeze(0))
            test_attention_masks.append(ids["attention_mask"].squeeze(0))
        else:
            break

    train_dataset = torch.utils.data.TensorDataset(
        torch.stack(train_input_ids),
        torch.stack(train_attention_masks),
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.stack(test_input_ids),
        torch.stack(test_attention_masks),
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader