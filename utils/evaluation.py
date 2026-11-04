import torch
from tqdm import tqdm


def inference(model, tokenizer, test_samples, batch_size=1):

    predictions, labels = [], []
    tokenizer.pad_token = tokenizer.eos_token
    pbar = tqdm(total=len(test_samples['text']), desc='Evaluating  ')
    with torch.inference_mode():
        for start in range(0, len(test_samples['text']), batch_size):
            end = min(start + batch_size, len(test_samples['text']))
            texts = [sample for sample in test_samples['text'][start: end]]
            input_ids = tokenizer(texts, return_tensors='pt', max_length=2048, padding=True, truncation=True).to(model.device)
            output_tokens = model.generate(**input_ids, max_new_tokens=50, do_sample=False, use_cache=True)
            for ele in output_tokens:
                decoded_output = tokenizer.decode(ele, skip_special_tokens=True)
                decoded_output = decoded_output.replace('<pad>', '').strip()
                predictions.append(decoded_output)
            for label in test_samples['label'][start: end]:
                labels.append(label)
            pbar.update(end - start)
    pbar.close()

    return predictions, labels

