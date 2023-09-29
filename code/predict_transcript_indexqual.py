import re
from pathlib import Path

import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm, trange
import json

from RobertaForMultiTokenClassification import RobertaForMultiLabelTokenClassification

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
max_length = 512
threshold = 0.5
hidden_size = 768


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    model_checkpoint = "./robbert-v2-dutch-base-finetuned-indexqual/checkpoint-3775"
    model = RobertaForMultiLabelTokenClassification.from_pretrained(model_checkpoint,
                                                                    num_labels=4)
    model.to(device)

    tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)  # tokenizer

    file_paths = [
        'data_transcripts/katya/AKENLA41B KS-AB.html'
    ]

    for file_path in file_paths:
        context_text = ''

        obj_list = []

        with torch.no_grad():
            with open(file_path) as f:
                filename = Path(file_path).stem
                print(filename)
                text = f.read()
                # lines = re.split("[\n.?]+", text)
                lines = re.findall(r'.*?[.!?]', text)
                lines = [line.strip() for line in lines if len(line) > 1]

                for line in lines:
                    print(line)
                # lines = f.readlines()

                for i in trange(len(lines)):
                    line = re.sub("\\n", "", lines[i])

                    context = tokenizer(context_text, truncation=True,
                                        max_length=max_length,
                                        padding=True, return_tensors='pt')
                    sentence = tokenizer(line, truncation=True,
                                         max_length=max_length,
                                         padding=True, return_tensors='pt')

                    context.to(device)
                    sentence.to(device)

                    result = model(sentence.input_ids, sentence.attention_mask, context.input_ids, context.attention_mask)

                    tokens = sentence['input_ids'].detach().cpu().numpy()[0]

                    embeddings = torch.sigmoid(result.token_labels.squeeze(0))
                    embeddings = embeddings.detach().cpu().numpy()

                    token_values = []
                    token_text = ""
                    token_value = np.zeros(4)
                    for j in range(1, len(tokens) - 1):
                        embedding = embeddings[j]
                        token = tokenizer.decoder.get(tokens[j])
                        next_token = tokenizer.decoder.get(tokens[j + 1])

                        token_text += token
                        token_value += embedding

                        regexp = re.compile(r'[.,;:?!\'"]]+')

                        if re.search(regexp, token) or "Ġ" in next_token or "</s>" in next_token:

                            token_text = re.sub(r"Ġ", "", token_text)

                            token_values.append({
                                "text": token_text,
                                "value": token_value.tolist()
                            })

                            token_value = 0
                            token_text = ""

                    pred_bools = [pl > threshold for pl in torch.sigmoid(result.labels).detach().cpu().numpy()][0]

                    obj = {
                        "text": line,
                        "importance": token_values,
                        "labels": pred_bools.tolist()
                    }

                    obj_list.append(obj)

                    context_text += ' ' + lines[i]

                    context_split = context_text.split(' ')
                    if len(context_split) > 511:
                        context_split = context_split[-511:]
                        context_text = ' '.join(context_split)

        json_dump = json.dumps(obj_list)

        print(json_dump)

        with open(filename + ".json", "w") as f:
            f.write(json_dump)

    print("Done!")
