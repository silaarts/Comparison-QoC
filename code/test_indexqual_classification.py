import json
import numpy as np
import torch
from transformers import RobertaTokenizerFast, RobertaTokenizer
from transformers.modeling_outputs import TokenClassifierOutput

from RobertaForMultiTokenClassification import RobertaForMultiLabelTokenClassification
from preprocess_interviews import load_transcript, preprocess, labels_to_one_hot

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

MAX_LENGTH = 512
NEG_STRIDE = 32
STRIDE = MAX_LENGTH - NEG_STRIDE


def reproject_model_output(model_result, num_label_names: int, attention_end: int):
    pred_labels = torch.zeros((1, 0, num_label_names))
    offset_mapping = torch.IntTensor()

    for el in model_result:
        pred_labels = torch.cat((pred_labels, el[0]['logits'][:, 0:NEG_STRIDE, :]), dim=1)
        offset_mapping = torch.cat((offset_mapping, el[1][0:NEG_STRIDE]))

    last_el = model_result[-1]
    pred_labels = torch.cat((pred_labels, last_el[0]['logits'][:, NEG_STRIDE:attention_end, :]), dim=1)
    offset_mapping = torch.cat((offset_mapping, last_el[1][NEG_STRIDE:attention_end]))

    return pred_labels, offset_mapping


def calc_label(item, threshold=0.5):
    return [i for i,v in enumerate(item.tolist()) if v >= threshold]


def predict_long(path: str, text: str, label_names: list[str], model: RobertaForMultiLabelTokenClassification, tokenizer: RobertaTokenizerFast):
    # print(len(text))

    print(path)

    # encode text into tokens
    tokenized_text = tokenizer(text,
                               max_length=MAX_LENGTH,
                               truncation=True,
                               padding=True,
                               return_offsets_mapping=True,
                               return_overflowing_tokens=True,
                               stride=STRIDE,
                               return_tensors='pt')

    attention_end = MAX_LENGTH - np.count_nonzero(tokenized_text['attention_mask'] == 0)
    # print(attention_end, tokenized_text['attention_mask'])

    model_result = [(model(torch.unsqueeze(ids, 0), attention_mask=torch.unsqueeze(mask, 0)), offset) for ids, mask, offset
                    in zip(tokenized_text['input_ids'], tokenized_text['attention_mask'], tokenized_text['offset_mapping'])]

    # sanity check
    transcript, codes = preprocess(mode="top_level_sentiment", path=path)
    # print(label_names)
    # print(codes)
    # lowered_label_names = [c.lower() for c in label_names]
    lowered_label_names = ['context', 'expectations', 'experienced_qoc', 'experiences', 'negative', 'positive'] # TODO adapt
    labelled_data_test = [[labels_to_one_hot(input_id, tokenizer.all_special_ids, labels, lowered_label_names) for input_id, labels in zip(el.input_ids, el.labels)]
                          for el in transcript[0].labelled_text]
    manual_result = [(TokenClassifierOutput(
        logits=torch.FloatTensor(label).unsqueeze(0)
    ), offset) for label, ids, mask, offset
        in zip(labelled_data_test, tokenized_text['input_ids'], tokenized_text['attention_mask'], tokenized_text['offset_mapping'])]

    # print(model_result[0])

    pred_labels, offset_mapping = reproject_model_output(model_result, len(label_names), attention_end)
    m_pred_labels, m_offset_mapping = reproject_model_output(manual_result, len(label_names), attention_end)
    pred_labels = pred_labels.detach().numpy()
    m_pred_labels = m_pred_labels.detach().numpy()
    # print(pred_labels.shape)
    # print(m_pred_labels.shape)
    pred_labels = np.concatenate([pred_labels, m_pred_labels], axis=2)

    pred_labels = pred_labels.squeeze(0)

    a_label_names = [l + " (A)" for l in label_names]
    m_label_names = [l + " (M)" for l in label_names]

    label_names = a_label_names + m_label_names
    # print(label_names)

    label_results = []
    label_name = None
    prev_label_name = None
    prev_end = 0
    for i in range(len(pred_labels) - 1):
        label_ids = pred_labels[i]
        offset = offset_mapping[i]

        label = calc_label(label_ids)
        start = int(offset[0])
        end = int(offset[1])

        next_label_ids = pred_labels[i + 1]
        next_label = calc_label(next_label_ids)

        if start == 0 and end == 0:
            continue

        if end - start > 0:
            label_name = [label_names[l] for l in label]
            next_label_name = [label_names[l] for l in next_label]
            # print(label_name)

            if prev_label_name is None or (set(prev_label_name) != set(label_name) and
                                           set(next_label_name) != set(prev_label_name)):
                label_results.append({
                    "text": text[start:end],
                    "label": label_name,
                    "length": 1
                })
            else:
                label_results[-1]['text'] = label_results[-1]['text'] + text[prev_end:end]
                label_results[-1]['length'] = label_results[-1]['length'] + 1

        prev_end = end
        prev_label_name = label_name

    print("")
    print(json.dumps(label_results))
    print("")

    return label_results


if __name__ == "__main__":
    model_checkpoint = "./checkpoint-15040"
    roberta_model = RobertaForMultiLabelTokenClassification.from_pretrained(model_checkpoint,
                                                                            num_labels=6)
    roberta_model.to(device)

    roberta_tokenizer = RobertaTokenizerFast.from_pretrained('pdelobelle/robbert-v2-dutch-base')

    # {'context', 'expectations', 'experienced_qoc', 'experiences'}
    unique_labels = ['Context',
                     'Expectations',
                     'Experienced QoC',
                     'Experiences',
                     'Negative',
                     'Positive']

    file_paths = [
        'data_transcripts/katya/AKENLA41F KS-AB.html',
        'data_transcripts/katya/JBMEDO135Z KS-AB.html', # take from actual data
        'data_transcripts/katya/911 B.html',
    ]

    for file_path in file_paths:
        transcript = load_transcript(file_path)

        predict_long(file_path, transcript.text, unique_labels, roberta_model, roberta_tokenizer)
