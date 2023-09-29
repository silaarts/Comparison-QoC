import json
import os
from collections import Counter
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import pandas as pd
import re
import difflib

import torch
from pandas import DataFrame
from transformers import RobertaTokenizerFast
from transformers.modeling_outputs import TokenClassifierOutput

MAX_LENGTH = 512
STRIDE = MAX_LENGTH - 32

HTML_REGEX = re.compile(r">(.*?)<", re.MULTILINE)

tokenizer = RobertaTokenizerFast.from_pretrained('pdelobelle/robbert-v2-dutch-base')


class LabelInformation:
    """
    PlaceHolder for document information.
    """

    input_ids: list[int] = None
    attention_mask: list[int] = None
    offsets: list[tuple[int, int]] = None
    labels: list[list[Union[str, int]]] = None

    def __init__(self, input_ids: list[int],
                 attention_mask: list[int],
                 offsets: list[tuple[int, int]],
                 labels: list[list[str]]):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.offsets = offsets
        self.labels = labels

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


class Transcript:
    """
    PlaceHolder for document information.
    """

    file_name: str = None
    text: str = None

    def __init__(self, file_name: str, text: str):
        self.file_name = file_name
        self.text = text

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


class LabelledTranscript(Transcript):
    """
    PlaceHolder for labelled document information.
    """

    file_name: str = None
    labelled_text: list[LabelInformation] = None

    def __init__(self, text: str, file_name: str, labelled_text: list[LabelInformation]):
        super().__init__(file_name, text)
        self.text = text
        self.labelled_text = labelled_text


def find_tokens_from_zip(text: str, sequence, pattern: str) -> Union[list[Union[Tuple, None]], None]:
    result = text.find(pattern, 2)

    if result > -1:
        output: list[Union[Tuple, None]] = [None] * len(sequence['offset_mapping'])

        for i in range(len(sequence['offset_mapping'])):
            zip_tokens = list(zip(sequence['offset_mapping'][i], range(len(sequence['offset_mapping'][i]))))
            start_id = [idx for offset, idx in zip_tokens if offset[0] <= result < offset[1]]
            end_id = [idx for offset, idx in zip_tokens if offset[0] < result + len(pattern) <= offset[1]]

            # print(start_id, end_id)

            if len(start_id) == 0 and len(end_id) == 0:
                continue
            elif len(start_id) == 0:
                start_id.append(0)
            elif len(end_id) == 0:
                end_id.append(len(sequence['offset_mapping'][i]) - 1)

            output[i] = start_id[0], end_id[0]

            # print(output)

        return output

    return None


def preprocess(mode="top_level",
                path="data_transcripts/katya/1009-F.html",
                stride=STRIDE) -> tuple[list[LabelledTranscript], list[str]]:
    """
    preprocess is used to create the preprocessed data

    :param mode: select a mode of preprocessing [default, top_level]
    :param path:
    :param stride:
    :param deliminator: separator for file loading
    """

    # load MAXQDA data
    codes_df = pd.read_excel("data_indexqual/indexqual_codes_18-19.xlsx")
    codes_df = codes_df.replace(r'\n', ' ', regex=True)
    print(codes_df.columns)

    useful_codes = [code for code in codes_df['Code'] if get_code(code) is not None]

    # load transcripts
    if path is not None:
        transcripts = [load_transcript(path)]
    else:
        transcripts = load_html_transcripts()

    output: list[LabelledTranscript] = []
    unique_codes = set()
    num_discovered_codes = 0

    for transcript in transcripts:
        labelled_transcript, discovered_codes = encode_transcript(transcript, codes_df, stride, mode)
        output.append(labelled_transcript)
        unique_codes = unique_codes.union(discovered_codes)

    print(num_discovered_codes, "codes found out of", len(useful_codes))

    unique_codes = sorted(set(unique_codes))

    return output, unique_codes


def encode_transcript(transcript: Transcript,
                      codes_df: DataFrame,
                      stride=STRIDE,
                      mode="top_level") -> tuple[LabelledTranscript, list[str]]:

    print(transcript.file_name)
    # print(transcript.text)

    # find all codes that belong to this transcript
    transcript_codes = codes_df[codes_df['Document name'] == transcript.file_name]

    if len(transcript_codes) == 0:
        print("WARNING: no codes found")

    transcript_useful_codes = 0
    transcript_discovered_codes = 0
    unique_codes: list[str] = list()

    # encode text into tokens
    tokenized_text = tokenizer(transcript.text,
                               max_length=MAX_LENGTH,
                               truncation=True,
                               padding=True,
                               return_offsets_mapping=True,
                               return_overflowing_tokens=True,
                               stride=stride)

    # for el in tokenized_text['offset_mapping']:
    #     print(len(el))

    length = np.sum([len(el) for el in tokenized_text['offset_mapping']])
    # print(length, len(transcript.text))

    label_tokens: list[list[list[str]]] = []
    for el in tokenized_text['input_ids']:
        label_tokens.append([[]] * len(el))

    for code, segment in zip(transcript_codes['Code'], transcript_codes['Segment']):
        lowered_code = get_code(code, mode)
        if lowered_code is not None:
            transcript_useful_codes += 1

            processed_segment = process_segment(segment)

            # find sentence in transcript.text
            if processed_segment in transcript.text:
                matches = find_tokens_from_zip(transcript.text, tokenized_text, processed_segment)
                if matches:
                    for i in range(len(matches)):
                        match = matches[i]

                        if match is None:
                            continue

                        if lowered_code not in unique_codes:
                            unique_codes.append(lowered_code)

                        for token_idx in range(match[0], match[1] + 1):
                            if lowered_code not in label_tokens[i][token_idx]:
                                label_tokens[i][token_idx] = [lowered_code] + label_tokens[i][token_idx]

                    transcript_discovered_codes += 1
            else:
                print(processed_segment, "NOT FOUND")

    # print(label_tokens)

    label_text = [LabelInformation(text, attention_mask, offsets, labels) for text, attention_mask, offsets, labels in
                  zip(tokenized_text['input_ids'], tokenized_text['attention_mask'], tokenized_text['offset_mapping'], label_tokens)]

    if len(transcript_codes) == 0:
        print("Warning: no codes found")
    else:
        print(transcript_discovered_codes, "codes found out of", transcript_useful_codes)

    return (LabelledTranscript(
        text=transcript.text,
        file_name=transcript.file_name,
        labelled_text=label_text
    ), unique_codes)


# def decode_transcript(text: str,
#                       model_result: list[tuple[torch.FloatTensor, int, torch.FloatTensor]],
#                       label_names: list[str],
#                       inv_stride=MAX_LENGTH - STRIDE,
#                       threshold=0.5) -> Union[Tuple[Transcript, DataFrame], None]:
#
#     pred_labels = torch.zeros((1, 0, len(label_names)))
#     offset_mapping = torch.IntTensor()
#     for el in model_result:
#         pred_labels = torch.cat((pred_labels, el[0]['logits'][:, 0:inv_stride, :]), dim=1)
#         offset_mapping = torch.cat((offset_mapping, el[1][0:inv_stride]))
#
#     last_el = model_result[-1]
#     pred_labels = torch.cat((pred_labels, last_el[0]['logits'][:, inv_stride:, :]), dim=1)
#     offset_mapping = torch.cat((offset_mapping, last_el[1][inv_stride:]))
#
#     pred_labels = pred_labels.squeeze(0)
#
#     label_results = []
#     label_name = None
#     prev_label_name = None
#     prev_end = 0
#     for i in range(len(pred_labels)):
#         label_ids = pred_labels[i]
#         offset = offset_mapping[i]
#
#         label = [i for i,v in enumerate(label_ids.tolist()) if v >= threshold]
#         start = int(offset[0])
#         end = int(offset[1])
#
#         if start == 0 and end == 0:
#             continue
#
#         if end - start > 0:
#             label_name = [label_names[l] for l in label]
#             print(label_name)
#
#             if prev_label_name != label_name:
#                 label_results.append({
#                     "text": text[start:end],
#                     "label": label_name
#                 })
#             else:
#                 label_results[-1]['text'] = label_results[-1]['text'] + text[prev_end:end]
#
#         prev_end = end
#         prev_label_name = label_name
#
#     return None


def load_html_transcripts(transcripts_dir="data_transcripts/katya/"):
    """
    preprocess is used to create the preprocessed data

    :param transcripts_dir: header for file loading
    """

    # init list
    documents: list[Transcript] = []

    # load transcript
    for filename in sorted(os.listdir(transcripts_dir)):
        filepath = os.path.join(transcripts_dir, filename)

        if '.html' in filepath:
            document = load_transcript(filepath)
            documents.append(document)

    # return sentence list
    return documents


def load_transcript(filepath):
    filename = Path(filepath).stem

    document = []

    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

        for i in range(0, len(lines)):
            for segment in re.findall(r'>(.*?)<', lines[i]):

                segment = process_segment(segment)

                if 'minuten:' in segment:
                    continue

                if 'taal:' in segment:
                    continue

                if segment == 'nederlands':
                    continue

                if 'de interviewer is aangeduid als sp1' in segment:
                    continue

                if 'de respondent is aangeduid als sp2' in segment:
                    continue

                if 'toelichting transcriptie:' in segment:
                    continue

                if 'onduidelijkheden zijn rood gemarkeerd' in segment:
                    continue

                if 'sp1 aan het opnemen' in segment:
                    continue

                if 'sprekers:' in segment:
                    continue

                if 'deelnemer b de bewoonster' in segment:
                    continue

                if 'code interview:' in segment:
                    continue

                if 'datum interview:' in segment:
                    continue

                if 'duur interview:' in segment:
                    continue

                if 'aanwezigen:' in segment:
                    continue

                if segment == '[00:00:00]':
                    continue

                if re.search('[a-zA-Z]', segment) is None:
                    continue

                document.append(segment)

    return Transcript(
        file_name=filename,
        text=process_segment(" ".join(document))
    )


def process_segment(segment):
    # convert to lowercase and remove line breaks
    segment = segment.lower().replace("[\n\r]+", " ")

    # convert HTML spaces to normal spaces
    segment = segment.replace(" ", " ")

    # trim string
    segment = re.sub(r' +', " ", segment)
    return segment.strip()


def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2))
    return s1[pos_a:pos_a + size], pos_a, pos_b


def get_code(code, mode="top_level"):
    # select only the top level code
    code = code.lower().replace(" ", "_").split("\\")[0]

    if mode == "sentiment" or mode == "top_level_sentiment" or mode == "default_sentiment":
        if 'pos' in code:
            return "positive"

        if 'neg' in code:
            return "negative"

        if mode == "sentiment":
            return None

    # continue of code is not indexqual
    if 'overig' in code:
        return None

    # continue if code is blue
    if 'blue' in code:
        return None

    # convert code
    if 'positive' in code or \
            'negative' in code or \
            'pos' in code or \
            'neg' in code:
        return None

    # mode for classification of main themes (instead of sub-themes)
    if mode == "top_level" or mode == "top_level_sentiment":
        if 'word-of-mouth' in code or \
                'personal_needs' in code or \
                'past_experiences' in code:
            code = 'expectations'

        if 'relationship-centered_care' in code or \
                'resident-caregiver' in code or \
                'resident-family' in code or \
                'family-caregiver' in code or \
                'care_environment' in code:
            code = 'experiences'

        if 'satisfaction' in code or \
                'perceived_care_services' in code or \
                'experienced_quality_of_care' in code or \
                'perceived_care_outcomes' in code:
            code = 'experienced_qoc'

    return code


def labels_to_one_hot(token_id: int, all_special_ids: list[int],
                      labels: list[str], unique_labels):
    output: list[int] = []

    for label in unique_labels:
        if token_id in all_special_ids:
            output.append(-100)
        else:
            output.append(int(label in labels))

    return output


def get_token_labels(stride=STRIDE,
                     path=None,
                     code_mode="top_level"):

    global tokenizer
    if tokenizer is None:
        tokenizer = RobertaTokenizerFast.from_pretrained('pdelobelle/robbert-v2-dutch-base')

    all_special_ids = tokenizer.all_special_ids

    coded_transcripts, unique_labels = preprocess(path=path, mode=code_mode, stride=stride)

    unique_labels = sorted(unique_labels)
    if path is not None:
        unique_labels = ['care_environment',
                         'context',
                         'expectations',
                         'experienced_quality_of_care',
                         'experiences',
                         'family-caregiver',
                         'negative',
                         'past_experiences',
                         'perceived_care_outcomes',
                         'perceived_care_services',
                         'personal_needs',
                         'positive',
                         'relationship-centered_care',
                         'resident-caregiver',
                         'resident-family',
                         'satisfaction',
                         'word-of-mouth']

    token_labels = [(item, coded_transcript.file_name) for coded_transcript in coded_transcripts
                    for item in coded_transcript.labelled_text]

    transcript_names = [file_name for item, file_name in token_labels]
    token_input_ids = [item.input_ids for item, file_name in token_labels]
    attention_masks = [item.attention_mask for item, file_name in token_labels]
    one_hot_labels = [[labels_to_one_hot(input_id, all_special_ids, label, unique_labels)
                      for label, input_id in zip(item.labels, item.input_ids)]
                      for item, file_name in token_labels]

    return [{
        "transcript_name": transcript_name,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    } for transcript_name, input_ids, attention_mask, labels
        in zip(transcript_names, token_input_ids, attention_masks, one_hot_labels)], unique_labels


# for testing
if __name__ == '__main__':
    print("testing preprocessing")
    # codes, label_names = get_token_labels()
    # print(label_names)
    # df = pd.DataFrame(codes)
    # print(df)
    # print(json.dumps(codes, indent=4))
    # main_tokenizer = RobertaTokenizerFast.from_pretrained('pdelobelle/robbert-v2-dutch-base')

    # transcript, codes = preprocess(mode="single_test", path="data_transcripts/katya/DMENLA19F AB-KS.html")
    transcripts, codes = preprocess()

    # text = transcript[0].text
    #
    # labelled_data_test = [[(labels_to_one_hot(labels, codes), text[offset[0]:offset[1]]) for labels, offset in zip(el.labels, el.offsets)]
    #                       for el in transcript[0].labelled_text]
    #
    # labelled_data_test = [str(el) for el in labelled_data_test]
    #
    # print("\n".join(labelled_data_test)) # <--- correct

    flat_labelled_data = [[labels_to_one_hot(input_id, labels, codes) for labels, input_id in zip(el.labels, el.input_ids)]
                          for transcript in transcripts for el in transcript.labelled_text]
    flat_list = [str(item) for sublist in flat_labelled_data for item in sublist]

    count = Counter(flat_list)

    keys = count.keys()
    values = list(count.values())
    total_sum = np.sum(values)

    print(codes)
    print(keys)
    print(values)
    print(total_sum)

    occurrences_table = pd.DataFrame({
        'Permutation': keys,
        'Occurrences': np.round(np.divide(values, total_sum / 100), 2)
    })

    occurrences_table = occurrences_table.sort_values(by='Occurrences', ascending=False)

    print(occurrences_table.to_markdown(index=False))
