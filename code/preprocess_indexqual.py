import os
import numpy
import pandas as pd
import re
import difflib

MAX_LENGTH = 511

HTML_REGEX = re.compile(r">(.*?)<", re.MULTILINE)


def preprocess(mode="default"):
    """
    preprocess is used to create the preprocessed data

    :param mode: select a mode of preprocessing [default, top_level]
    :param deliminator: separator for file loading
    """

    # load MAXQDA data
    df = pd.read_excel("data_indexqual/indexqual_codes_18-19.xlsx")

    # load transcripts
    all_sentences, transcripts, _ = load_html_transcripts()

    # initialise lists
    unique_codes = []
    segments = []
    labels = []

    for index, row in df.iterrows():
        # select only the top level code
        code = row['Code'].lower().replace(" ", "_").split("\\")[0]

        # continue of code is not indexqual
        if 'overig' in code:
            continue

        # continue if code is blue
        if 'blue' in code:
            continue

        # convert code
        if 'positive' in code or \
                'negative' in code or \
                'pos' in code or \
                'neg' in code:
            continue  # these codes are only for the sentiment analysis

        # mode for classification of main themes (instead of sub-themes)
        if mode == "top_level":
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

        # add code to set
        if code not in unique_codes:
            unique_codes.append(code)

        # convert to lowercase and remove line breaks
        segment = process_segment(row['Segment'])

        # workaround for line breaks in cells
        if segment != "":
            if code != "":
                segments.append(segment)
                labels.append(code)
            else:
                segments[len(segments) - 1] += " " + segment

    # create dictionary for further processing of overlapping segments
    segments_dic = dict(zip(segments, labels))

    # figure out overlapping codes
    for i in range(0, len(segments)):
        for j in range(0, len(segments)):
            seg_a = segments[i]
            seg_b = segments[j]

            if seg_a == seg_b:
                continue

            label_a = labels[i]

            if seg_a in seg_b and seg_b in segments_dic:
                if label_a not in segments_dic[seg_b]:
                    segments_dic[seg_b] += " " + label_a

    # output code types
    print("Codes found:")
    column_labels = ['sentence', 'precursor', "successor"]
    for code in unique_codes:
        column_labels.append(code)
        print(code)

    # split dict items
    for key in segments_dic.keys():
        key_labels = segments_dic[key].split(' ')

        number_labels = []
        for code in unique_codes:
            number_labels.append(1 if code in key_labels else 0)

        segments_dic[key] = number_labels

    # load additional transcript information
    for t_segment in all_sentences:
        for l_segment in segments:
            # TODO check if behaviour is desired
            if t_segment not in l_segment and l_segment not in t_segment:
                continue

            # TODO get more performance out of this (e.g. just look at a sequence of words instead of chars)
            overlap, pos_a, pos_b = get_overlap(t_segment, l_segment)
            if len(overlap) < 15 and len(t_segment) > 0:
                # load segment with no code
                segments_dic[t_segment] = [0] * len(unique_codes)

    # calculate occurrences
    num_occurrences = numpy.zeros(len(unique_codes))
    for key in segments_dic.keys():
        number_labels = segments_dic[key]
        num_occurrences = num_occurrences + numpy.array(number_labels)

    # split dict items
    for key, value in segments_dic.items():

        overlap_length = 0
        overlap_transcript_id = -1
        overlap_pos_a = -1
        overlap_pos_b = -1

        # use transcript_id from code export
        for transcript_id in range(0, len(transcripts)):
            text = ' '.join(transcripts[transcript_id])
            overlap, pos_a, pos_b = get_overlap(text, key)
            if len(overlap) > overlap_length:
                overlap_length = len(overlap)
                overlap_transcript_id = transcript_id
                overlap_pos_a = pos_a
                overlap_pos_b = pos_b

        if overlap_transcript_id > 0:
            text = ' '.join(transcripts[overlap_transcript_id])
            precursor = text[:overlap_pos_a] + " "
            successor = " " + text[overlap_pos_b:]

            if precursor == " ":
                precursor = "-"

            if successor == " ":
                successor = "-"

            precursor_split = precursor.split(' ')
            if len(precursor_split) > MAX_LENGTH:
                precursor_split = precursor_split[-MAX_LENGTH:]
                precursor = ' '.join(precursor_split)

            successor_split = successor.split(' ')
            if len(successor_split) > MAX_LENGTH:
                successor_split = successor_split[0:MAX_LENGTH]
                successor = ' '.join(successor_split)

            segments_dic[key] = [precursor, successor] + value
        else:
            segments_dic[key] = ["-", "-"] + value

    # output imbalance
    print("imbalance:", num_occurrences)

    # create dataframe
    df = pd.DataFrame.from_dict(segments_dic, orient='index')
    df.reset_index(level=0, inplace=True)
    df.columns = column_labels

    # return
    return unique_codes, df


def load_html_transcripts(transcripts_dir="data_transcripts/katya/"):
    """
    preprocess is used to create the preprocessed data

    :param transcripts_dir: header for file loading
    """

    # init list
    sentences = []
    documents = []
    filenames = []

    # load transcript
    for filename in sorted(os.listdir(transcripts_dir)):
        filenames.append(filename)

        filepath = os.path.join(transcripts_dir, filename)

        if '.html' in filepath:
            document = load_transcript(filepath)

            sentences += document
            documents.append(document)

    # return sentence list
    return sentences, documents, filenames


def load_transcript(filepath):
    document = []

    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

        for i in range(0, len(lines)):
            for segment in re.findall(r'>(.*?)<', lines[i]):

                if 'Code interview:' in segment:
                    continue

                if 'Datum interview:' in segment:
                    continue

                if 'Duur interview:' in segment:
                    continue

                if 'Aanwezigen:' in segment:
                    continue

                if re.search('[a-zA-Z0-9-]', segment) is None:
                    continue

                segment = process_segment(segment)
                document.append(segment)

    return document

def process_segment(segment):
    # convert to lowercase and remove line breaks
    segment = segment.lower().replace("[\n\r]+", " ")

    # convert HTML spaces to normal spaces
    segment = segment.replace(" ", " ")

    # remove punctuation marks
    segment = re.sub(r'[,.?!\';:]', " ", segment)

    # trim string
    segment = re.sub(r' +', " ", segment)
    return segment.strip()


def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2))
    return s1[pos_a:pos_a + size], pos_a, pos_b


# for testing
if __name__ == '__main__':
    print("testing preprocessing")
    codes, df_out = preprocess(mode="top_level")

    # to csv
    df_out.to_csv('preprocessed.csv', index=False, header=True)

    print(df_out)