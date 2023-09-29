import os
import numpy
import pandas as pd
import re
import difflib
from preprocess_indexqual import process_segment, get_overlap, load_html_transcripts


def preprocess_sentiment_test():
    """
    preprocess is used to create the preprocessed data

    :param mode: select a mode of preprocessing [default, top_level]
    :param deliminator: separator for file loading
    """

    # load MAXQDA data
    df = pd.read_excel("data_indexqual/indexqual_codes.xlsx")

    # load transcripts
    all_sentences, transcripts = load_html_transcripts()

    # initialise lists
    unique_codes = []
    segments = []
    labels = []

    for index, row in df.iterrows():
        # select only the top level code
        code = row['Code'].lower().replace(" ", "_").split("\\")[0]

        # continue if code is not about sentiment
        if 'positive' not in code and 'negative' not in code:
            continue

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

    # force these3 codes
    unique_codes = ['negative', 'neutral', 'positive']
    
    # output code types
    print("Codes found:")
    column_labels = ['sentence']
    for code in unique_codes:
        column_labels.append(code)
        print(code)

    # split dict items
    for key in segments_dic.keys():
        key_labels = segments_dic[key].split(' ')

        number_labels = []
        for code in unique_codes:
            number_labels.append(1 if code in key_labels else 0)

#         number_labels.insert(3, 0)
            
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
                segments_dic[t_segment] = [0,1,0] # neutral

    # calculate occurrences
    num_occurrences = numpy.zeros(len(unique_codes))
    for key in segments_dic.keys():
        number_labels = segments_dic[key]
        num_occurrences = num_occurrences + numpy.array(number_labels)

    # output imbalance
    print("imbalance:", num_occurrences)

    # create dataframe
    df = pd.DataFrame.from_dict(segments_dic, orient='index')
    df.reset_index(level=0, inplace=True)
    df.columns = column_labels

    # return
    return unique_codes, df


# for testing
if __name__ == '__main__':
    print("testing preprocessing")
    codes, df_out = preprocess_sentiment_test()

    # to csv
    df_out.to_csv('preprocessed_sentiment_test.csv', index=False, header=True)

    print(df_out)