import json
import os
import numpy as np
import pandas as pd


def preprocess_sent(transcripts_dir="data_sentiment"):
    # init dictionary of all sentences
    segments = dict()

    # load responses
    for filename in os.listdir(transcripts_dir):
        filepath = os.path.join(transcripts_dir, filename)
        if '.json' in filepath:
            with open(filepath, encoding='cp1252') as f:
                data = json.load(f)

                # add the responses for each unique segment
                for key, value in data['answers'].items():
                    if value == "-1":   # negative
                        segments[key] = np.add(segments.get(key, [0, 0, 0]), [1, 0, 0])
                    elif value == "0":  # neutral
                        segments[key] = np.add(segments.get(key, [0, 0, 0]), [0, 1, 0])
                    else:  # positive
                        segments[key] = np.add(segments.get(key, [0, 0, 0]), [0, 0, 1])

    # calculate the sentiment value of each sentence
    # TODO decide maybe on multiclass ('negative', 'neutral', 'positive') instead of regression
    for key in segments.keys():
        # print(key, segments[key])
        value = segments[key]
        n = np.sum(value)
        val = np.sum(np.multiply(value, [-1, 0, 1])) / n
        segments[key] = [int(val < -0.3333), int(abs(val) <= 0.3333), int(val > 0.3333)]

    df = pd.DataFrame.from_dict(segments, orient='index')
    df.reset_index(level=0, inplace=True)
    df.columns = ['text', 'negative', 'neutral', 'positive']

    return df


# for testing
if __name__ == '__main__':
    print("testing preprocessing")

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    segments_df = preprocess()
    print(segments_df[:10])