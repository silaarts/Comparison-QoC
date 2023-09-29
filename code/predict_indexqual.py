import numpy as np
import pandas as pd
from IPython.display import display, HTML
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
from tqdm import trange
from transformers import RobertaModel
from tqdm import tqdm, trange


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class MultilabelClassification(nn.Module):
    def __init__(self, bert_model, num_labels, hidden_dropout_prob=0.1, hidden_size=768):
        super(MultilabelClassification, self).__init__()

        self.config = {}
        self.bert_model = bert_model

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.double_dense = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, a_input_ids, a_input_mask, b_input_ids, b_input_mask):
        x1 = self.bert_model(a_input_ids, attention_mask=a_input_mask)
        x2 = self.bert_model(b_input_ids, attention_mask=b_input_mask)

        x1 = self.dropout(x1.last_hidden_state)
        x1 = x1[:, 0, :]  # take <s> token (equiv. to [CLS])

        x2 = self.dropout(x2.last_hidden_state)
        x2 = x2[:, 0, :]  # take <s> token (equiv. to [CLS])

        x = torch.cat((x1, x2), dim=1)

        x = self.double_dense(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    df = pd.read_csv('preprocessed.csv')
    df = df[df["sentence"].str.len() < 512]

    display(df)

    sentences = list(df["sentence"])
    precursors = list(df["precursor"])

    display(sentences[0:5])

    flat_sentences = sentences
    sentences = list(chunks(sentences, 256))

    print(len(sentences))

    # select label columns
    cols = df.columns
    label_cols = list(cols[3:])
    num_labels = len(label_cols)
    print('Label columns: ', label_cols)
    classes = label_cols

    # set header for all label columns
    df['labels'] = list(df[label_cols].values)
    display(df.head())

    # get input and outputs
    labels = list(df.labels.values)
    display(labels[0:5])

    bert = RobertaModel.from_pretrained("pdelobelle/robbert-v2-dutch-base", add_pooling_layer=False)
    model = MultilabelClassification(bert, 4)
    model.load_state_dict(torch.load('./bert_model_multi_label_indexqual_37.49077490774908', map_location=torch.device('cuda')))
    model.to('cuda')

    tokenizer = RobertaTokenizer.from_pretrained('pdelobelle/robbert-v2-dutch-base')  # tokenizer

    p_embeddings = tokenizer(precursors, max_length=512,
                            truncation=True, padding=True, return_tensors='pt')
    s_embeddings = tokenizer(flat_sentences, max_length=512,
                            truncation=True, padding=True, return_tensors='pt')

    p_embeddings.to('cuda')
    s_embeddings.to('cuda')

    pred_labels = np.zeros((0, len(classes)))

    with torch.no_grad():
        for i in trange(len(pred_labels), len(flat_sentences)):
            p_embedding_input = p_embeddings.input_ids[i].unsqueeze(0)
            p_embedding_mask = p_embeddings.attention_mask[i].unsqueeze(0)
            s_embedding_input = s_embeddings.input_ids[i].unsqueeze(0)
            s_embedding_mask = s_embeddings.attention_mask[i].unsqueeze(0)

            pred_label = model(p_embedding_input,
                               p_embedding_mask,
                               s_embedding_input,
                               s_embedding_mask)

            pred_labels = np.concatenate((pred_labels, pred_label.cpu().detach().numpy()), axis=0)

    threshold = 0.50
    pred_bools = [pl > threshold for pl in pred_labels]
    
    np.savetxt("predictions.csv", pred_bools, delimiter=",")