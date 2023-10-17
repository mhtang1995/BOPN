import json
import math
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os

from src import utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"
dis2idx = np.zeros((1500), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):

    def __init__(self, dataset, window_size, offset_mode):
        self.dataset = dataset
        self.window_size = window_size
        self.offset_mode = offset_mode
        if self.offset_mode == "both":
            self.start = True
            self.end = True
        elif self.offset_mode == 'start':
            self.start = True
            self.end = False
        elif self.offset_mode == 'end':
            self.start = False
            self.end = True
        elif self.offset_mode == 'none':
            self.start = False
            self.end = False
        else:
            raise Exception
        self.label2id = {}
        self.id2label = {}
        self.window2id = {'pad': 0}
        self.id2window = {0: 'pad'}

    def build_window(self):
        for i in range(-self.window_size, self.window_size + 1):
            if i != 0:
                if self.start:
                    self.add_window('s:' + str(i))
                if self.end:
                    self.add_window('e:' + str(i))
            else:
                self.add_window('c:' + str(0))

    def load_label(self):
        if os.path.exists('./data/{}/ontology.json'.format(self.dataset)):
            with open('./data/{}/ontology.json'.format(self.dataset), 'r', encoding='utf-8') as f:
                infos = json.load(f)
                self.id2label = infos["id2label"]
                self.label2id = infos["label2id"]
            return True
        else:
            return False

    def build_label(self, train_data, dev_data, test_data):
        train_ent_num = self.count(train_data)
        dev_ent_num = self.count(dev_data)
        test_ent_num = self.count(test_data)
        with open('./data/{}/ontology.json'.format(self.dataset), 'w', encoding='utf-8') as f:
            infos = {
                "train": [len(train_data), train_ent_num],
                "dev": [len(dev_data), dev_ent_num],
                "test": [len(test_data), test_ent_num],
                "label2id": self.label2id,
                "id2label": self.id2label
            }
            f.write(json.dumps(infos))

    def count(self, samples):
        entity_num = 0
        for instance in samples:
            for entity in instance["ner"]:
                self.add_label(entity["type"])
            entity_num += len(instance["ner"])
        return entity_num


    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label
        assert label == self.id2label[self.label2id[label]]

    def add_window(self, window):
        if window not in self.window2id:
            self.window2id[window] = len(self.window2id)
            self.id2window[self.window2id[window]] = window
        assert window == self.id2window[self.window2id[window]]

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]

    def window_to_id(self, window):
        return self.window2id[window]

    def id_to_window(self, i):
        return self.id2window[i]

    def len_of_windows(self):
        return len(self.window2id)

    def __len__(self):
        return len(self.label2id)


class RelationDataset(Dataset):

    def __init__(self, bert_inputs, pieces2word, sent_length,
                 grid_labels, grid_mask2d, dist_inputs, entity_text):
        self.bert_inputs = bert_inputs
        self.pieces2word = pieces2word
        self.sent_length = sent_length
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.entity_text = entity_text
        self.dist_inputs = dist_inputs

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               self.sent_length[item], \
               torch.LongTensor(self.grid_labels[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               self.entity_text[item]

    def __len__(self):
        return len(self.bert_inputs)


def collate_fn(data):
    bert_inputs, pieces2word, sent_length, grid_labels, grid_mask2d, dist_inputs, entity_text = map(list, zip(*data))

    max_tok = np.max(sent_length)
    type_size = grid_labels[0].size(0)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            if len(x.shape) == 2:
                new_data[j, :x.shape[0], :x.shape[1]] = x
            if len(x.shape) == 3:
                new_data[j, :x.shape[0], :x.shape[1], :x.shape[2]] = x
            if len(x.shape) == 4:
                new_data[j, :x.shape[0], :x.shape[1], :x.shape[2], :x.shape[3]] = x
        return new_data

    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)
    grid_labels_mat = torch.zeros((batch_size, type_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, grid_labels_mat)
    grid_mask2d_mat = torch.zeros((batch_size, type_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, grid_mask2d_mat)
    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)

    return bert_inputs, pieces2word, sent_length, grid_labels, grid_mask2d, dist_inputs, entity_text



def encode_offset(vocab, centers, grid_labels, length):

    for center in centers:
        ty = center[0].item()
        cur = center[1].item()
        per = center[2].item()
        if cur > per:
            continue

        for window in reversed(range(-vocab.window_size, 0)):
            start_boundary_stop = False
            end_boundary_stop = False
            # start
            if vocab.start:
                if cur + window < 0:
                    start_boundary_stop = True
                elif grid_labels[ty, cur + window, per] != 0:
                    distance = int(vocab.id_to_window(grid_labels[ty, cur + window, per]).split(":")[-1])
                    if math.fabs(window) >= math.fabs(distance):
                        start_boundary_stop = True
                if not start_boundary_stop:
                    grid_labels[ty, cur + window, per] = vocab.window_to_id('s:' + str(window))
                    grid_labels[ty, per, cur + window] = vocab.window_to_id('s:' + str(window))

            # end
            if vocab.end:
                if per + window < cur:
                    end_boundary_stop = True
                elif grid_labels[ty, cur, per + window] != 0:
                    distance = int(vocab.id_to_window(grid_labels[ty, cur, per + window]).split(":")[-1])
                    if math.fabs(window) >= math.fabs(distance):
                        end_boundary_stop = True
                if not end_boundary_stop:
                    grid_labels[ty, cur, per + window] = vocab.window_to_id('e:' + str(window))
                    grid_labels[ty, per + window, cur] = vocab.window_to_id('e:' + str(window))

        for window in range(1, vocab.window_size + 1):
            start_boundary_stop = False
            end_boundary_stop = False
            if vocab.start:
                # start
                if cur + window > per:
                    start_boundary_stop = True
                elif grid_labels[ty, cur + window, per] != 0:
                    distance = int(vocab.id_to_window(grid_labels[ty, cur + window, per]).split(":")[-1])
                    if math.fabs(window) >= math.fabs(distance):
                        start_boundary_stop = True
                if not start_boundary_stop:
                    grid_labels[ty, cur + window, per] = vocab.window_to_id('s:' + str(window))
                    grid_labels[ty, per, cur + window] = vocab.window_to_id('s:' + str(window))

            # end
            if vocab.end:
                if per + window >= length:
                    end_boundary_stop = True
                elif grid_labels[ty, cur, per + window] != 0:
                    distance = int(vocab.id_to_window(grid_labels[ty, cur, per + window]).split(":")[-1])
                    if math.fabs(window) >= math.fabs(distance):
                        end_boundary_stop = True
                if not end_boundary_stop:
                    grid_labels[ty, cur, per + window] = vocab.window_to_id('e:' + str(window))
                    grid_labels[ty, per + window, cur] = vocab.window_to_id('e:' + str(window))
    return grid_labels


def process_bert(data, tokenizer, vocab, is_train=False):

    bert_inputs, pieces2word = [], []
    sent_length, grid_labels, grid_mask2d, dist_inputs, entity_text = [], [], [], [], []

    # if is_train:
    #     percent = int(0.125 * len(data))
    #     data = data[0:percent]

    for instance in data:
        if len(instance['sentence']) == 0:
            continue

        sentence = instance['sentence']
        _sent_length = len(instance['sentence'])

        # Tokenizer sentence
        tokens = [tokenizer.tokenize(word) for word in sentence]
        pieces = [piece for pieces in tokens for piece in pieces]
        if len(pieces) > 512 - len(vocab.label2id):
            # print("Find a over-length sample!")
            continue

        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])
        _pieces2word = np.zeros((_sent_length, len(_bert_inputs)), dtype=np.bool)
        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0]+1:pieces[-1]+2] = 1
                start += len(pieces)

        # Encode entity span
        _grid_labels = np.zeros((len(vocab), _sent_length, _sent_length), dtype=np.int)
        for entity in instance["ner"]:
            start, end = entity["index"][0], entity["index"][-1]
            _grid_labels[vocab.label_to_id(entity["type"]), start, end] = vocab.window_to_id('c:0')
            _grid_labels[vocab.label_to_id(entity["type"]), end, start] = vocab.window_to_id('c:0')

        _grid_labels_tensor = torch.LongTensor(_grid_labels)
        centers = torch.nonzero(_grid_labels_tensor == vocab.window_to_id('c:0'), as_tuple=False)

        _grid_labels = encode_offset(vocab, centers, _grid_labels, _sent_length)

        _dist_inputs = np.zeros((_sent_length, _sent_length), dtype=np.int)
        for k in range(_sent_length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(_sent_length):
            for j in range(_sent_length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19

        # Encode mask
        _grid_mask2d = np.ones((len(vocab), _sent_length, _sent_length), dtype=np.bool)

        # Record entity
        _entity_text = set([utils.convert_index_to_text(e["index"], vocab.label_to_id(e["type"]))
                            for e in instance["ner"]])

        bert_inputs.append(_bert_inputs)
        pieces2word.append(_pieces2word)
        sent_length.append(_sent_length)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        entity_text.append(_entity_text)

    return bert_inputs, pieces2word, sent_length, grid_labels, grid_mask2d, dist_inputs, entity_text


def concat_samples(samples, save_file=None):
    # concat samples for saving computing resources
    if not os.path.exists(save_file):
        new_samples = []
        new_lengths = []
        lengths = []
        for sample in samples:
            lengths.append(len(sample['sentence']))

        concat_length = []
        concat_id = []
        for i, value in enumerate(lengths):
            sum_value = sum(concat_length)
            if sum_value + value <= 100:
                concat_length.append(value)
                concat_id.append(i)
            else:
                new_lengths.append(concat_id)
                concat_length = [value]
                concat_id = [i]

        for concat_id in new_lengths:
            sentence = []
            ner = []
            ner_start = 0
            for i in concat_id:
                sentence.extend(samples[i]["sentence"])
                for entity in samples[i]["ner"]:
                    ner.append({"index": [index+ner_start for index in entity["index"]], "type": entity['type']})
                ner_start += len(samples[i]["sentence"])
            new_samples.append({"sentence": sentence, "ner": ner})
        with open(save_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(new_samples, ensure_ascii=False))
    else:
        with open(save_file, 'r', encoding='utf-8') as f:
            new_samples = json.load(f)
    return new_samples

