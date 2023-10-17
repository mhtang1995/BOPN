import os
import logging
import pickle
import prettytable as pt
import torch
from collections import defaultdict, deque


def get_logger(output_path, name='log.txt'):
    pathname = os.path.join(output_path, name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_node_to_text(start, end, ty):
    text = "-".join([str(i) for i in range(start, end + 1)])
    text = text + "-#-{}".format(ty)
    return text



def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)



class Record(object):
    def __init__(self, windows):
        self.r_num = 0
        self.p_num = 0
        self.c_num = 0
        self.windows = windows
        self.r_num_group = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        self.p_num_group = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        self.c_num_group = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

        self.r_num_offset = {}
        self.p_num_offset = {}
        self.c_num_offset = {}
        self.init_offset(windows)

    def init_offset(self, windows):
        for window in windows:
            if window != 'pad':
                self.r_num_offset[window] = 0
                self.p_num_offset[window] = 0
                self.c_num_offset[window] = 0


    def cal_f1(self):
        if self.r_num == 0 or self.p_num == 0:
            return 0, 0, 0

        recall = self.c_num / self.r_num if self.r_num else 0
        precision = self.c_num / self.p_num if self.p_num else 0

        if recall and precision:
            return 2 * precision * recall / (precision + recall), precision, recall
        return 0, precision, recall


    def record(self, entities, entity_text):
        for entity, ent_set in zip(entities, entity_text):
            r_num = len(ent_set)
            p_num = len(entity)
            c_num = len(entity.intersection(ent_set))
            self.p_num = self.p_num + p_num
            self.r_num = self.r_num + r_num
            self.c_num = self.c_num + c_num


    def record_by_length(self, entities, entity_text):
        for entity, ent_set in zip(entities, entity_text):
            predict = {1: set(), 2: set(), 3: set(), 4: set(), 5: set(), 6: set(), 7: set(), 8: set(), 9: set()}
            golden = {1: set(), 2: set(), 3: set(), 4: set(), 5: set(), 6: set(), 7: set(), 8: set(), 9: set()}
            for p in entity:
                index, _ = convert_text_to_index(text=p)
                if len(index) > 8:
                    predict[9].add(p)
                else:
                    predict[len(index)].add(p)


            for p in ent_set:
                index, _ = convert_text_to_index(text=p)
                if len(index) > 8:
                    golden[9].add(p)
                else:
                    golden[len(index)].add(p)


            for i in range(1, 10):
                r_num = len(golden[i])
                p_num = len(predict[i])
                c_num = len(predict[i].intersection(golden[i]))
                self.p_num_group[i] = self.p_num_group[i] + p_num
                self.r_num_group[i] = self.r_num_group[i] + r_num
                self.c_num_group[i] = self.c_num_group[i] + c_num


    def record_by_offset(self, offset_entities, golden_entities):
        for entity, ent_set in zip(offset_entities, golden_entities):
            for window in self.windows:
                if window != 'pad':
                    predict = entity.get(window) or set()
                    golden = ent_set.get(window) or set()

                    r_num = len(golden)
                    p_num = len(predict)
                    c_num = len(predict.intersection(golden))
                    self.p_num_offset[window] = self.p_num_offset[window] + p_num
                    self.r_num_offset[window] = self.r_num_offset[window] + r_num
                    self.c_num_offset[window] = self.c_num_offset[window] + c_num


    def log(self, epoch, logger):
        f1, precision, recall = self.cal_f1()
        table = pt.PrettyTable(["{}".format(epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [f1, precision, recall]])
        logger.info("\n{}".format(table))
        return f1, precision, recall


    def log_group_by_length(self, logger):
        table = pt.PrettyTable(["Length", 'F1', "Precision", "Recall", "Support"])

        for i in range(1, 10):
            r_num = self.r_num_group[i]
            c_num = self.c_num_group[i]
            p_num = self.p_num_group[i]
            if r_num == 0 or p_num == 0 or c_num == 0:
                f1 = 0
                recall = 0
                precision = 0
            else:
                recall = c_num / r_num if r_num else 0
                precision = c_num / p_num if p_num else 0
                f1 = 2 * precision * recall / (precision + recall)
            table.add_row(["{}".format(i)] + ["{:3.4f}".format(x) for x in [f1, precision, recall]] + ["{}".format(r_num)])

        logger.info("\n{}".format(table))

    def log_group_by_offset(self, logger):
        table = pt.PrettyTable(["Offset", 'F1', "Precision", "Recall", "Support"])

        for window in self.windows:
            if window != 'pad':
                r_num = self.r_num_offset[window]
                c_num = self.c_num_offset[window]
                p_num = self.p_num_offset[window]
                if r_num == 0 or p_num == 0 or c_num == 0:
                    f1 = 0
                    recall = 0
                    precision = 0
                else:
                    recall = c_num / r_num if r_num else 0
                    precision = c_num / p_num if p_num else 0
                    f1 = 2 * precision * recall / (precision + recall)
                table.add_row([window] + ["{:3.4f}".format(x) for x in [f1, precision, recall]] + ["{}".format(r_num)])

        logger.info("\n{}".format(table))



def decode_with_rule(vocab, outputs, grid_labels, entity_text, length):
    center_entities = []
    offset_entities = []
    for instance, grid_labels, ent, l in zip(outputs, grid_labels, entity_text, length):
        center_entity = set()
        offset_entity = set()
        start = {}
        end = {}
        for cur in range(l):
            for per in range(cur, l):
                for ty in range(len(vocab.label2id)):
                    window_id = instance[ty, cur, per]
                    if window_id > 0:
                        window = vocab.id_to_window(window_id).split(':')

                        if window[-1] == '0':
                            text = convert_node_to_text(cur, per, ty)
                            center_entity.add(text)
                            offset_entity.add(text)

                        elif window[0] == 's':
                            if 0 <= cur - int(window[-1]) <= per:
                                text = convert_node_to_text(cur - int(window[-1]), per, ty)
                                if text not in start:
                                    start[text] = [int(window[-1])]
                                else:
                                    start[text].append(int(window[-1]))

                        elif window[0] == 'e':
                            if cur <= per - int(window[-1]) < l:
                                text = convert_node_to_text(cur, per - int(window[-1]), ty)
                                if text not in end:
                                    end[text] = [int(window[-1])]
                                else:
                                    end[text].append(int(window[-1]))

        for text, offsets in start.items():
            if text in offset_entity:
                continue
            # offset_entity.add(text)
            if len(offsets) >= vocab.window_size:
                offset_entity.add(text)

        for text, offsets in end.items():
            if text in offset_entity:
                continue
            # offset_entity.add(text)
            if len(offsets) >= vocab.window_size:
                offset_entity.add(text)

        center_entities.append(center_entity)
        offset_entities.append(offset_entity)
    return center_entities, offset_entities


def decode_group_by_offset(vocab, outputs, grid_labels, entity_text, length):

    offset_entities = []
    golden_entities = []
    for instance, grid_labels, ent, l in zip(outputs, grid_labels, entity_text, length):
        offset_entity = {}
        golden_entity = {}

        for cur in range(l):
            for per in range(cur, l):
                for ty in range(len(vocab.label2id)):
                    predict = instance[ty, cur, per]
                    if predict > 0:
                        window = vocab.id_to_window(predict)
                        window_split = window.split(':')
                        if window not in offset_entity:
                            offset_entity[window] = set()

                        if window_split[-1] == '0':
                            text = convert_node_to_text(cur, per, ty)
                            offset_entity[window].add(text)
                        elif window_split[0] == 's':
                            if 0 <= cur - int(window_split[-1]) <= per:
                                text = convert_node_to_text(cur - int(window_split[-1]), per, ty)
                                offset_entity[window].add(text)
                        elif window_split[0] == 'e':
                            if cur <= per - int(window_split[-1]) < l:
                                text = convert_node_to_text(cur, per - int(window_split[-1]), ty)
                                offset_entity[window].add(text)
                    golden = grid_labels[ty, cur, per]
                    if golden > 0:
                        window = vocab.id_to_window(golden)
                        window_split = window.split(':')
                        if window not in golden_entity:
                            golden_entity[window] = set()

                        if window_split[-1] == '0':
                            text = convert_node_to_text(cur, per, ty)
                            golden_entity[window].add(text)
                        elif window_split[0] == 's':
                            if 0 <= cur - int(window_split[-1]) <= per:
                                text = convert_node_to_text(cur - int(window_split[-1]), per, ty)
                                golden_entity[window].add(text)
                        elif window_split[0] == 'e':
                            if cur <= per - int(window_split[-1]) < l:
                                text = convert_node_to_text(cur, per - int(window_split[-1]), ty)
                                golden_entity[window].add(text)

        offset_entities.append(offset_entity)
        golden_entities.append(golden_entity)
    return offset_entities, golden_entities