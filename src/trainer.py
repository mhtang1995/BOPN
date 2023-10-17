import os
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn

import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from tqdm import tqdm

from src.model import CELoss
from src import utils


def build_optimizer_scheduler(config, model, updates_total):
    bert_params = set(model.bert_encoder.bert.parameters())
    other_params = list(set(model.parameters()) - bert_params)
    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {'params': [p for n, p in model.bert_encoder.bert.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': config.bert_learning_rate,
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.bert_encoder.bert.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': config.bert_learning_rate,
         'weight_decay': 0.0},
        {'params': other_params,
         'lr': config.learning_rate,
         'weight_decay': config.weight_decay},
    ]

    optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warm_factor * updates_total,
        num_training_steps=updates_total
    )
    return optimizer, scheduler


class Trainer(object):
    def __init__(self, config, model, updates_total):
        self.config = config
        self.model = model
        self.optimizer, self.scheduler = build_optimizer_scheduler(config, model, updates_total)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.loss_epsilon)

    def train(self, epoch, data_loader):

        self.model.train()
        for data_batch in tqdm(data_loader, desc="Train Batches: "):
            data_batch = [data.cuda(self.config.device) for data in data_batch[:-1]]
            bert_inputs, pieces2word, sent_length, grid_labels, grid_mask2d, dist_inputs = data_batch

            outputs = self.model(bert_inputs, pieces2word, sent_length, dist_inputs, grid_mask2d)
            loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()


    def eval_with_offset(self, epoch, data_loader):
        self.model.eval()

        record_with_rules = utils.Record(self.config.vocab.window2id.keys())

        with torch.no_grad():
            for data_batch in tqdm(data_loader, desc="Test Batches: "):
                entity_text = data_batch[-1]
                data_batch = [data.cuda(self.config.device) for data in data_batch[:-1]]
                bert_inputs, pieces2word, sent_length, grid_labels, grid_mask2d, dist_inputs = data_batch

                outputs = self.model(bert_inputs, pieces2word, sent_length, dist_inputs, grid_mask2d)
                outputs = torch.argmax(outputs, -1)

                center_entities, offset_entities = utils.decode_with_rule(
                    vocab=self.config.vocab,
                    outputs=outputs.cpu().numpy(),
                    grid_labels=grid_labels.cpu().numpy(),
                    entity_text=entity_text,
                    length=sent_length.cpu().numpy())
                if self.config.parse_offset:
                    record_with_rules.record(offset_entities, entity_text)
                else:
                    record_with_rules.record(center_entities, entity_text)

        f1, precision, recall = record_with_rules.log(epoch, self.config.logger)

        return f1, precision, recall

    def eval_group_by_length(self, epoch, data_loader):
        self.model.eval()

        record_with_rules = utils.Record(self.config.vocab.window2id.keys())

        with torch.no_grad():
            for data_batch in tqdm(data_loader, desc="Test Batches: "):
                entity_text = data_batch[-1]
                data_batch = [data.cuda(self.config.device) for data in data_batch[:-1]]
                bert_inputs, pieces2word, sent_length, grid_labels, grid_mask2d, dist_inputs = data_batch

                outputs = self.model(bert_inputs, pieces2word, sent_length, dist_inputs, grid_mask2d)
                outputs = torch.argmax(outputs, -1)

                center_entities, offset_entities = utils.decode_with_rule(
                    vocab=self.config.vocab,
                    outputs=outputs.cpu().numpy(),
                    grid_labels=grid_labels.cpu().numpy(),
                    entity_text=entity_text,
                    length=sent_length.cpu().numpy())
                if not self.config.parse_offset:
                    record_with_rules.record_by_length(offset_entities, entity_text)
                else:
                    record_with_rules.record_by_length(center_entities, entity_text)

        record_with_rules.log_group_by_length(self.config.logger)

    def eval_group_by_offset(self, epoch, data_loader):
        self.model.eval()

        record_with_rules = utils.Record(self.config.vocab.window2id.keys())

        with torch.no_grad():
            for data_batch in tqdm(data_loader, desc="Test Batches: "):
                entity_text = data_batch[-1]
                data_batch = [data.cuda(self.config.device) for data in data_batch[:-1]]
                bert_inputs, pieces2word, sent_length, grid_labels, grid_mask2d, dist_inputs = data_batch

                outputs = self.model(bert_inputs, pieces2word, sent_length, dist_inputs, grid_mask2d)
                outputs = torch.argmax(outputs, -1)

                offset_entities, golden_entities = utils.decode_group_by_offset(
                    vocab=self.config.vocab,
                    outputs=outputs.cpu().numpy(),
                    grid_labels=grid_labels.cpu().numpy(),
                    entity_text=entity_text,
                    length=sent_length.cpu().numpy())

                record_with_rules.record_by_offset(offset_entities, golden_entities)

        record_with_rules.log_group_by_offset(self.config.logger)