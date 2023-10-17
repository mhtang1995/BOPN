import os
import argparse
import time
import torch
import torch.autograd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import Config
from src.data_loader import *
from src.utils import *
from src.model import Model
from src.trainer import Trainer



if __name__ == '__main__':
    # Config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/msra/main.json')
    args = parser.parse_args()
    config = Config(args)

    # Logger
    output_path = "./log/{}/{}/".format(config.dataset, config.exe)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = utils.get_logger(output_path)
    logger.info(config)
    config.logger = logger

    # Cuda
    if torch.cuda.is_available():
        torch.cuda.set_device(config.device)

    # Create Dataset
    logger.info("Loading Data")
    with open('./data/{}/train.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/{}/dev.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")
    config.tokenizer = tokenizer
    vocab = Vocabulary(config.dataset, config.window_size, config.offset_mode)
    if not vocab.load_label():
        vocab.build_label(train_data, dev_data, test_data)
    vocab.build_window()
    config.vocab = vocab
    if config.concat_data:
        train_data = concat_samples(train_data, './data/{}/concat_train.json'.format(config.dataset))
        dev_data = concat_samples(dev_data, './data/{}/concat_dev.json'.format(config.dataset))
        test_data = concat_samples(test_data, './data/{}/concat_test.json'.format(config.dataset))

    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab, is_train=True))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, collate_fn=collate_fn,
                              shuffle=True, num_workers=0, drop_last=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=config.batch_size, collate_fn=collate_fn,
                            shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, collate_fn=collate_fn,
                             shuffle=False, num_workers=0, drop_last=False)

    updates_total = len(train_dataset) // config.batch_size * config.epochs

    # Model
    logger.info("Building Model")
    if config.use_checkpoint:
        model = torch.load(os.path.join(output_path, config.save_path)).cuda(config.device)
    else:
        model = Model(config).cuda(config.device)
    trainer = Trainer(config, model, updates_total)

    best_dev = (0, 0, 0)
    best_test = (0, 0, 0)
    best_epoch = 0
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader)
        if i >= 0:
            config.logger.info("--------------On DEV----------------")
            dev_score = trainer.eval_with_offset(i, dev_loader)
            config.logger.info("--------------On TEST----------------")
            test_score = trainer.eval_with_offset(i, test_loader)
            if dev_score[0] > best_dev[0]:
                best_epoch = i
                best_dev = dev_score
                best_test = test_score
                torch.save(model, os.path.join(output_path, config.save_path))
                logger.info("Best Epoch Update!")
            else:
                logger.info("Best Epoch is still " + str(best_epoch))
    logger.info("Best DEV F1: {:3.4f}, P: {:3.4f}, R: {:3.4f}".format(best_dev[0], best_dev[1], best_dev[2]))
    logger.info("Best TEST F1: {:3.4f}, P: {:3.4f}, R: {:3.4f}".format(best_test[0], best_test[1], best_test[2]))

