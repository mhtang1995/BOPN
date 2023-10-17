import json


class Config:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.dataset = config["dataset"]
        self.exe = config["exe"]
        self.offset_mode = config["offset_mode"]

        self.save_path = config["save_path"]
        self.predict_path = config["predict_path"]
        self.use_checkpoint = config["use_checkpoint"]

        self.dist_emb_size = config["dist_emb_size"]
        self.lstm_hid_size = config["lstm_hid_size"]
        self.bert_hid_size = config["bert_hid_size"]
        self.biaffine_hid_size = config['biaffine_hid_size']
        self.window_size = config["window_size"]

        self.emb_dropout = config["emb_dropout"]
        self.out_dropout = config["out_dropout"]
        self.loss_epsilon = config["loss_epsilon"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.device = config["device"]

        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.clip_grad_norm = config["clip_grad_norm"]
        self.bert_name = config["bert_name"]
        self.bert_learning_rate = config["bert_learning_rate"]
        self.warm_factor = config["warm_factor"]

        self.concat_data = config["concat_data"]
        self.parse_offset = config["parse_offset"]
        self.use_grid = config['use_grid']
        self.use_conv = config['use_conv']
        self.seed = config["seed"]

    def __repr__(self):
        return "{}".format(self.__dict__.items())
