import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel


def combine(sub, sup_mask, pool_type="max"):
    """ Combine different level representations """
    sup = None
    if pool_type == "mean":
        size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
        m = (sup_mask.unsqueeze(-1) == 1).float()
        sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
        sup = sup.sum(dim=2) / size
    if pool_type == "sum":
        m = (sup_mask.unsqueeze(-1) == 1).float()
        sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
        sup = sup.sum(dim=2)
    if pool_type == "max":
        # 实际效果是取mask对应true的值
        # b*token_size*context_size to b*token_size*context_size*1
        m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
        sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
        sup = sup.max(dim=2)[0]  # b*token_size*bert_embedding_size
        sup[sup == -1e30] = 0

    return sup


class BertEncoder(nn.Module):

    def __init__(self, bert_name, vocab, bert_hid_size, lstm_hid_size, emb_dropout=0.4):
        super(BertEncoder, self).__init__()

        self.bert = AutoModel.from_pretrained(bert_name, cache_dir="./cache/", output_hidden_states=True)
        self.lstm = nn.LSTM(bert_hid_size, lstm_hid_size // 2, batch_first=True, bidirectional=True, num_layers=1)
        self.prompt = nn.Parameter(torch.randn(len(vocab), bert_hid_size), requires_grad=True)
        # self.prompt = nn.Parameter(torch.randn(len(vocab), lstm_hid_size), requires_grad=True)
        self.dropout = nn.Dropout(emb_dropout)


    def forward(self, bert_inputs, pieces2word, sent_length):

        prompt_embeddings = self.prompt[:, :].unsqueeze(0).expand(bert_inputs.size(0), -1, -1)
        sentence_embeddings = self.bert.get_input_embeddings()(bert_inputs)

        # token_embeddings = self.bert(
        #     inputs_embeds=sentence_embeddings,
        #     attention_mask=bert_inputs.ne(0).float())[0]
        # token_embeddings = combine(token_embeddings, pieces2word, pool_type='mean')
        # token_embeddings = self.dropout(token_embeddings)
        # packed_embeddings = pack_padded_sequence(token_embeddings, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        # packed_outs, (hidden, _) = self.lstm(packed_embeddings)
        # token_embeddings, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())

        prompt_attention = torch.ones_like(prompt_embeddings[..., :1]).squeeze(-1)
        bert_embeddings = self.bert(
            inputs_embeds=torch.cat([prompt_embeddings, sentence_embeddings], dim=1),
            attention_mask=torch.cat([prompt_attention, bert_inputs], dim=-1).ne(0).float()
        )[0]
        # bert_embeddings = torch.stack(bert_embeddings[2][-4:], dim=-1).mean(-1)

        prompt_embeddings = bert_embeddings[:, :prompt_embeddings.size(1), :]
        token_embeddings = bert_embeddings[:, prompt_embeddings.size(1):, :]
        token_embeddings = combine(token_embeddings, pieces2word, pool_type='mean')

        # Through LSTM
        sent_length = sent_length + prompt_embeddings.size(1)
        lstm_embeddings = torch.cat([prompt_embeddings, token_embeddings], dim=1)
        lstm_embeddings = self.dropout(lstm_embeddings)
        packed_embeddings = pack_padded_sequence(lstm_embeddings, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.lstm(packed_embeddings)
        lstm_embeddings, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())
        prompt_embeddings = lstm_embeddings[:, :prompt_embeddings.size(1), :]
        token_embeddings = lstm_embeddings[:, prompt_embeddings.size(1):, :]

        return prompt_embeddings, token_embeddings



class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, epsilon=None):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        self.beta = nn.Parameter(torch.zeros(input_dim))
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
        self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.constant_(self.beta_dense.weight, 0)
        torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        # Norm
        outputs = inputs
        mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
        outputs = outputs - mean
        variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
        std = (variance + self.epsilon) ** 0.5
        outputs = outputs / std

        # Add Cond
        for _ in range(len(inputs.shape) - len(cond.shape)):
            cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)
        beta = self.beta_dense(cond) + self.beta
        gamma = self.gamma_dense(cond) + self.gamma
        outputs = outputs * gamma
        outputs = outputs + beta

        return outputs



class GridEncoder(nn.Module):

    def __init__(self, dist_emb_size):
        super(GridEncoder, self).__init__()
        self.distance_embeddings = nn.Embedding(20, dist_emb_size)
        self.region_embeddings = nn.Embedding(3, dist_emb_size)

    def forward(self, dist_inputs, grid_mask2d, cln):
        dis_emb = self.distance_embeddings(dist_inputs)
        tril_mask = torch.triu(grid_mask2d[:, 0, :, :].squeeze(1).long())
        reg_inputs = tril_mask + grid_mask2d.clone()[:, 0, :, :].squeeze(1).long()
        reg_emb = self.region_embeddings(reg_inputs)
        return torch.cat([cln, dis_emb, reg_emb], dim=-1)



class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.1):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.ReLU()
        self.reinit_layer_(self.linear, "relu")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.activation(self.linear(self.dropout(x)))

    def reinit_layer_(self, layer: torch.nn.Module, nonlinearity='relu'):
        for name, param in layer.named_parameters():
            if name.startswith('bias'):
                torch.nn.init.zeros_(param.data)
            elif name.startswith('weight'):
                if nonlinearity.lower() in ('relu', 'leaky_relu'):
                    torch.nn.init.kaiming_uniform_(param.data, nonlinearity=nonlinearity)
                elif nonlinearity.lower() in ('glu',):
                    torch.nn.init.xavier_uniform_(param.data, gain=torch.nn.init.calculate_gain('sigmoid'))
                else:
                    torch.nn.init.xavier_uniform_(param.data, gain=torch.nn.init.calculate_gain(nonlinearity))
            else:
                raise TypeError(f"Invalid Layer {layer}")


class Biaffine(nn.Module):
    def __init__(self, left_dim, right_dim, biaffine_in=150, biaffine_out=1, bias_x=True, bias_y=True, dropout=0.33):
        super(Biaffine, self).__init__()
        self.left = MLP(left_dim, biaffine_in, dropout=dropout)
        self.right = MLP(right_dim, biaffine_in, dropout=dropout)
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((biaffine_out, biaffine_in + int(bias_x), biaffine_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)


    def forward(self, x, y):
        x = self.left(x)
        y = self.right(y)
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        dilation = [1, 2]
        self.base = nn.Sequential(
            nn.Dropout3d(dropout),
            nn.Conv3d(input_size, input_size, kernel_size=1),
            nn.GELU(),
        )


        self.convs = nn.ModuleList(
            [nn.Conv3d(input_size, input_size, kernel_size=3, groups=input_size, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)

        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 4, 1).contiguous()
        return outputs


class OffsetOpt(nn.Module):
    def __init__(self, config):
        super(OffsetOpt, self).__init__()

        self.conv = ConvolutionLayer(input_size=len(config.vocab.id2window), dropout=0.2)
        self.predictor = nn.Linear(len(config.vocab.id2window)*2, len(config.vocab.id2window))

    def forward(self, inputs):
        outputs = self.predictor(self.conv(inputs))
        return outputs


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.use_grid = config.use_grid
        self.use_conv = config.use_conv

        self.bert_encoder = BertEncoder(
            bert_name=config.bert_name,
            vocab=config.vocab,
            bert_hid_size=config.bert_hid_size,
            lstm_hid_size=config.lstm_hid_size,
            emb_dropout=config.emb_dropout)
        self.w2w = LayerNorm(config.lstm_hid_size, config.lstm_hid_size)
        if self.use_grid:
            self.grid_encoder = GridEncoder(config.dist_emb_size)
            grid_dim = config.lstm_hid_size + 2*config.dist_emb_size
        else:
            grid_dim = config.lstm_hid_size


        self.biaffine = Biaffine(
            left_dim=config.lstm_hid_size,
            right_dim=grid_dim,
            biaffine_in=config.biaffine_hid_size,
            biaffine_out=len(config.vocab.id2window),
            dropout=config.out_dropout
        )

        self.p2w2w = LayerNorm(config.lstm_hid_size, grid_dim)
        self.distance = nn.Linear(config.lstm_hid_size, len(config.vocab.id2window))

        if self.use_conv:
            self.opt = OffsetOpt(config)


    def forward(self, bert_inputs, pieces2word, sent_length, dist_inputs, grid_mask2d):
        prompt_embeddings, token_embeddings = self.bert_encoder(bert_inputs, pieces2word, sent_length)
        w2w = self.w2w(token_embeddings.unsqueeze(2), token_embeddings.unsqueeze(1))
        if self.use_grid:
            w2w = self.grid_encoder(dist_inputs, grid_mask2d, w2w)

        batch_size = bert_inputs.size(0)
        length = w2w.size(1)
        pro = prompt_embeddings.size(1)
        outputs = self.biaffine(
            prompt_embeddings,
            w2w.reshape(batch_size, length * length, -1)
        ).reshape(batch_size, pro, length, length, -1)
        p2w2w = self.p2w2w(prompt_embeddings.unsqueeze(-2).unsqueeze(-2), w2w.unsqueeze(1))
        outputs_linear = self.distance(p2w2w)
        outputs = outputs + outputs_linear
        if self.use_conv:
            outputs_nor = self.opt(outputs)
            return outputs + outputs_nor
        return outputs






class CELoss(nn.Module):
    """Cross Entropy Loss"""
    def __init__(self, label_smoothing: float = 0.0):
        super(CELoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, pred, target, mask):
        num_classes = pred.size(dim=-1)
        log_prob = pred.log_softmax(dim=-1)
        one_hot_target = torch.nn.functional.one_hot(target, num_classes=num_classes)
        smooth_target = one_hot_target * (1-self.label_smoothing) + self.label_smoothing/num_classes
        loss = -(log_prob[mask] * smooth_target[mask]).sum(dim=-1)
        return loss.mean()




