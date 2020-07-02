#
# Copyright (c) 2020 Adobe Systems Incorporated. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch.nn as nn
import optparse
import sys
import json
import numpy as np
import random
import tqdm

sys.path.append('')


class Encodings:
    def __init__(self, filename=None):
        self._char2int = {'<PAD>': 0, '<UNK>': 1}
        self._type2int = {}
        self._subtype2int = {'<UNK>': 0}  # this will not get backpropagated
        self._type_list = []
        self._subtype_list = []
        if filename is not None:
            self.load(filename)

    def save(self, filename):
        json.dump({'char2int': self._char2int, 'type2int': self._type2int, 'subtype2int': self._subtype2int},
                  open(filename, 'w'))

    def load(self, file):
        if isinstance(file, str):
            stream = open(file, 'r')
        else:
            stream = file
        obj = json.load(stream)
        self._char2int = obj['char2int']
        self._type2int = obj['type2int']
        self._subtype2int = obj['subtype2int']
        self._type_list = [None for _ in range(len(self._type2int))]
        self._subtype_list = [None for _ in range(len(self._subtype2int))]
        for t in self._type2int:
            self._type_list[self._type2int[t]] = t

        for t in self._subtype2int:
            self._subtype_list[self._subtype2int[t]] = t

    def update_encodings(self, dataset, cutoff=2):
        char2count = {}
        for entry in dataset:
            domain = entry[0]
            ttype = entry[1]
            tsubtype = entry[2]
            for char in domain:
                char = char.lower()
                if char in char2count:
                    char2count[char] += 1
                else:
                    char2count[char] = 1
            if ttype not in self._type2int:
                self._type2int[ttype] = len(self._type2int)
                self._type_list.append(ttype)
            if tsubtype not in self._subtype2int:
                self._subtype2int[tsubtype] = len(self._subtype2int)
                self._subtype_list.append(tsubtype)

        for char in char2count:
            if char not in self._char2int:
                self._char2int[char] = len(self._char2int)


class AwDoCConfig:
    def __init__(self):
        self.char_emb_size = 100
        self.rnn_layers = 2
        self.rnn_size = 100
        self.hidden = 500

    def save(self, filename):
        json.dump({'char_emb_size': self.char_emb_size, 'rnn_layers': self.rnn_layers, 'rnn_size': self.rnn_size,
                   'hidden': self.hidden},
                  open(filename, 'w'))

    def load(self, file):
        if isinstance(file, str):
            stream = open(file, 'r')
        else:
            stream = file
        obj = json.load(stream)
        self.char_emb_size = obj['char_emb_size']
        self.rnn_size = obj['rnn_size']
        self.rnn_layers = obj['rnn_layers']
        self.hidden = obj['hidden']


class AwDoC(nn.Module):
    def __init__(self, config, encodings):
        super(AwDoC, self).__init__()
        self._config = config
        self._encodings = encodings
        self._char_emb = nn.Embedding(len(encodings._char2int), config.char_emb_size)

        self._rnn = nn.LSTM(config.char_emb_size, config.rnn_size, config.rnn_layers, batch_first=True)
        self._hidden = nn.Sequential(nn.Linear(config.rnn_size, config.hidden), nn.Tanh(), nn.Dropout(0.5))
        self._softmax_type = nn.Linear(config.hidden, len(encodings._type2int))
        self._softmax_subtype = nn.Linear(config.hidden, len(encodings._subtype2int))

    def _make_input(self, domain_list):
        # we pad domain names and feed them in reversed character order to the LSTM
        max_seq_len = max([len(domain) for domain in domain_list])

        x = np.zeros((len(domain_list), max_seq_len))
        for iBatch in range(x.shape[0]):
            domain = domain_list[iBatch]
            n = len(domain)
            ofs_x = max_seq_len - n
            for iSeq in range(x.shape[1]):
                if iSeq < n:
                    char = domain[-iSeq - 1].lower()
                    if char in self._encodings._char2int:
                        iChar = self._encodings._char2int[char]
                    else:
                        iChar = self._encodings._char2int['<UNK>']
                    x[iBatch, iSeq + ofs_x] = iChar
        return x

    def forward(self, domain_list):

        x = torch.tensor(self._make_input(domain_list), dtype=torch.long, device=self._get_device())
        hidden = self._char_emb(x)
        hidden = torch.dropout(hidden, 0.5, self.training)
        output, _ = self._rnn(hidden)
        output = output[:, -1, :]

        hidden = self._hidden(output)

        return self._softmax_type(hidden), self._softmax_subtype(hidden)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

    def _get_device(self):
        if self._char_emb.weight.device.type == 'cpu':
            return 'cpu'
        return '{0}:{1}'.format(self._char_emb.weight.device.type, str(self._char_emb.weight.device.index))


def _load_dataset(filename):
    lines = open(filename, encoding='utf-8').readlines()
    dataset = []
    for line in lines:
        line = line.strip()
        if line != '':
            parts = line.split('\t')
            if len(parts) == 3:
                dataset.append(parts)
    return dataset


def _eval(model, dataset, encodings):
    model.eval()
    test_x, test_y = _make_batches(dataset, batch_size=128)
    total_t = 0
    total_st = 0
    ok_t = 0
    ok_st = 0
    with torch.no_grad():
        pgb = tqdm.tqdm(zip(test_x, test_y), total=len(test_x), ncols=80, desc='\t\t\t\t')
        for x, y in pgb:
            y_pred_t, y_pred_st = model(x)
            y_tar_t, y_tar_st = _get_targets(y, encodings)
            y_pred_t = torch.argmax(y_pred_t, dim=1).detach().cpu().numpy()
            y_pred_st = torch.argmax(y_pred_st, dim=1).detach().cpu().numpy()
            for y_t_t, y_t_st, y_p_t, y_p_st in zip(y_tar_t, y_tar_st, y_pred_t, y_pred_st):
                total_t += 1
                if y_t_st != 0:
                    total_st += 1
                    if y_t_st == y_p_st:
                        ok_st += 1
                if y_t_t == y_p_t:
                    ok_t += 1

    return ok_t / total_t, ok_st / total_st


def _make_batches(dataset, batch_size=32):
    batches_x = []
    batches_y = []

    batch_x = []
    batch_y = []

    for entry in dataset:
        domain = entry[0]
        t = entry[1]
        st = entry[2]
        batch_x.append(domain)
        batch_y.append((t, st))
        if len(batch_x) == batch_size:
            batches_x.append(batch_x)
            batches_y.append(batch_y)
            batch_x = []
            batch_y = []

    if len(batch_x) != 0:
        batches_x.append(batch_x)
        batches_y.append(batch_y)

    return batches_x, batches_y


def _get_targets(y, encodings):
    y_t = np.zeros((len(y)))
    y_st = np.zeros((len(y)))
    for i in range(len(y)):
        y_t[i] = encodings._type2int[y[i][0]]
        y_st[i] = encodings._subtype2int[y[i][1]]

    return y_t, y_st


def _drop_tld(domain_list, p):
    new_list = []
    for domain in domain_list:
        parts = domain.split('.')
        dp = random.random()
        if dp < p:
            if dp < p / 2:
                parts[-1] = '  '
            else:
                parts[-1] = '   '
        dom = '.'.join(parts)
        new_list.append(dom)
    return new_list


def _start_train(params):
    trainset = _load_dataset(params.train_file)
    devset = _load_dataset(params.dev_file)
    if params.resume:
        encodings = Encodings('{0}.encodings'.format(params.output_base))
    else:
        encodings = Encodings()
        encodings.update_encodings(trainset)
    print('chars={0}, types={1}, subtypes={2}'.format(len(encodings._char2int), len(encodings._type2int),
                                                      len(encodings._subtype2int)))

    config = AwDoCConfig()
    if params.resume:
        config.load('{0}.conf'.format(params.output_base))
    model = AwDoC(config, encodings)
    model.to(params.device)
    if params.resume:
        model.load('{0}.last'.format(params.output_base))
    optimizer = torch.optim.Adam(model.parameters())
    criterion_t = torch.nn.CrossEntropyLoss()
    criterion_st = torch.nn.CrossEntropyLoss(ignore_index=0)  # we ignore unknown types

    patience_left = params.patience
    best_type, best_subtype = _eval(model, devset, encodings)
    encodings.save('{0}.encodings'.format(params.output_base))
    config.save('{0}.conf'.format(params.output_base))
    model.save('{0}.last'.format(params.output_base))
    print("Deveset evaluation type_acc={0} subtype_acc={1}".format(best_type, best_subtype))
    epoch = 0
    eval_at = 5000
    while patience_left > 0:
        epoch += 1
        random.shuffle(trainset)
        train_x, train_y = _make_batches(trainset, batch_size=params.batch_size)
        sys.stdout.write('Starting epoch {0}\n'.format(epoch))

        pgb = tqdm.tqdm(zip(train_x, train_y), total=len(train_x), ncols=80, desc='\tloss=N/A')
        model.train()
        total_loss = 0
        cnt = 0
        for x, y in pgb:
            cnt += 1
            if cnt % eval_at == 0:
                patience_left -= 1
                sys.stderr.flush()
                sys.stderr.flush()
                sys.stderr.write('\n\tEvaluating...')
                sys.stderr.flush()
                acc_t, acc_st = _eval(model, devset, encodings)
                sys.stderr.write(' type_acc={0}, subtype_acc={1}\n'.format(acc_t, acc_st))
                sys.stderr.flush()
                filename = '{0}.last'.format(params.output_base)
                sys.stderr.write('\t\tStoring {0}\n'.format(filename))
                sys.stderr.flush()
                model.save(filename)
                if acc_t > best_type:
                    patience_left = params.patience
                    best_type = acc_t
                    filename = '{0}.bestType'.format(params.output_base)
                    sys.stderr.write('\t\tStoring {0}\n'.format(filename))
                    sys.stderr.flush()
                    model.save(filename)
                if acc_st > best_subtype:
                    patience_left = params.patience
                    best_subtype = acc_st
                    filename = '{0}.bestSubtype'.format(params.output_base)
                    sys.stderr.write('\t\tStoring {0}\n'.format(filename))
                    sys.stderr.flush()
                    model.save(filename)
                sys.stderr.write('\n')
                sys.stderr.flush()
                model.train()
            if patience_left <= 0:
                print("Stopping with maximum patience reached")
                sys.exit(0)

            x = _drop_tld(x, 0.5)
            y_pred_t, y_pred_st = model(x)

            y_tar_t, y_tar_st = _get_targets(y, encodings)
            y_tar_t = torch.tensor(y_tar_t, dtype=torch.long, device=params.device)
            y_tar_st = torch.tensor(y_tar_st, dtype=torch.long, device=params.device)

            loss = criterion_t(y_pred_t, y_tar_t) + \
                   criterion_st(y_pred_st, y_tar_st)

            optimizer.zero_grad()
            total_loss += loss.item()
            pgb.set_description('\tloss={0:.4f}'.format(total_loss / cnt))
            loss.backward()
            optimizer.step()

        sys.stdout.write('AVG train loss={0}\n'.format(total_loss / len(train_x)))


def _start_interactive(params):
    encodings = Encodings('{0}.encodings'.format(params.output_base))
    config = AwDoCConfig()
    config.load('{0}.conf'.format(params.output_base))
    model = AwDoC(config, encodings)
    model.load('{0}.bestType'.format(params.output_base))
    model.to(params.device)
    model.eval()
    sys.stdout.write('>>> ')
    sys.stdout.flush()
    domain = input()
    while domain != '/exit':
        p_t, p_st = model([domain])
        print(p_t)
        print(p_st)
        p_d_t = torch.argmax(p_t, dim=1).detach().cpu().item()
        p_d_st = torch.argmax(p_st, dim=1).detach().cpu().item()
        print("Results for '{0}'".format(domain))
        print(encodings._type_list[p_d_t])

        print(encodings._subtype_list[p_d_st])

        print("")
        sys.stdout.write('>>> ')
        sys.stdout.flush()
        domain = input()


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--interactive', action='store_true', dest='interactive')
    parser.add_option('--train', action='store_true', dest='train')
    parser.add_option('--resume', action='store_true', dest='resume')
    parser.add_option('--train-file', action='store', dest='train_file')
    parser.add_option('--dev-file', action='store', dest='dev_file')
    parser.add_option('--store', action='store', dest='output_base')
    parser.add_option('--patience', action='store', dest='patience', type='int', default=20, help='(default=20)')
    parser.add_option('--batch-size', action='store', dest='batch_size', default=32, type='int', help='(default=32)')
    parser.add_option('--device', action='store', dest='device', default='cpu')

    (params, _) = parser.parse_args(sys.argv)

    if params.train:
        _start_train(params)
    elif params.interactive:
        _start_interactive(params)
    else:
        parser.print_help()
