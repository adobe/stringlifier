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


class Encodings:
    def __init__(self, filename=None):
        self._char2int = {'<PAD>': 0, '<UNK>': 1}
        self._label2int = {'<PAD>': 0}
        self._label_list = ['<PAD>']
        if filename is not None:
            self.load(filename)

    def save(self, filename):
        json.dump({'char2int': self._char2int, 'label2int': self._label2int},
                  open(filename, 'w'))

    def load(self, file):
        if isinstance(file, str):
            stream = open(file, 'r')
        else:
            stream = file
        obj = json.load(stream)
        self._char2int = obj['char2int']
        self._label2int = obj['label2int']
        self._label_list = [None for _ in range(len(self._label2int))]
        for t in self._label2int:
            self._label_list[self._label2int[t]] = t

    def update_encodings(self, dataset, cutoff=2):
        char2count = {}
        for entry in tqdm.tqdm(dataset):
            text = entry[0]
            label = entry[1]
            for char in text:
                char = char.lower()
                if char in char2count:
                    char2count[char] += 1
                else:
                    char2count[char] = 1
            for ttype in label:
                if ttype not in self._label2int:
                    self._label2int[ttype] = len(self._label2int)
                    self._label_list.append(ttype)

        for char in char2count:
            if char not in self._char2int and char2count[char] > cutoff:
                self._char2int[char] = len(self._char2int)


class CTaggerConfig:
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


class CTagger(nn.Module):
    def __init__(self, config, encodings):
        super(CTagger, self).__init__()
        self._config = config
        self._encodings = encodings
        self._char_emb = nn.Embedding(len(encodings._char2int), config.char_emb_size, padding_idx=0)
        self._case_emb = nn.Embedding(4, 16, padding_idx=0)

        self._rnn = nn.LSTM(config.char_emb_size + 16, config.rnn_size, config.rnn_layers, batch_first=True,
                            bidirectional=True)
        self._hidden = nn.Sequential(nn.Linear(config.rnn_size * 2, config.hidden), nn.Tanh(), nn.Dropout(0.5))
        self._softmax_type = nn.Linear(config.hidden, len(encodings._label2int))

    def _make_input(self, word_list):
        # we pad domain names and feed them in reversed character order to the LSTM
        max_seq_len = max([len(word) for word in word_list])

        x_char = np.zeros((len(word_list), max_seq_len))
        x_case = np.zeros((len(word_list), max_seq_len))
        for iBatch in range(x_char.shape[0]):
            word = word_list[iBatch]
            for index in range(len(word)):
                char = word[index]
                case_idx = 0
                if char.lower() == char.upper():
                    case_idx = 1  # symbol
                elif char.lower() != char:
                    case_idx = 2  # uppercase
                else:
                    case_idx = 3  # lowercase
                char = char.lower()
                if char in self._encodings._char2int:
                    char_idx = self._encodings._char2int[char]
                else:
                    char_idx = 1  # UNK
                x_char[iBatch, index] = char_idx
                x_case[iBatch, index] = case_idx

        return x_char, x_case

    def forward(self, string_list):
        x_char, x_case = self._make_input(string_list)
        x_char = torch.tensor(x_char, dtype=torch.long, device=self._get_device())
        x_case = torch.tensor(x_case, dtype=torch.long, device=self._get_device())
        hidden = torch.cat([self._char_emb(x_char), self._case_emb(x_case)], dim=-1)
        hidden = torch.dropout(hidden, 0.5, self.training)
        output, _ = self._rnn(hidden)

        hidden = self._hidden(output)

        return self._softmax_type(hidden)

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
    for ii in range(len(lines) // 2):
        string = lines[ii * 2][:-1]
        mask = lines[ii * 2 + 1][:-1]
        dataset.append((string, mask))
    return dataset


def _eval(model, dataset, encodings):
    model.eval()
    test_x, test_y = _make_batches(dataset, batch_size=128)
    total_t = 0
    ok_t = 0
    with torch.no_grad():
        pgb = tqdm.tqdm(zip(test_x, test_y), total=len(test_x), ncols=80, desc='\t\t\t\t')
        for x, y in pgb:
            y_pred_t = model(x)
            y_tar_t = _get_targets(y, encodings).reshape(-1)
            y_pred_t = torch.argmax(y_pred_t, dim=-1).detach().cpu().numpy().reshape(-1)
            for y_t_t, y_p_t in zip(y_tar_t, y_pred_t):
                if y_t_t != 0:
                    total_t += 1

                    if y_t_t == y_p_t:
                        ok_t += 1

    return ok_t / total_t


def _make_batches(dataset, batch_size=32):
    batches_x = []
    batches_y = []

    batch_x = []
    batch_y = []

    for entry in dataset:
        domain = entry[0]
        t = entry[1]
        batch_x.append(domain)
        batch_y.append(t)
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
    max_len = max([len(yy) for yy in y])
    y_t = np.zeros((len(y), max_len), dtype=np.long)
    for i in range(len(y)):
        for j in range(max_len):
            if j < len(y[i]):
                y_t[i, j] = encodings._label2int[y[i][j]]

    return y_t


def _generate_dataset(count):
    from training import generate_next_cmd
    dataset = []
    for ii in range(count):
        cmd, mask = generate_next_cmd()
        dataset.append((cmd, mask))
    return dataset


def _start_train(params):
    eval_at = 5000

    if params.resume:
        encodings = Encodings('{0}.encodings'.format(params.output_base))
    else:
        sys.stdout.write('Generating new random data...')
        sys.stdout.flush()
        trainset = _generate_dataset(int(eval_at * 4 * params.batch_size))
        sys.stdout.write('done\n')
        encodings = Encodings()
        encodings.update_encodings(trainset)

    print('chars={0}, types={1}'.format(len(encodings._char2int), len(encodings._label2int)))
    print(encodings._label2int)

    config = CTaggerConfig()
    if params.resume:
        config.load('{0}.conf'.format(params.output_base))
    model = CTagger(config, encodings)
    model.to(params.device)
    if params.resume:
        model.load('{0}.last'.format(params.output_base))
    optimizer = torch.optim.Adam(model.parameters())
    criterion_t = torch.nn.CrossEntropyLoss(ignore_index=0)

    patience_left = params.patience
    best_type = 0  # _eval(model, devset, encodings)
    encodings.save('{0}.encodings'.format(params.output_base))
    config.save('{0}.conf'.format(params.output_base))
    model.save('{0}.last'.format(params.output_base))
    print("Deveset evaluation acc={0}".format(best_type))
    epoch = 0
    eval_at = 5000

    while patience_left > 0:
        sys.stdout.write('Generating new random data...')
        sys.stdout.flush()
        trainset = _generate_dataset(int(eval_at * params.batch_size))
        devset = _generate_dataset(int(eval_at / 10 * params.batch_size))
        sys.stdout.write('done\n')
        sys.stdout.flush()
        sys.stderr.flush()
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
                acc_t = _eval(model, devset, encodings)
                sys.stderr.write(' acc={0}\n'.format(acc_t))
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

                sys.stderr.write('\n')
                sys.stderr.flush()
                model.train()

            if patience_left <= 0:
                print("Stopping with maximum patience reached")
                sys.exit(0)

            y_pred_t = model(x)

            y_tar_t = _get_targets(y, encodings)
            y_tar_t = torch.tensor(y_tar_t, dtype=torch.long, device=params.device)
            y_pred = y_pred_t.view(-1, y_pred_t.shape[-1])
            y_target = y_tar_t.view(-1)
            if y_pred.shape[0] != y_target.shape[0]:
                from ipdb import set_trace
                set_trace()
            loss = criterion_t(y_pred, y_target)

            optimizer.zero_grad()
            total_loss += loss.item()
            pgb.set_description('\tloss={0:.4f}'.format(total_loss / cnt))
            loss.backward()
            optimizer.step()

        sys.stdout.write('AVG train loss={0} \n'.format(total_loss / len(train_x)))


def _start_interactive(params):
    encodings = Encodings('{0}.encodings'.format(params.output_base))
    config = CTaggerConfig()
    config.load('{0}.conf'.format(params.output_base))
    model = CTagger(config, encodings)
    model.load('{0}.bestType'.format(params.output_base))
    model.to(params.device)
    model.eval()
    sys.stdout.write('>>> ')
    sys.stdout.flush()
    string = input()
    while string != '/exit':
        p_t = model([string])
        p_d_t = torch.argmax(p_t, dim=-1).detach().cpu().numpy()
        print("Results for \n{0}".format(string))
        for ii in range(p_d_t.shape[-1]):
            sys.stdout.write(encodings._label_list[p_d_t[0, ii]])
        sys.stdout.write('\n')

        print("")
        sys.stdout.write('>>> ')
        sys.stdout.flush()
        string = input()


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--interactive', action='store_true', dest='interactive')
    parser.add_option('--train', action='store_true', dest='train')
    parser.add_option('--resume', action='store_true', dest='resume')

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
