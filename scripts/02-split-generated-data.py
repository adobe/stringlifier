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


string_list = []
hash_list = []

lines = open('corpus/generated').readlines()

for line in lines:
    parts = line.strip().split('\t')
    if parts[1] == 'STRING':
        string_list.append(parts[0])
    else:
        hash_list.append(parts[0])

train_data = [
    ('usr', 'STRING'),
    ('var', 'STRING'),
    ('lib', 'STRING'),
    ('etc', 'STRING'),
    ('tmp', 'STRING'),
    ('dev', 'STRING'),
    ('libexec', 'STRING'),
    ('lib32', 'STRING'),
    ('lib64', 'STRING'),
    ('bin', 'STRING')
]
dev_data = []


def add_data(train, dev, list, label):
    for ii in range(len(list)):
        if ii % 10 == 0:
            dev.append((list[ii], label))
        else:
            train.append((list[ii], label))


add_data(train_data, dev_data, string_list, "STRING")
add_data(train_data, dev_data, hash_list, "HASH")

import random

random.shuffle(train_data)
random.shuffle(dev_data)

f_train = open('corpus/string-train', 'w')
f_dev = open('corpus/string-dev', 'w')

for ii in range(len(train_data)):
    if train_data[ii][1] == 'HASH':
        stype = 'HASH'
    else:
        stype = 'WORD'
    f_train.write(train_data[ii][0] + '\t' + train_data[ii][1] + '\t' + stype + '\n')
for ii in range(len(dev_data)):
    if dev_data[ii][1] == 'HASH':
        stype = 'HASH'
    else:
        stype = 'WORD'
    f_dev.write(dev_data[ii][0] + '\t' + dev_data[ii][1] + '\t' + stype + '\n')

f_train.close()
f_dev.close()
