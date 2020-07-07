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

lines = open('corpus/generated-enhanced').readlines()
f_train = open('corpus/enhanced-train', 'w')
f_dev = open('corpus/enhanced-dev', 'w')

for ii in range(len(lines) // 2):
    word = lines[ii * 2]
    mask = lines[ii * 2 + 1]
    f = f_train
    if ii % 10 == 5:
        f = f_dev
    f.write(word + mask)

f_train.close()
f_dev.close()
