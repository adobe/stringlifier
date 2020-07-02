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

known_words = []


def generate_words(count, known_words):
    import uuid
    import datetime
    import base64
    generated = []
    for ii in range(count):
        if ii % 4 == 0:
            generated.append(str(uuid.uuid4()))
        elif ii % 4 == 1:
            generated.append(str(uuid.uuid4().hex))
        elif ii % 4 == 2:
            generated.append(str(datetime.datetime.now().timestamp()))
        elif ii % 4 == 3:
            message = known_words[ii]
            message_bytes = message.encode('ascii')
            base64_bytes = base64.b64encode(message_bytes)
            base64_message = base64_bytes.decode('ascii')
            generated.append(base64_message)
    return generated


lines = open('corpus/words_alpha.txt').readlines()
for line in lines:
    known_words.append(line.strip())

generated_words = generate_words(len(known_words), known_words)

f = open('corpus/generated', 'w')
for ii in range(len(known_words)):
    f.write(known_words[ii] + '\tSTRING\n')
    f.write(generated_words[ii] + '\tHASH\n')
f.close()
