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


def _generate_word(known_words):
    import uuid
    import datetime
    import base64
    generated = None
    ii = random.randint(0, 5)
    mask = 'H'
    if ii == 0:
        generated = str(uuid.uuid4())
        mask = 'U'
    elif ii == 1:
        generated = str(uuid.uuid4().hex)
        mask = 'H'
    elif ii == 2:
        c = random.randint(0, 3)
        if c == 0:
            generated = str(datetime.datetime.now().timestamp())
        elif c == 1:
            generated = str(random.randint(0, 100000000000))
        elif c == 2:
            generated = str(random.randint(0, 999)) + '.' + str(random.randint(0, 999))
        else:
            generated = str(random.randint(0, 999)) + '.' + str(random.randint(0, 9999)) + '.' + str(
                random.randint(0, 9999))
        mask = 'N'
    elif ii == 3:
        import string
        N = random.randint(5, 20)
        message = [random.choice(string.ascii_uppercase +
                                 string.digits +
                                 string.ascii_lowercase) for _ in range(N)]
        message = ''.join(message)
        i = random.randint(0, 2)
        if i == 0:
            message = message.lower()
        elif i == 1:
            message = message.upper()
        generated = message
    elif ii == 4:
        toks = []
        for _ in range(4):
            toks.append(str(random.randint(0, 255)))
        generated = '.'.join(toks)
        mask = 'I'
    elif ii == 5:
        generated = _generate_JWT_token(known_words)
        mask = 'J'
    return str(generated), mask[0]


lines = open('corpus/words_alpha.txt').readlines()
for line in lines:
    known_words.append(line.strip())


def _generate_JWT_token(known_words):
    import jwt

    payload = {"id": str(random.random()), "client_id": str(random.random()), "user_id": str(random.random()),
               "type": "access_token",
               "expires_in": str(random.randint(10, 3600000)), "scope": "read, write",
               "created_at": str(random.randint(1900000, 9000000))}
    encoded_jwt = jwt.encode(payload, 'secret', algorithm='HS256')

    return str(encoded_jwt)[2:-1]


# generated_words = generate_words(len(known_words), known_words)

known_index = 0

import random

random.shuffle(known_words)


def _get_next_known():
    global known_index
    s = known_words[known_index]
    known_index += 1
    if known_index == len(known_words):
        known_index = 0
        random.shuffle(known_words)
    return s


def _get_next_gen():
    global known_words
    s, m = _generate_word(known_words)
    return s, m


import random


def generate_next_cmd():
    delimiters = ' /.,?!~|<>-=_~:;\\+-&*%$#@!'
    enclosers = '[]{}``""\'\'()'
    mask = ''
    cmd = ''
    num_words = random.randint(3, 15)
    use_space = False
    use_delimiter = False
    use_encloser = False
    append_number = False
    for ii in range(num_words):

        use_delimiter = random.random() > 0.5
        use_encloser = random.random() > 0.8
        case_style = random.randint(0, 2)
        use_gen_word = random.random() > 0.7

        del_index = random.randint(0, len(delimiters) - 1)
        enc_index = random.randint(0, len(enclosers) // 2 - 1) * 2
        if use_space:
            mask += 'C'
            cmd += ' '
        if use_gen_word:
            wrd, label = _get_next_gen()
            if case_style == 1:
                wrd = wrd[0].upper() + wrd[1:]
            elif case_style == 2:
                wrd = wrd.upper()
            msk = ''
            for _ in range(len(wrd)):
                msk += label
        else:
            wrd = _get_next_known()
            append_number = random.random() > 0.97
            if append_number:
                wrd = wrd + str(random.randint(0, 99))
            if case_style == 1:
                wrd = wrd[0].upper() + wrd[1:]
            elif case_style == 2:
                wrd = wrd.upper()
            msk = ''
            for _ in range(len(wrd)):
                msk += 'C'

        if use_delimiter:
            wrd = delimiters[del_index] + wrd
            msk = 'C' + msk
        if use_encloser:
            wrd = enclosers[enc_index] + wrd + enclosers[enc_index + 1]
            msk = 'C' + msk + 'C'

        cmd += wrd
        mask += msk
        use_space = random.random() > 0.7

    return cmd, mask
