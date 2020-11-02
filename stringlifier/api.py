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

from nptyping import NDArray, Int64
from stringlifier.modules.stringc import AwDoC, AwDoCConfig, Encodings
from stringlifier.modules.stringc2 import CTagger, CTaggerConfig
from stringlifier.modules.stringc2 import Encodings as CEncodings
import torch
from typing import List, Optional, Tuple, Union
import pkg_resources


# def _tokenize(string):
#     tokens = []
#     ignore = []
#     current_token = ''
#     for char in string:
#         if char.isalnum():
#             current_token += char
#         else:
#             if current_token != '':
#                 if current_token.isalnum():
#                     ignore.append(False)
#                 else:
#                     ignore.append(True)
#                 tokens.append(current_token)
#             current_token = ''
#
#             tokens.append(char)
#             ignore.append(True)
#
#     tokens.append(current_token)
#     if current_token.isalnum():
#         ignore.append(False)
#     else:
#         ignore.append(True)
#     return tokens, ignore
#
#
# class Stringlifier:
#     def __init__(self, model_base=None):
#         encodings = Encodings()
#         if model_base is None:
#             enc_file = pkg_resources.resource_filename(__name__, 'data/string-c.encodings')
#             conf_file = pkg_resources.resource_filename(__name__, 'data/string-c.conf')
#             model_file = pkg_resources.resource_filename(__name__, 'data/string-c.bestType')
#         else:
#             enc_file = '{0}.encodings'.format(model_base)
#             conf_file = '{0}.conf'.format(model_base)
#             model_file = '{0}.bestType'.format(model_base)
#         encodings.load(enc_file)
#         config = AwDoCConfig()
#         config.load(conf_file)
#         self.classifier = AwDoC(config, encodings)
#         self.classifier.load(model_file)
#         self.classifier.eval()
#         self.encodings = encodings
#
#     def __call__(self, string_or_list, return_tokens=False):
#         if isinstance(string_or_list, str):
#             tokens, ignore_list = _tokenize(string_or_list)
#         else:
#             tokens = string_or_list
#             ignore_list = [False for _ in range(len(tokens))]
#         with torch.no_grad():
#             p_ts, _ = self.classifier(tokens)
#
#         p_ts = torch.argmax(p_ts, dim=-1).detach().cpu().numpy()
#         token_output = []
#         for ignore, token, p_t in zip(ignore_list, tokens, p_ts):
#             if not ignore:
#                 string_type = self.encodings._type_list[p_t]
#             else:
#                 string_type = 'SYMBOL'
#
#             token_output.append({'token': token, 'type': string_type})
#
#         output_string = ''
#         for entry in token_output:
#             if entry['type'] == 'HASH':
#                 output_string += '<RANDOM_STRING>'
#             else:
#                 output_string += entry['token']
#         if return_tokens:
#             return output_string, token_output
#         else:
#             return output_string
#

class Stringlifier:
    def __init__(self, model_base: Optional[str] = None):
        encodings = CEncodings()
        if model_base is None:
            enc_file = pkg_resources.resource_filename(__name__, 'data/enhanced-c.encodings')
            conf_file = pkg_resources.resource_filename(__name__, 'data/enhanced-c.conf')
            model_file = pkg_resources.resource_filename(__name__, 'data/enhanced-c.bestType')
        else:
            enc_file = '{0}.encodings'.format(model_base)
            conf_file = '{0}.conf'.format(model_base)
            model_file = '{0}.bestType'.format(model_base)
        encodings.load(enc_file)
        config = CTaggerConfig()
        config.load(conf_file)
        self.classifier = CTagger(config, encodings)
        self.classifier.load(model_file)
        self.classifier.eval()
        self.encodings = encodings
        self._c_index: int = encodings._label2int['C']

    def __call__(self, string_or_list: Union[str, List[str]], return_tokens: bool = False) -> Union[
        Tuple[List[str], List[List[Tuple[str, int, int, str]]]], List[str]]:
        if isinstance(string_or_list, str):
            tokens = [string_or_list]
        else:
            tokens = string_or_list

        with torch.no_grad():
            p_ts = self.classifier(tokens)

        p_ts = torch.argmax(p_ts, dim=-1).detach().cpu().numpy()
        ext_tokens: List[List[Tuple[str, int, int, str]]] = []
        new_strings: List[str] = []

        for iBatch in range(p_ts.shape[0]):
            new_str, toks = self._extract_tokens(tokens[iBatch], p_ts[iBatch])
            new_strings.append(new_str)
            ext_tokens.append(toks)

        if return_tokens:
            return new_strings, ext_tokens
        else:
            return new_strings

    def _extract_tokens_2class(self, string: str, pred: NDArray[Int64]) -> Tuple[str, List[Tuple[str, int, int]]]:
        CUTOFF = 5
        mask = ''
        for p in pred:
            mask += self.encodings._label_list[p]
        start = 0
        tokens: List[Tuple[str, int, int]] = []
        c_tok = ''
        for ii in range(len(string)):
            if mask[ii] == 'C':
                # check if we have a token

                if c_tok != '':
                    stop = ii
                    tokens.append((c_tok, start, stop))
                    c_tok = ''
            else:
                if c_tok == '':
                    start = ii
                c_tok += string[ii]
        if c_tok != '':
            stop = len(string)
            tokens.append((c_tok, start, stop))

        # filter small tokens
        final_toks: List[Tuple[str, int, int]] = []
        for token in tokens:
            if token[2] - token[1] > 5:
                final_toks.append(token)
        # compose new string
        new_str: str = ''
        last_pos = 0
        for token in final_toks:
            if token[1] > last_pos:
                new_str += string[last_pos:token[1]]
            new_str += token[0]
            last_pos = token[2] + 1
        if last_pos < len(string):
            new_str += string[last_pos:]
        return new_str, final_toks

    def _extract_tokens(self, string: str, pred: NDArray[Int64]) -> Tuple[str, List[Tuple[str, int, int, str]]]:
        mask = ''
        numbers = {str(ii): 1 for ii in range(10)}

        for ii in range(len(pred)):
            p = pred[ii]
            cls = self.encodings._label_list[p]
            if cls == 'C' and string[ii] in numbers:
                mask += 'N'
            else:
                mask += cls
        start = 0
        tokens = []
        c_tok = ''
        last_label = mask[0]
        type_: Optional[str] = None
        for ii in range(len(string)):
            # check if the label-type has changed
            if last_label != mask[ii]:
                if c_tok != '':
                    if last_label == 'C':
                        pass
                    elif last_label == 'H':
                        type_ = '<RANDOM_STRING>'
                    elif last_label == 'N':
                        type_ = '<NUMERIC>'
                    elif last_label == 'I':
                        type_ = '<IP_ADDR>'
                    elif last_label == 'U':
                        type_ = '<UUID>'
                    elif last_label == 'J':
                        type_ = '<JWT>'

                    if last_label != 'C' and type_ is not None:
                        tokens.append((c_tok, start, ii, type_))
                    c_tok = ''
                start = ii

            last_label = mask[ii]
            c_tok += string[ii]

        if c_tok != '':
            if last_label == 'C':
                pass
            elif last_label == 'H':
                type_ = '<RANDOM_STRING>'
            elif last_label == 'N':
                type_ = '<NUMERIC>'
            elif last_label == 'I':
                type_ = '<IP_ADDR>'
            elif last_label == 'U':
                type_ = '<UUID>'
            elif last_label == 'J':
                type_ = '<JWT>'
            if last_label != 'C' and type_ is not None:
                tokens.append((c_tok, start, ii, type_))

        # filter small tokens
        final_toks: List[Tuple[str, int, int, str]] = []
        for token in tokens:
            if token[2] - token[1] > 5:
                final_toks.append(token)
        # compose new string
        new_str: str = ''
        last_pos = 0
        for token in final_toks:
            if token[1] > last_pos:
                new_str += string[last_pos:token[1]]
            new_str += token[3]
            last_pos = token[2]
        if last_pos < len(string) - 1:
            new_str += string[last_pos:]
        return new_str, final_toks
