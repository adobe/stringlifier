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

"""
Training utilities for stringlifier models with improved vectorization and pipeline support.
"""

import random
import uuid
import datetime
import jwt
import string
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Load known words once at module level
try:
    with open('corpus/words_alpha.txt') as f:
        KNOWN_WORDS = [line.strip() for line in f]
        random.shuffle(KNOWN_WORDS)
except FileNotFoundError:
    KNOWN_WORDS = []

# Global index for cycling through known words
known_index = 0

class WordGenerator:
    """
    Generator class for creating synthetic words and their corresponding masks.
    
    This class provides methods to generate different types of synthetic words
    (UUIDs, timestamps, random strings, IP addresses, JWT tokens) along with
    their corresponding mask labels for training.
    """
    
    def __init__(self, known_words: List[str]):
        """
        Initialize the word generator with a list of known words.
        
        Args:
            known_words: List of known words to use for generation
        """
        self.known_words = known_words
        self.known_index = 0
        random.shuffle(self.known_words)
    
    def generate_word(self) -> Tuple[str, str]:
        """
        Generate a synthetic word with its corresponding mask.
        
        Returns:
            Tuple containing (generated_word, mask_character)
        """
        generated = None
        ii = random.randint(0, 5)
        mask = 'H'  # Default mask for random strings
        
        if ii == 0:
            # UUID
            generated = str(uuid.uuid4())
            mask = 'U'
        elif ii == 1:
            # UUID hex
            generated = str(uuid.uuid4().hex)
            mask = 'H'
        elif ii == 2:
            # Numeric
            c = random.randint(0, 3)
            if c == 0:
                generated = str(datetime.datetime.now().timestamp())
            elif c == 1:
                generated = str(random.randint(0, 100000000000))
            elif c == 2:
                generated = f"{random.randint(0, 999)}.{random.randint(0, 999)}"
            else:
                generated = f"{random.randint(0, 999)}.{random.randint(0, 9999)}.{random.randint(0, 9999)}"
            mask = 'N'
        elif ii == 3:
            # Random string
            N = random.randint(5, 20)
            chars = string.ascii_uppercase + string.digits + string.ascii_lowercase
            message = ''.join(random.choice(chars) for _ in range(N))
            
            i = random.randint(0, 2)
            if i == 0:
                message = message.lower()
            elif i == 1:
                message = message.upper()
            generated = message
        elif ii == 4:
            # IP address
            toks = [str(random.randint(0, 255)) for _ in range(4)]
            generated = '.'.join(toks)
            mask = 'I'
        elif ii == 5:
            # JWT token
            generated = self._generate_jwt_token()
            mask = 'J'
            
        return str(generated), mask
    
    def _generate_jwt_token(self) -> str:
        """
        Generate a JWT token for training data.
        
        Returns:
            A string representation of a JWT token
        """
        payload = {
            "id": str(random.random()), 
            "client_id": str(random.random()), 
            "user_id": str(random.random()),
            "type": "access_token",
            "expires_in": str(random.randint(10, 3600000)), 
            "scope": "read, write",
            "created_at": str(random.randint(1900000, 9000000))
        }
        encoded_jwt = jwt.encode(payload, 'secret', algorithm='HS256')
        # Handle both string and bytes return types from different jwt versions
        if isinstance(encoded_jwt, bytes):
            return encoded_jwt.decode('utf-8')
        return encoded_jwt
    
    def get_next_known(self) -> str:
        """
        Get the next known word from the list.
        
        Returns:
            A known word from the internal list
        """
        if not self.known_words:
            return "placeholder"
            
        s = self.known_words[self.known_index]
        self.known_index += 1
        if self.known_index >= len(self.known_words):
            self.known_index = 0
            random.shuffle(self.known_words)
        return s
    
    def get_next_generated(self) -> Tuple[str, str]:
        """
        Get the next generated word and its mask.
        
        Returns:
            Tuple of (word, mask)
        """
        return self.generate_word()


class CommandGenerator:
    """
    Generator for creating synthetic commands with their corresponding masks.
    
    This class uses the WordGenerator to create realistic-looking commands
    with proper masking for training sequence labeling models.
    """
    
    def __init__(self, word_generator: WordGenerator):
        """
        Initialize the command generator.
        
        Args:
            word_generator: WordGenerator instance to use for word generation
        """
        self.word_generator = word_generator
        self.delimiters = ' /.,?!~|<>-=_~:;\\+-&*%$#@!'
        self.enclosers = '[]{}``""\'\'()'
    
    def generate_next_command(self) -> Tuple[str, str]:
        """
        Generate a synthetic command with its corresponding mask.
        
        Returns:
            Tuple of (command, mask)
        """
        mask = ''
        cmd = ''
        num_words = random.randint(3, 15)
        use_space = False
        
        for _ in range(num_words):
            use_delimiter = random.random() > 0.5
            use_encloser = random.random() > 0.8
            case_style = random.randint(0, 2)
            use_gen_word = random.random() > 0.7
            del_index = random.randint(0, len(self.delimiters) - 1)
            enc_index = random.randint(0, len(self.enclosers) // 2 - 1) * 2
            
            if use_space:
                mask += 'C'
                cmd += ' '
                
            if use_gen_word:
                wrd, label = self.word_generator.get_next_generated()
                if case_style == 1:
                    wrd = wrd[0].upper() + wrd[1:] if wrd else wrd
                elif case_style == 2:
                    wrd = wrd.upper()
                msk = label * len(wrd)  # Vectorized mask creation
            else:
                wrd = self.word_generator.get_next_known()
                append_number = random.random() > 0.97
                if append_number:
                    wrd = wrd + str(random.randint(0, 99))
                if case_style == 1:
                    wrd = wrd[0].upper() + wrd[1:] if wrd else wrd
                elif case_style == 2:
                    wrd = wrd.upper()
                msk = 'C' * len(wrd)  # Vectorized mask creation
                
            if use_delimiter:
                wrd = self.delimiters[del_index] + wrd
                msk = 'C' + msk
                
            if use_encloser:
                wrd = self.enclosers[enc_index] + wrd + self.enclosers[enc_index + 1]
                msk = 'C' + msk + 'C'
                
            cmd += wrd
            mask += msk
            use_space = random.random() > 0.7
            
        return cmd, mask


class DatasetGenerator(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for generating synthetic datasets.
    
    This class implements the scikit-learn transformer interface to enable
    integration with scikit-learn pipelines for data generation.
    """
    
    def __init__(self, size: int = 1000, known_words: Optional[List[str]] = None):
        """
        Initialize the dataset generator.
        
        Args:
            size: Number of examples to generate
            known_words: List of known words to use (if None, uses module-level KNOWN_WORDS)
        """
        self.size = size
        self.known_words = known_words if known_words is not None else KNOWN_WORDS
        self.word_generator = WordGenerator(self.known_words)
        self.command_generator = CommandGenerator(self.word_generator)
    
    def fit(self, X=None, y=None):
        """
        Fit method (does nothing but required for scikit-learn compatibility).
        
        Returns:
            self
        """
        return self
    
    def transform(self, X=None) -> List[Tuple[str, str]]:
        """
        Generate a synthetic dataset.
        
        Args:
            X: Ignored, exists for scikit-learn compatibility
            
        Returns:
            List of (command, mask) tuples
        """
        return [self.command_generator.generate_next_command() for _ in range(self.size)]
    
    def fit_transform(self, X=None, y=None) -> List[Tuple[str, str]]:
        """
        Fit and transform (generate dataset).
        
        Args:
            X: Ignored, exists for scikit-learn compatibility
            y: Ignored, exists for scikit-learn compatibility
            
        Returns:
            List of (command, mask) tuples
        """
        self.fit(X, y)
        return self.transform(X)


class BatchCreator(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for creating batches from datasets.
    
    This class implements the scikit-learn transformer interface to enable
    integration with scikit-learn pipelines for batch creation.
    """
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize the batch creator.
        
        Args:
            batch_size: Size of batches to create
        """
        self.batch_size = batch_size
    
    def fit(self, X, y=None):
        """
        Fit method (does nothing but required for scikit-learn compatibility).
        
        Returns:
            self
        """
        return self
    
    def transform(self, X: List[Tuple[str, str]]) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Create batches from a dataset.
        
        Args:
            X: List of (command, mask) tuples
            
        Returns:
            Tuple of (batched_commands, batched_masks)
        """
        random.shuffle(X)
        commands, masks = zip(*X)
        
        # Create batches
        batched_commands = [commands[i:i+self.batch_size] for i in range(0, len(commands), self.batch_size)]
        batched_masks = [masks[i:i+self.batch_size] for i in range(0, len(masks), self.batch_size)]
        
        return batched_commands, batched_masks


def create_data_pipeline(batch_size: int = 32, dataset_size: int = 1000) -> Pipeline:
    """
    Create a scikit-learn pipeline for data generation and batch creation.
    
    Args:
        batch_size: Size of batches to create
        dataset_size: Number of examples to generate
        
    Returns:
        Scikit-learn pipeline for data generation and batch creation
    """
    return Pipeline([
        ('generate', DatasetGenerator(size=dataset_size)),
        ('batch', BatchCreator(batch_size=batch_size))
    ])


# For backward compatibility
def generate_next_cmd() -> Tuple[str, str]:
    """
    Legacy function for generating a command and mask (for backward compatibility).
    
    Returns:
        Tuple of (command, mask)
    """
    global known_index, KNOWN_WORDS
    
    word_generator = WordGenerator(KNOWN_WORDS)
    command_generator = CommandGenerator(word_generator)
    return command_generator.generate_next_command()


def _get_next_known() -> str:
    """
    Legacy function for getting the next known word (for backward compatibility).
    
    Returns:
        A known word
    """
    global known_index, KNOWN_WORDS
    
    if not KNOWN_WORDS:
        return "placeholder"
        
    s = KNOWN_WORDS[known_index]
    known_index += 1
    if known_index >= len(KNOWN_WORDS):
        known_index = 0
        random.shuffle(KNOWN_WORDS)
    return s


def _get_next_gen() -> Tuple[str, str]:
    """
    Legacy function for getting the next generated word (for backward compatibility).
    
    Returns:
        Tuple of (word, mask)
    """
    global KNOWN_WORDS
    
    word_generator = WordGenerator(KNOWN_WORDS)
    return word_generator.generate_word()


def _generate_word(known_words: List[str]) -> Tuple[str, str]:
    """
    Legacy function for generating a word (for backward compatibility).
    
    Args:
        known_words: List of known words
        
    Returns:
        Tuple of (word, mask)
    """
    word_generator = WordGenerator(known_words)
    return word_generator.generate_word()


def _generate_JWT_token(known_words: List[str]) -> str:
    """
    Legacy function for generating a JWT token (for backward compatibility).
    
    Args:
        known_words: List of known words (unused)
        
    Returns:
        JWT token string
    """
    word_generator = WordGenerator([])
    return word_generator._generate_jwt_token()
