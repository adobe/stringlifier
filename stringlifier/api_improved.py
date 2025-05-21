import numpy as np
import torch
from typing import List, Tuple, Optional, Union, Dict, Any
from numpy.typing import NDArray

class StringlifierAPI:
    """
    API for the Stringlifier model with improved vectorization.
    
    This class provides methods to identify and extract random strings, UUIDs,
    IP addresses, and other non-natural language tokens from text using
    a trained sequence labeling model.
    """
    
    def __init__(self, classifier, encodings):
        """
        Initialize the Stringlifier API.
        
        Args:
            classifier: Trained classifier model
            encodings: Encodings for the model
        """
        self.classifier = classifier
        self.encodings = encodings
    
    def process(self, string_or_list: Union[str, List[str]], cutoff: int = 5, 
                return_tokens: bool = False) -> Union[List[str], Tuple[List[str], List[List[Tuple[str, int, int, str]]]]]:
        """
        Process input string(s) to identify and replace random strings.
        
        Args:
            string_or_list: Input string or list of strings to process
            cutoff: Minimum length of tokens to consider
            return_tokens: Whether to return extracted tokens along with processed strings
            
        Returns:
            If return_tokens is False, returns list of processed strings
            If return_tokens is True, returns tuple of (processed_strings, extracted_tokens)
        """
        # Handle single string input
        if isinstance(string_or_list, str):
            tokens = [string_or_list]
        else:
            tokens = string_or_list
            
        # Handle empty input
        max_len = max([len(s) for s in tokens]) if tokens else 0
        if max_len == 0:
            if return_tokens:
                return [''], []
            else:
                return ['']
        
        # Get model predictions
        with torch.no_grad():
            p_ts = self.classifier(tokens)
        p_ts = torch.argmax(p_ts, dim=-1).detach().cpu().numpy()
        
        # Process each input string
        ext_tokens: List[List[Tuple[str, int, int, str]]] = []
        new_strings: List[str] = []
        
        for iBatch in range(p_ts.shape[0]):
            new_str, toks = self._extract_tokens_vectorized(tokens[iBatch], p_ts[iBatch], cutoff=cutoff)
            new_strings.append(new_str)
            ext_tokens.append(toks)
            
        if return_tokens:
            return new_strings, ext_tokens
        else:
            return new_strings
    
    def _extract_tokens_vectorized(self, string: str, pred: NDArray, cutoff: int = 5) -> Tuple[
        str, List[Tuple[str, int, int, str]]]:
        """
        Extract tokens from a string using vectorized operations.
        
        Args:
            string: Input string to process
            pred: Model predictions for each character
            cutoff: Minimum length of tokens to consider
            
        Returns:
            Tuple of (processed_string, extracted_tokens)
        """
        if len(string) == 0:
            return "", []
            
        # Convert predictions to mask labels
        mask_array = np.array([self.encodings._label_list[p] for p in pred])
        
        # Special handling for numeric characters
        numbers = set('0123456789')
        string_array = np.array(list(string))
        is_number = np.isin(string_array, list(numbers))
        
        # Apply numeric rule: if character is 'C' and is a number, change to 'N'
        mask_array[(mask_array == 'C') & is_number] = 'N'
        
        # Find label transitions
        transitions = np.diff(np.concatenate([[0], (mask_array[:-1] != mask_array[1:]).astype(int), [1]]))
        start_indices = np.where(transitions == 1)[0]
        end_indices = np.where(transitions == -1)[0]
        
        # Extract tokens
        tokens = []
        for i in range(len(start_indices)):
            start = start_indices[i]
            end = end_indices[i]
            label = mask_array[start]
            
            if label != 'C' and (end - start) > cutoff:
                token_text = string[start:end]
                token_type = self._get_token_type(label)
                if token_type:
                    tokens.append((token_text, start, end, token_type))
        
        # Compose new string with replacements
        if not tokens:
            return string, []
            
        # Use numpy for efficient string composition
        segments = []
        last_pos = 0
        
        for token in tokens:
            if token[1] > last_pos:
                segments.append(string[last_pos:token[1]])
            segments.append(token[3])  # Append token type
            last_pos = token[2]
            
        # Add remaining part of string
        if last_pos < len(string):
            segments.append(string[last_pos:])
            
        return ''.join(segments), tokens
    
    def _get_token_type(self, label: str) -> Optional[str]:
        """
        Get token type based on label.
        
        Args:
            label: Label character from model prediction
            
        Returns:
            Token type string or None if label is 'C' (common text)
        """
        if label == 'C':
            return None
        elif label == 'H':
            return '<RANDOM_STRING>'
        elif label == 'N':
            return '<NUMERIC>'
        elif label == 'I':
            return '<IP_ADDR>'
        elif label == 'U':
            return '<UUID>'
        elif label == 'J':
            return '<JWT>'
        return None
    
    # Legacy methods for backward compatibility
    
    def _extract_tokens_2class(self, string: str, pred: NDArray) -> Tuple[str, List[Tuple[str, int, int]]]:
        """
        Legacy method for extracting tokens with 2-class model (vectorized version).
        
        Args:
            string: Input string to process
            pred: Model predictions for each character
            
        Returns:
            Tuple of (processed_string, extracted_tokens)
        """
        CUTOFF = 5
        
        # Convert predictions to mask
        mask_array = np.array([self.encodings._label_list[p] for p in pred])
        
        # Find transitions between C and non-C
        is_c = mask_array == 'C'
        transitions = np.diff(np.concatenate([[False], ~is_c, [False]]))
        start_indices = np.where(transitions == 1)[0]
        end_indices = np.where(transitions == -1)[0]
        
        # Extract tokens
        tokens = []
        for i in range(len(start_indices)):
            start = start_indices[i]
            end = end_indices[i]
            if end - start > CUTOFF:
                tokens.append((string[start:end], start, end))
        
        # Compose new string
        if not tokens:
            return string, []
            
        new_str = ''
        last_pos = 0
        
        for token in tokens:
            if token[1] > last_pos:
                new_str += string[last_pos:token[1]]
            new_str += token[0]
            last_pos = token[2] + 1
            
        if last_pos < len(string):
            new_str += string[last_pos:]
            
        return new_str, tokens
    
    def _extract_tokens(self, string: str, pred: NDArray, cutoff: int = 5) -> Tuple[
        str, List[Tuple[str, int, int, str]]]:
        """
        Legacy method for extracting tokens (redirects to vectorized version).
        
        Args:
            string: Input string to process
            pred: Model predictions for each character
            cutoff: Minimum length of tokens to consider
            
        Returns:
            Tuple of (processed_string, extracted_tokens)
        """
        return self._extract_tokens_vectorized(string, pred, cutoff)
