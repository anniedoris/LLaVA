import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    print("Preprocess: sources initially")
    print(sources)
    
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        
        print("Single source")
        print(source)
        for j, sentence in enumerate(source):
            print("Single sentence")
            print(sentence)
            
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    print("Preprocess: Conversations")
    print(conversations)
    
    # raise ValueError("pause")
    
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    print(f"Sep{sep}")
    for conversation, target in zip(conversations, targets):
        
        print(f"Conversation: {conversation}")
        print(f"Target: {target}")
        print(f"Pad token id: {tokenizer.pad_token_id}")
        total_len = int(target.ne(tokenizer.pad_token_id).sum()) # tokenized length of the entire thing
        print(f"Total length: {total_len}")
        print("TARGET NO PAD EXCLUSION")
        print(target)
        
        rounds = conversation.split(conv.sep2)
        
        print(f"Rounds:{rounds}")
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX # set the first token to IGNORE INDEX
        
        print("Before rounds loop")
        print(target)
        for i, rou in enumerate(rounds):
            print(f"Round #: {i}")
            if rou == "":
                break

            parts = rou.split(sep)
            print("parts")
            print(parts) # splits off into system prompt + question and answer (without answer trigger token)
            if len(parts) != 2:
                break
            parts[0] += sep # add the assistant trigger token back

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) # this should be 1- the total_len
                print(f"Round length full: {round_len}")
                print(rou)
                print("####")
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2 # why minus 2?
                print(f"Round length just instruction: {instruction_len}")
                print(parts[0])
                print("####")
                sep_len = len(tokenizer_image_token(sep, tokenizer))
                print(f"Sep token length: {sep_len}")
                print(sep)
                ans_len = len(tokenizer_image_token(parts[1], tokenizer))
                print(f"Answer len: {ans_len}")
                print(parts[1])
                print("####")
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # This won't trigger for single turn
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            
            print(f"Some masking?{cur_len, cur_len + instruction_len}")
            print(target)

            cur_len += round_len
        
        target[cur_len:] = IGNORE_INDEX
        
        print(f"Final target: {cur_len}")
        print(target)
        
        tokens = [t.item() for t in target if t != -100]

        # Decode back to text
        decoded_text = tokenizer.decode([673])

        print("Decoded text")
        print(decoded_text)

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )
    
######
tokenizer = transformers.AutoTokenizer.from_pretrained(
            "lmsys/vicuna-13b-v1.5",
            model_max_length=4096,
            padding_side="right",
            use_fast=False,
        )
sources = [[{'from': 'human', 'value': '<image>\nLong prompt here.'}, {'from': 'gpt', 'value': 'Answer here.'}]]
preprocess_v1(sources, tokenizer, has_image=True)