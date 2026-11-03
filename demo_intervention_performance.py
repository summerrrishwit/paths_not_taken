from demo_load_datasets_model import *
import torch.nn.functional as F
import numpy as np
import circuitsvis as cv

import torch
from tqdm import tqdm

from transformer_lens import HookedEncoder, ActivationCache,HookedTransformer
from transformer_lens import patching
import transformer_lens.utils as utils

from jaxtyping import Float
from typing import Callable
from functools import partial
import itertools
import os
import numpy as np
import json
import pandas as pd
import csv


import os
import random
from functools import partial
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# -----------
# 1) HELPER TO SPLIT TRAIN/TEST BY KNOWN_ID
# -----------


from demo_load_datasets_model import *
import torch.nn.functional as F
import numpy as np
import circuitsvis as cv

import torch
from tqdm import tqdm

from transformer_lens import HookedEncoder, ActivationCache,HookedTransformer
from transformer_lens import patching
import transformer_lens.utils as utils

from jaxtyping import Float
from typing import Callable
from functools import partial
import itertools
import os
import numpy as np
import json
import pandas as pd
import csv


import os
import random
from functools import partial
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# -----------
# 1) HELPER TO SPLIT TRAIN/TEST BY KNOWN_ID
# -----------
from collections import defaultdict

def get_train_test_known_ids(dataset_list, languages, task_for_splitting="fact_recall", test_size=0.5, random_state=42):
    """
    Splits known_ids into train/test such that there is no overlap in `entry['attribute']`
    between the sets for each language and dataset.
    """
    train_ids_dict = defaultdict(dict)
    test_ids_dict = defaultdict(dict)
    for dataset_name in dataset_list:
        for lang in languages:
            data = new_load_ROME_data_for_task_dataset_language(
                task_for_splitting, dataset_name, lang, FROM_LAN="english")
            attribute_to_ids = defaultdict(list)
            for entry in data:
                attr = entry['attribute'].strip().lower()
                attribute_to_ids[attr].append(entry['known_id'])
            attributes = list(attribute_to_ids.keys())
            if len(attributes) < 2:
                print(f"⚠️ Not enough distinct attributes in {dataset_name} ({lang}) to split.")
                train_ids_dict[dataset_name][lang] = set()
                test_ids_dict[dataset_name][lang] = set()
                continue
            train_attrs, test_attrs = train_test_split(
                attributes, test_size=test_size, random_state=random_state)
            train_ids = set()
            for attr in train_attrs:
                train_ids.update(attribute_to_ids[attr])
            test_ids = set()
            for attr in test_attrs:
                test_ids.update(attribute_to_ids[attr])
            train_ids_dict[dataset_name][lang] = train_ids
            test_ids_dict[dataset_name][lang] = test_ids
    return train_ids_dict, test_ids_dict

def train_test_split_across_datasets(dataset_list, languages, task_for_splitting="fact_recall", test_size=0.5, RANDOM_SEED=42):
    """
    Splits datasets into train and test sets. All entries in a dataset go to either train or test.
    """
    train_ids_dict = defaultdict(dict)
    test_ids_dict = defaultdict(dict)

    # Split dataset list
    train_datasets, test_datasets = train_test_split(
        dataset_list, test_size=test_size, random_state=RANDOM_SEED
    )

    for dataset_name in dataset_list:
        for lang in languages:
            data = new_load_ROME_data_for_task_dataset_language(
                task_for_splitting, dataset_name, lang, FROM_LAN="english"
            )
            all_ids = set(entry["known_id"] for entry in data)

            if dataset_name in train_datasets:
                train_ids_dict[dataset_name][lang] = all_ids
                test_ids_dict[dataset_name][lang] = set()
            else:
                train_ids_dict[dataset_name][lang] = set()
                test_ids_dict[dataset_name][lang] = all_ids

    return train_ids_dict, test_ids_dict

def get_train_val_test_within_dataset_split(
    dataset_list,
    languages,
    task_for_splitting="fact_recall",
    split_sizes=(0.5, 0.2, 0.3),  # train, val, test
    random_state=42,
):
    from sklearn.model_selection import train_test_split
    from collections import defaultdict

    train_frac, val_frac, test_frac = split_sizes
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Splits must sum to 1.0"

    train_ids_dict = defaultdict(dict)
    val_ids_dict = defaultdict(dict)
    test_ids_dict = defaultdict(dict)

    for dataset_name in dataset_list:
        for lang in languages:
            data = new_load_ROME_data_for_task_dataset_language(
                task_for_splitting, dataset_name, lang, FROM_LAN="english")

            attribute_to_ids = defaultdict(list)
            for entry in data:
                attr = entry['attribute'].strip().lower()
                attribute_to_ids[attr].append(entry['known_id'])

            attributes = list(attribute_to_ids.keys())
            if len(attributes) < 3:
                print(f"⚠️ Not enough attributes to split: {dataset_name} ({lang})")
                train_ids_dict[dataset_name][lang] = set()
                val_ids_dict[dataset_name][lang] = set()
                test_ids_dict[dataset_name][lang] = set()
                continue

            # Step 1: train_val (70%) vs. test (30%)
            train_val_attrs, test_attrs = train_test_split(
                attributes, test_size=test_frac, random_state=random_state)

            # Step 2: train vs. val within train_val
            val_within_train_val = val_frac / (train_frac + val_frac)
            train_attrs, val_attrs = train_test_split(
                train_val_attrs, test_size=val_within_train_val, random_state=random_state)

            train_ids = {id_ for attr in train_attrs for id_ in attribute_to_ids[attr]}
            val_ids = {id_ for attr in val_attrs for id_ in attribute_to_ids[attr]}
            test_ids = {id_ for attr in test_attrs for id_ in attribute_to_ids[attr]}

            train_ids_dict[dataset_name][lang] = train_ids
            val_ids_dict[dataset_name][lang] = val_ids
            test_ids_dict[dataset_name][lang] = test_ids

    return train_ids_dict, val_ids_dict, test_ids_dict


def train_val_test_split_across_datasets_split(
    dataset_list,
    languages,
    task_for_splitting="fact_recall",
    split_sizes=(0.5, 0.2, 0.3),  # train, val, test
    RANDOM_SEED=42,
):
    from sklearn.model_selection import train_test_split
    from collections import defaultdict

    train_frac, val_frac, test_frac = split_sizes
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Splits must sum to 1.0"

    train_ids_dict = defaultdict(dict)
    val_ids_dict = defaultdict(dict)
    test_ids_dict = defaultdict(dict)

    # Step 1: train_val (70%) vs test (30%)
    train_val_datasets, test_datasets = train_test_split(
        dataset_list, test_size=test_frac, random_state=RANDOM_SEED)

    # Step 2: train vs val within train_val
    val_within_train_val = val_frac / (train_frac + val_frac)
    train_datasets, val_datasets = train_test_split(
        train_val_datasets, test_size=val_within_train_val, random_state=RANDOM_SEED)

    for dataset_name in dataset_list:
        for lang in languages:
            data = new_load_ROME_data_for_task_dataset_language(
                task_for_splitting, dataset_name, lang, FROM_LAN="english")
            all_ids = set(entry["known_id"] for entry in data)

            if dataset_name in train_datasets:
                train_ids_dict[dataset_name][lang] = all_ids
                val_ids_dict[dataset_name][lang] = set()
                test_ids_dict[dataset_name][lang] = set()
            elif dataset_name in val_datasets:
                train_ids_dict[dataset_name][lang] = set()
                val_ids_dict[dataset_name][lang] = all_ids
                test_ids_dict[dataset_name][lang] = set()
            elif dataset_name in test_datasets:
                train_ids_dict[dataset_name][lang] = set()
                val_ids_dict[dataset_name][lang] = set()
                test_ids_dict[dataset_name][lang] = all_ids

    return train_ids_dict, val_ids_dict, test_ids_dict


# -----------
# 2) UPDATED MEAN_LAYER_EMBEDDING_VECTOR TO ACCEPT A "known_ids_dict"
#    This ensures we only use the TRAIN portion in the embedding average.
# -----------
def embedding_add_difference_vector(
    value: Float[torch.Tensor, "batch pos d_model"],
    hook,
    pos,
    difference_vector
):
    #print(value[0,pos,:5])
    value[:, pos, :] = value[:, pos, :]+difference_vector.unsqueeze(0)
    #print(value[0,pos,:5])
    return value


def mean_layer_embedding_vector_train_test(
    languages,
    LAYER,
    TASK,
    dataset_list,
    known_ids_dict=None,
    task_vector_path=None,
    TASK_RESID_POST_LAYER=2,
    TASK_SCALE=4,
    ICL_prompt=False,
    ICL_n_shots=None,
):
    """
    Computes the mean resid_pre vector at a specific layer.
    Efficient: avoids storing all embeddings in memory.
    """

    total_vector = None
    count = 0

    for LANGUAGE in languages:
        for DATASET in dataset_list:
            data = new_load_ROME_data_for_task_dataset_language(
                TASK, DATASET, LANGUAGE, FROM_LAN='english', ICL_prompt=ICL_prompt
            )

            if known_ids_dict is not None:
                allowed_ids = known_ids_dict[DATASET][LANGUAGE]
                data = [d for d in data if d["known_id"] in allowed_ids]

            for entry in data:
                prompt = entry["prompt"]
                attribute = entry["attribute"]

                tokens = model.to_tokens(prompt).to(device)

                if task_vector_path is None:
                    _, orig_cache = model.run_with_cache(tokens, return_type="logits")
                else:
                    model.reset_hooks()
                    task_vector = torch.load(task_vector_path).to(device) * TASK_SCALE
                    model.add_hook(
                        lambda name: name == utils.get_act_name('resid_post', TASK_RESID_POST_LAYER),
                        partial(embedding_add_difference_vector, pos=-1, difference_vector=task_vector),
                        level=1
                    )
                    _, orig_cache = model.run_with_cache(tokens, return_type="logits")

                resid_pre = orig_cache[utils.get_act_name('resid_pre', LAYER)][0, -1, :]

                if total_vector is None:
                    total_vector = resid_pre.clone().detach()
                else:
                    total_vector += resid_pre

                count += 1

    if count > 0:
        return total_vector / count
    else:
        print("No embeddings were collected (check your splits!).")
        return None

import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
import re


def get_singular_and_plural(noun: str):
    """
    Naive approach to detect if 'noun' is singular or plural, then return:
        (singular_form, plural_form)

    Disclaimer:
    - This does not handle many irregular nouns (children, mice, sheep, geese, men, etc.).
    - This is purely heuristic-based and may not apply correctly to all English words.
    """

    noun = noun.strip().lower()

    def guess_singular(plural_noun):
        # If ends in "ies", convert to singular by replacing "ies" with "y"
        if plural_noun.endswith("ies") and len(plural_noun) > 3:
            return plural_noun[:-3] + "y"
        # If ends in "es", remove "es"
        elif plural_noun.endswith("es") and len(plural_noun) > 2:
            return plural_noun[:-2]
        # If ends in "s", remove "s"
        elif plural_noun.endswith("s") and len(plural_noun) > 1:
            return plural_noun[:-1]
        # Fallback: treat it as already singular
        return plural_noun

    def guess_plural(singular_noun):
        # If ends in "y" (and at least 2 letters), replace "y" with "ies"
        #   e.g. "city" -> "cities"
        if singular_noun.endswith("y") and len(singular_noun) > 1:
            # Check if second-last character is a vowel, to be extra naive about "play"/"plays" case
            if singular_noun[-2] not in "aeiou":
                return singular_noun[:-1] + "ies"
            else:
                return singular_noun + "s"
        # Else add "s"
        return singular_noun + "s"

    # Decide if it's plural or singular by a naive rule:
    #   If it ends with "s" (including "es", "ies"), assume it's plural;
    #   otherwise, assume it's singular.
    if noun.endswith("s"):
        # Guess the singular
        singular_form = guess_singular(noun)
        # From that singular, generate the "standard" plural again
        plural_form = guess_plural(singular_form)
    else:
        # Guess the plural
        singular_form = noun
        plural_form = guess_plural(singular_form)

    return singular_form, plural_form


def find_index_with_partial_match(predictions, attribute):
    """Finds the index of the first prediction whose singular or plural form matches the attribute as a prefix."""
    attribute = str(attribute).strip().lower()  # Ensure attribute is a string
    #attribute = re.sub(r'[^a-z]', '', attribute)

    for idx, pred in enumerate(predictions):

        pred = str(pred).strip().lower()  # Ensure pred is a string
        #pred = re.sub(r'[^a-z]', '', pred)
        if pred and attribute.startswith(pred):
            return idx  # Found a match
    return -1  # No match found



def find_index_with_partial_match_just_in_not_start(predictions, attribute):
    """Finds the index of the first prediction whose singular or plural form appears anywhere in the attribute."""
    attribute = str(attribute).strip().lower()  # Ensure attribute is a string
    attribute = re.sub(r'[^a-z]', '', attribute)
    for idx, pred in enumerate(predictions):
        pred = str(pred).strip().lower()  # Ensure pred is a string
        pred = re.sub(r'[^a-z]', '', pred)
        if pred and (pred in attribute):
            return idx  # Found a match

def find_index_with_partial_match_english(predictions, attribute):
    """Finds the index of the first prediction whose singular or plural form matches the attribute as a prefix."""
    attribute = str(attribute).strip().lower()  # Ensure attribute is a string
    attribute = re.sub(r'[^a-z]', '', attribute)

    singular_attribute, plural_attribute = get_singular_and_plural(attribute)

    for idx, pred in enumerate(predictions):
        pred = str(pred).strip().lower()  # Ensure pred is a string
        pred = re.sub(r'[^a-z]', '', pred)
        if pred and (singular_attribute.startswith(pred) or plural_attribute.startswith(pred)):

            return idx  # Found a match
    return -1  # No match found

def find_index_with_partial_match_just_in_not_start_english(predictions, attribute):
    """Finds the index of the first prediction whose singular or plural form appears anywhere in the attribute."""
    attribute = str(attribute).strip().lower() # Ensure attribute is a string
    attribute = re.sub(r'[^a-z]', '', attribute) #remove all things other than alphebetical letters

    singular_attribute, plural_attribute = get_singular_and_plural(attribute)
    for idx, pred in enumerate(predictions):
        pred = str(pred).strip().lower()  # Ensure pred is a string
        pred = re.sub(r'[^a-z]', '', pred)
        # if pred and (pred in attribute) or  or (len(pred)>3 and (pred[:-2] in attribute)):
        if pred:
            if (pred in singular_attribute or pred in plural_attribute) or (singular_attribute in pred or plural_attribute in pred) or (len(pred)>3 and ((pred[:-2] in attribute) or (attribute in pred[:-2]))):
                return idx  # Found a match
    return -1  # No match found
def find_subsequence_indices(full_list, sub_list):
    for i in range(len(full_list) - len(sub_list) + 1):
        if full_list[i : i + len(sub_list)] == sub_list:
            return i, i + len(sub_list) - 1
    return None,None  # Return None if the sequence is not found
def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]
#for translation, wan tto compare subject last
def find_token_range(tokenizer, token_array, substring,LANGUAGE):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  """Find the tokens corresponding to the given substring in token_array."""
  tok_start,tok_end = None,None
  if LANGUAGE in ['chinese','japanese','korean']:
      subject_ids = model.to_tokens(substring).to(device)
      subject_ids_list = subject_ids[0].tolist()[FIRST_ID_INDEX_LLAMA:]
      #print(token_array.tolist(), subject_ids_list)
      tok_start, tok_end = find_subsequence_indices(token_array.tolist(), subject_ids_list)
      if tok_start is None or tok_end is None:
          tok_start, tok_end = find_subsequence_indices(token_array.tolist(), [subject_ids_list[-1]])
      if tok_end is not None:
          tok_end+=1
  else:
      toks = decode_tokens(tokenizer, token_array)
      whole_string = "".join(toks)
      #print(toks)
      #print(substring)
      char_loc = whole_string.index(substring)
      loc = 0
      tok_start, tok_end = None, None
      for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
          tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
          tok_end = i + 1
          break
  return (tok_start, tok_end)


def check_agnostic_correct(TASK,DATASET,known_id,cache,DIFF_LAYER,attribute):
    english_data = new_load_ROME_data_for_task_dataset_language(TASK, DATASET, 'english', FROM_LAN='english')
    resid_post = cache[f'blocks.{DIFF_LAYER}.hook_resid_post']
    vocab_logits = resid_post @ model.W_E.T  # Shape: (batch, seq_len, vocab_size)
    token_index = -1 #prediction place
    vocab_last_logits = vocab_logits[0, token_index]  # Shape: (vocab_size,)
    vocab_probs = torch.nn.functional.softmax(vocab_last_logits, dim=-1)
    topk_prob, topk_id = torch.topk(vocab_probs, k=1, dim=-1)
    token_id = topk_id.item()  # Extract the top token
    token_text = model.tokenizer.decode([token_id]).strip().lower()
    english_entry = next((e for e in english_data if e["known_id"] == known_id), None)
    if english_entry:
        english_attribute = english_entry["attribute"].strip().lower()
        topk_first_prediction = token_text.strip().lower()
        if topk_first_prediction:
            if (topk_first_prediction in english_attribute) or (english_attribute in topk_first_prediction) or (topk_first_prediction in attribute) or (attribute in topk_first_prediction) :
                is_correct = True
            else:
                is_correct = False
        else:
            is_correct = False
        return is_correct

def partial_match_just_in_not_start_english(pred, attribute):
    attribute = str(attribute).strip().lower() # Ensure attribute is a string
    attribute = re.sub(r'[^a-z]', '', attribute) #remove all things other than alphebetical letters
    singular_attribute, plural_attribute = get_singular_and_plural(attribute)
    pred = str(pred).strip().lower()  # Ensure pred is a string
    pred = re.sub(r'[^a-z]', '', pred)
    if pred:
        if (pred in attribute or pred in attribute) or (len(pred)>3 and ((pred[:-2] in attribute) or (attribute in pred[:-2]))):
            return True
    return False

def check_agnostic_correct_partial_match(topk_first_prediction,item):#for relation, subject, last relation,
    if topk_first_prediction:
        topk_first_prediction = topk_first_prediction.strip().lower()
        item = item.strip().lower()
        if (topk_first_prediction in item) or (item in topk_first_prediction):
            return True
    return False


def check_english_relation_correct(TASK,DATASET,known_id,cache,EXAMINE_LAYER):
    english_data = new_load_ROME_data_for_task_dataset_language(TASK, DATASET, 'english', FROM_LAN='english')
    resid_post = cache[f'blocks.{EXAMINE_LAYER}.hook_resid_post']
    vocab_logits = resid_post @ model.W_E.T  # Shape: (batch, seq_len, vocab_size)
    token_index = -1 #prediction place
    vocab_last_logits = vocab_logits[0, token_index]  # Shape: (vocab_size,)
    vocab_probs = torch.nn.functional.softmax(vocab_last_logits, dim=-1)
    topk_prob, topk_id = torch.topk(vocab_probs, k=1, dim=-1)
    token_id = topk_id.item()  # Extract the top token
    token_text = model.tokenizer.decode([token_id]).strip().lower()
    english_entry = next((e for e in english_data if e["known_id"] == known_id), None)
    if english_entry:
        english_relation = english_entry["relation"].strip().lower()
        topk_first_prediction = token_text.strip().lower()
        if topk_first_prediction:
            #is_correct = partial_match_just_in_not_start_english(topk_first_prediction, english_relation)
            is_correct = check_agnostic_correct_partial_match(topk_first_prediction, english_relation)
        return is_correct

def is_generation_correct(prompt, full_output, attribute):
    import unicodedata
    full_output = full_output.strip()
    prompt = prompt.strip()
    if full_output.startswith(prompt):
        generated_part = full_output[len(prompt):].strip()
    else:
        print("❌ NOT full_output.startswith(prompt)")
        generated_part = full_output.strip()
    # Normalize both strings for comparison
    norm_gen = unicodedata.normalize("NFKC", generated_part.strip().lower())
    norm_attr = unicodedata.normalize("NFKC", attribute.strip().lower())
    is_correct = norm_gen.startswith(norm_attr)
    return is_correct, generated_part
import torch
import spacy
import re
import difflib
nlp = spacy.load("en_core_web_sm")
import re
import spacy

# Load the SpaCy language model for English or any other language you are using
nlp = spacy.load("en_core_web_sm")

def process_text(text):
    text = text.strip().lower()

    # ✅ Remove most common punctuation marks (English and others)
    text = re.sub(r'[.,;:!?/\-\\\'"()\[\]{}&_*\#@%\^|~$]', '', text)

    # ✅ Remove specific Chinese punctuation marks (e.g., ？, 。, ！, ...)
    # ✅ Remove specific Chinese punctuation marks
    text = re.sub(r'[？。！，…：；“”‘’（）【】《》－]', '', text)


    # ✅ Remove words like "what", "none", etc.
    # Define a list of words to remove
    words_to_remove = ['what', 'none']
    text = ' '.join([word for word in text.split() if word not in words_to_remove])

    # ✅ Keep Chinese, Japanese, Korean, etc., (no changes required in this line)
    doc = nlp(text)

    # Lemmatize and return the cleaned text
    lemmas = [token.lemma_.lower() for token in doc]
    return " ".join(lemmas)



def project_unembedding(cache, DIFF_LAYER, model):
    # === Get model logits ===
    resid_post = cache[f'blocks.{DIFF_LAYER}.hook_resid_post'].to(model.W_E.device)
    vocab_logits = resid_post @ model.W_E.T  # Shape: (batch, seq_len, vocab_size)
    token_index = -1
    vocab_last_logits = vocab_logits[0, token_index]  # Shape: (vocab_size,)
    vocab_probs = torch.nn.functional.softmax(vocab_last_logits, dim=-1)
    topk_prob, topk_id = torch.topk(vocab_probs, k=1, dim=-1)
    token_id = topk_id.item()
    token_text = model.tokenizer.decode([token_id]).strip().lower()
    return token_text


def check_matching(attribute,pred):

    if pred and attribute and len(pred)>0 and len(attribute)>0:
        attribute_lemmatized = process_text(attribute)
        prediction_lemmatized = process_text(pred)
    else:
        return False


    similarity = difflib.SequenceMatcher(None, prediction_lemmatized, attribute_lemmatized).ratio()
    threshold = 0.6
    is_match =  similarity >= threshold
    if is_match == False:
        is_match = pred.strip().lower() in attribute.strip().lower() or attribute.strip().lower() in pred.strip().lower()
    return is_match


def check_correctness_complete_generation_WITH_INTERVENTION_ag_final_correct(data,LANGUAGE,output_csv,task_vector_path=None,TASK_VECTOR_LAYER=None, TASK_VECTOR_SCALE = None,diff_vector_path=None,DIFF_VECTOR_LAYER= None):
    english_data = new_load_ROME_data_for_task_dataset_language(TASK, DATASET, 'english', FROM_LAN='english')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_entries = 0
    correct_predictions = 0
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['known_id','is_relation_prop_correct','is_agnostic_correct','is_correct','attribute', 'predicted_token'])

    for entry in data:
      known_id = entry["known_id"]
      prompt = entry["prompt"]
      attribute = entry["attribute"]
      relation = entry['relation']
      clean_tokens = model.to_tokens(prompt).to(device)
      attribute_tokens = model.to_tokens(attribute).to(device)


      # Tokenize attribute and count number of tokens
      num_attr_tokens = len(tokenizer(attribute, return_tensors='pt')['input_ids'][0])
      MAX_GENERATION_LEN = num_attr_tokens+2

      english_entry = next((e for e in english_data if e["known_id"] == known_id), None)
      english_attribute = english_entry["attribute"]
      english_relation = english_entry["relation"]

      generated = clean_tokens.clone()  # Start with prompt
      device = model.cfg.device if hasattr(model, "cfg") else "cuda"  # Adjust to your setup
      model.reset_hooks()
      if task_vector_path is not None:
          task_vector = torch.load(task_vector_path).to(device) * TASK_VECTOR_SCALE
          model.add_hook(lambda name: name == utils.get_act_name('resid_pre', TASK_VECTOR_LAYER),
                         partial(embedding_add_difference_vector, pos=-1, difference_vector=task_vector),level=1)
      if diff_vector_path is not None:
          difference_vector = torch.load(diff_vector_path, map_location=device)
          model.add_hook(lambda name: name == utils.get_act_name('resid_pre', DIFF_VECTOR_LAYER),
                partial(embedding_add_difference_vector, pos=-1, difference_vector=difference_vector),
                level=1)
      #print("added the vectors")
      #if MAX_GENERATION_LEN>6:
      #  continue
      #print("MAX_GENERATION_LEN",MAX_GENERATION_LEN)
      for i in range(MAX_GENERATION_LEN):  # Generate tokens one by one
        with torch.no_grad():
            logits, cache = model.run_with_cache(generated, prepend_bos=True, device='cpu')  # store cache on CPU

        if i == 0:
            pred = project_unembedding(cache, DIFF_VECTOR_LAYER, model)
            eng_is_agnostic_correct = check_matching(english_attribute, pred)
            lang_is_agnostic_correct = check_matching(attribute, pred)
            is_agnostic_correct = eng_is_agnostic_correct  # or lang_is_agnostic_correct

            pred = project_unembedding(cache, 16, model)
            eng_relation_prop_correct = check_matching(english_relation, pred)
            lang_relation_prop_correct = check_matching(relation, pred)
            is_relation_prop_correct = eng_relation_prop_correct  # or lang_relation_prop_correct

        next_logits = logits[:, -1, :]  # shape: (batch, vocab)
        next_token_id = next_logits.argmax(dim=-1)  # shape: (batch,)
        next_token = model.tokenizer.decode(next_token_id)
        generated = torch.cat([generated, next_token_id.unsqueeze(1)], dim=1)

        # Free memory
        del logits, cache
        torch.cuda.empty_cache()

      total_output = tokenizer.decode(generated[0], skip_special_tokens=True)
      is_correct, generated_part = is_generation_correct(prompt, total_output, attribute)
      #print(generated_part,attribute,is_correct)
      total_entries+=1
      if is_correct:
          correct_predictions+=1
      with open(output_csv, mode='a', newline='') as file:
          writer = csv.writer(file)
          writer.writerow([known_id,is_relation_prop_correct,is_agnostic_correct,is_correct,attribute, generated_part])
    if total_entries!=0:
      percentage_correct = (correct_predictions / total_entries) * 100
      return percentage_correct
    else:
      return None


def save_diff_vector(train_ids_dict, LAYER,dataset_list,output_difference_vector_path,task_vector_path = None,TASK_RESID_POST_LAYER=2,TASK_SCALE=4,languages =  None):
    # ---- A) Split known_ids into train/test for each (DATASET, LANGUAGE)
    print('save_diff_vector(): task_vector_path',task_vector_path)
    translation_mean_embedding = mean_layer_embedding_vector_train_test(
        ['chinese','japanese','korean','french','spanish'],
        LAYER,
        "translation",
        dataset_list,
        known_ids_dict=train_ids_dict,

    )
    fact_mean_embedding = mean_layer_embedding_vector_train_test(
        languages,
        LAYER,
        "fact_recall",
        dataset_list,
        known_ids_dict=train_ids_dict,
        task_vector_path=task_vector_path,#if we pass in a path, then we have task_vector
        TASK_RESID_POST_LAYER=TASK_RESID_POST_LAYER,
        TASK_SCALE = TASK_SCALE,
    )
    similarity = F.cosine_similarity(
        fact_mean_embedding.flatten(),
        translation_mean_embedding.flatten(),
        dim=0)
    print(f"Layer {LAYER} similarity = {similarity.item():.4f}")
    difference_vector = translation_mean_embedding - fact_mean_embedding
    difference_vector_np = difference_vector.detach().cpu().numpy()
    top_10_indices = np.argsort(np.abs(difference_vector_np))[-10:][::-1]
    print(f"Top-10 indices with largest difference for layer {LAYER}:", top_10_indices)

    torch.save(difference_vector, output_difference_vector_path)
    print(f"Difference vector saved to {output_difference_vector_path}")



from tqdm import tqdm

def compute_h_vector(model, prompts, LAYER, device):
    total = None
    count = 0

    for prompt in tqdm(prompts, desc="Computing h-vectors"):
        tokens = model.to_tokens(prompt).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, return_type="logits")
            resid_post = cache[utils.get_act_name('resid_post', LAYER)][0, -1, :]  # shape: (d_model,)

        if total is None:
            total = resid_post.clone().detach()
        else:
            total += resid_post

        count += 1

    mean_vector = total / count
    return mean_vector
def save_task_vector(train_ids_dict, LAYER,dataset_list,output_task_vector_path,languages = None,split_strategy=None):
    ICL_prompt = True
    ICL_n_shots = 5
    total_data = []
    all_prompts= []
    for DATASET in dataset_list:
        for LANGUAGE in languages:
            data = [entry for entry in new_load_ROME_data_for_task_dataset_language(TASK, DATASET, LANGUAGE, ICL_prompt=ICL_prompt,ICL_n_shots = ICL_n_shots)
                if entry["known_id"] in train_ids_dict[DATASET][LANGUAGE]]
            prompts = [entry['prompt'] for entry in data]
            all_prompts.extend(prompts)
            total_data.extend(data)
    print(len(total_data))
    #output_dir = f"knowLLM_model_results/{MODEL}_results/fact_recall/EVALUATION/{split_strategy}/{condition}/seed_{RANDOM_SEED}/task_vectors_experiment"
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    h = compute_h_vector(model, all_prompts, LAYER, device)
    h = h.to(torch.float32)
    torch.save(h, output_task_vector_path)
    print(f"task vector saved to {output_task_vector_path}")
    return h

#python demo_intervention_performance.py --model llama-3b --diff_layer 25 --task_layer 3 --task_scale 2 --split_strategy within_dataset --seed 15 --condition diff_task



import argparse

parser = argparse.ArgumentParser(description="Run intervention experiments")

parser.add_argument('--model', type=str, default='llama-3b',
                    choices=['llama-1b','llama-3b','llama-8b','Qwen2_1.5b','Qwen2-0.5B','Qwen2.5-3B','llama-7b','llama-3b_instruct'])

parser.add_argument('--diff_layer', type=int, default=25)
parser.add_argument('--task_layer', type=int, default=3)
parser.add_argument('--task_scale', type=int, default=2)

parser.add_argument('--split_strategy', type=str, default='within_dataset',
                    choices=['within_dataset', 'across_dataset'])

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--condition', type=str, default='diff_task',
                    choices=['orig', 'diff', 'task', 'diff_task'])

args = parser.parse_args()


MODEL = 'llama-3b'
model,tokenizer = load_hooked_transformer_model(MODEL)

FIRST_ID_INDEX_LLAMA= 1
ICL_prompt = False
device = utils.get_device()
languages =["chinese","japanese","korean","french","spanish","english"]#"chinese","japanese",
dataset_list = [
    "object_color","language_family","country_language","country_currency",
    "book_language","person_university","animal_classification",
    "musician_instruments","musician_country","country_religion"
]
TASK = 'fact_recall'
layers_for_eval = range(28)
split_sizes = (0.4,0.1,0.5)  # train, val, test
split_sizes_str = '_'.join(str(s) for s in split_sizes)
RANDOM_SEED=15
DIFF_LAYER = 25
H_VECTOR_LAYER = 2
H_VECTOR_SCALE = 3   
condition= 'diff_task'
split_strategy = 'within_dataset'
task_vector_path = None
difference_vector_path = None


if 'diff_task' in condition:
    output_dir = f"knowLLM_model_results/{MODEL}_results/fact_recall/EVALUATION_{split_sizes_str}/{split_strategy}/seed_{RANDOM_SEED}/diff_layer_{DIFF_LAYER}_task_{H_VECTOR_LAYER}_{H_VECTOR_SCALE}"
elif 'task' in condition:
    output_dir = f"knowLLM_model_results/{MODEL}_results/fact_recall/EVALUATION_{split_sizes_str}/{split_strategy}/seed_{RANDOM_SEED}/task_{H_VECTOR_LAYER}_{H_VECTOR_SCALE}"
elif 'diff' in condition:
    output_dir = f"knowLLM_model_results/{MODEL}_results/fact_recall/EVALUATION_{split_sizes_str}/{split_strategy}/seed_{RANDOM_SEED}/diff_layer_{DIFF_LAYER}"
elif 'orig' in condition:
    output_dir = f"knowLLM_model_results/{MODEL}_results/fact_recall/EVALUATION_{split_sizes_str}/{split_strategy}/seed_{RANDOM_SEED}/orig"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
######## GET the difference vector ##########
if split_strategy =='within_dataset':
    train_ids_dict, val_ids_dict,test_ids_dict = get_train_val_test_within_dataset_split(
    dataset_list=dataset_list,
    languages=languages,
    task_for_splitting="fact_recall",  # or "translation" if you like
    split_sizes=split_sizes,  
    random_state=RANDOM_SEED)
    #get_train_val_test_known_ids_split
else:
    train_ids_dict, val_ids_dict,test_ids_dict = train_val_test_split_across_datasets_split(dataset_list, languages, task_for_splitting="fact_recall",  # or "translation" if you like
                                                                        split_sizes=split_sizes,  RANDOM_SEED=RANDOM_SEED)
print(train_ids_dict, val_ids_dict,test_ids_dict )
total_train = 0
total_test = 0
for dataset in train_ids_dict:
    for lang in train_ids_dict[dataset]:
        total_train += len(train_ids_dict[dataset][lang])
        total_test += len(test_ids_dict[dataset][lang])
print(f"✅ Total Train known_ids: {total_train}")
print(f"✅ Total Test known_ids: {total_test}")

if 'task' in condition:
    #task_vector_path = "fvnnsights/chapter1_transformer_interp/exercises/h_vector_animal_classification_5shots_layer3.pt"
    task_vector_path = os.path.join(output_dir,f'h_vector_layer_{H_VECTOR_LAYER}.pt')
    if not os.path.exists(task_vector_path):
        h = save_task_vector(train_ids_dict, H_VECTOR_LAYER,dataset_list,task_vector_path,languages = languages,split_strategy = split_strategy)
        
    else:
        print(f"task vector exists: {task_vector_path}, directly EVAL!")


if 'diff' in condition:
    difference_vector_path = os.path.join(output_dir, f"difference_vector_{DIFF_LAYER}.pt")
    if not os.path.exists(os.path.join(difference_vector_path)):
        save_diff_vector(train_ids_dict, DIFF_LAYER,dataset_list,
                        output_difference_vector_path=difference_vector_path,
                        task_vector_path = task_vector_path,
                        TASK_RESID_POST_LAYER=H_VECTOR_LAYER,TASK_SCALE=H_VECTOR_SCALE,
                        languages = languages)#here if task_vector_path is None, not taken into account
    else:
        print(f"✅ {difference_vector_path} already exists, directly EVAL")
for LANGUAGE in languages:
    for DATASET in dataset_list:
        print(TASK, DATASET, LANGUAGE)

        val_data = [
            entry for entry in new_load_ROME_data_for_task_dataset_language(TASK, DATASET, LANGUAGE, ICL_prompt=False)
            if entry["known_id"] in val_ids_dict[DATASET][LANGUAGE]
        ]
        test_data = [
            entry for entry in new_load_ROME_data_for_task_dataset_language(TASK, DATASET, LANGUAGE, ICL_prompt=False)
            if entry["known_id"] in test_ids_dict[DATASET][LANGUAGE]
        ]
        #print(val_data)
        #print(test_data)
        if len(val_data)>0:
            val_output_dir = os.path.join(output_dir,'val')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
            if not os.path.exists(val_output_dir):
                os.makedirs(val_output_dir)
            val_output_csv = os.path.join(val_output_dir, f"{DATASET}_{LANGUAGE}_ag_final_completions.csv")
            print(val_output_csv)
            print("val accuracy")
            accuracy = check_correctness_complete_generation_WITH_INTERVENTION_ag_final_correct(val_data, LANGUAGE, val_output_csv,task_vector_path = task_vector_path,TASK_VECTOR_LAYER = H_VECTOR_LAYER,
                                                                                                TASK_VECTOR_SCALE = H_VECTOR_SCALE,
                                                                                                diff_vector_path = difference_vector_path,DIFF_VECTOR_LAYER=DIFF_LAYER)
            print(accuracy)
        if len(test_data)>0:
            test_output_dir = os.path.join(output_dir,'test')
            if not os.path.exists(test_output_dir):
                os.makedirs(test_output_dir)
        
            test_output_csv = os.path.join(test_output_dir, f"{DATASET}_{LANGUAGE}_ag_final_completions.csv")
        
            print("test accuracy")
            accuracy = check_correctness_complete_generation_WITH_INTERVENTION_ag_final_correct(test_data, LANGUAGE, test_output_csv,task_vector_path = task_vector_path,TASK_VECTOR_LAYER = H_VECTOR_LAYER,
                                                                                                TASK_VECTOR_SCALE = H_VECTOR_SCALE,
                                                                                                diff_vector_path = difference_vector_path,DIFF_VECTOR_LAYER=DIFF_LAYER)
            print(accuracy)
    
