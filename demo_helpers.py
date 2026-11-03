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
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
import re

# -----------
# 1) HELPER TO SPLIT TRAIN/TEST BY KNOWN_ID
# -----------


from sklearn.model_selection import train_test_split
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
    split_sizes=(0.5, 0.0, 0.5),  # train, val, test
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
                task_for_splitting, dataset_name, lang, FROM_LAN="english"
            )

            attribute_to_ids = defaultdict(list)
            for entry in data:
                attr = entry["attribute"].strip().lower()
                attribute_to_ids[attr].append(entry["known_id"])

            attributes = list(attribute_to_ids.keys())
            if len(attributes) < 3:
                print(f"⚠️ Not enough attributes to split: {dataset_name} ({lang})")
                train_ids_dict[dataset_name][lang] = set()
                val_ids_dict[dataset_name][lang] = set()
                test_ids_dict[dataset_name][lang] = set()
                continue

            # Step 1: train_val vs. test
            train_val_attrs, test_attrs = train_test_split(
                attributes, test_size=test_frac, random_state=random_state
            )

            # Step 2: train vs. val (if val > 0)
            if val_frac > 0.0:
                val_within_train_val = val_frac / (train_frac + val_frac)
                train_attrs, val_attrs = train_test_split(
                    train_val_attrs, test_size=val_within_train_val, random_state=random_state
                )
            else:
                train_attrs = train_val_attrs
                val_attrs = []

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


def plot_mean_ranking_results(
    MODEL, TASK, LANGUAGE, dataset_list,
    specific_known_id_dictionary,
    MODE='examine_last_tok', rank_output_dir=None,plot_title=''
):
   
    
    language_colors = {
        'french': '#2B9DE4',
        'spanish': '#3BCBD8',
        'chinese': 'red',
        'japanese': 'orange',
        'korean': '#EF476F',
        'english': 'green'
    }

    all_languages = ['english', 'french', 'spanish', 'chinese', 'japanese', 'korean']
    avg_ranks_by_language = {}
    avg_probs_by_language = {}

    for lang in all_languages:
        total_ranks = []
        total_probs = []

        for DATASET in dataset_list:
            valid_known_ids = set(specific_known_id_dictionary.get(DATASET, []))
            if not valid_known_ids:
                continue
            if lang ==LANGUAGE:
                comp_file = os.path.join(rank_output_dir,f"{DATASET}/attribute_ranks/{MODE}/{LANGUAGE}/{LANGUAGE}_rank_prob_results.csv")
            else:
                comp_file = os.path.join(rank_output_dir,f"{DATASET}/attribute_ranks/{MODE}/{LANGUAGE}/comparison_{lang}_rank_prob_results.csv")
           
          

            df = pd.read_csv(comp_file)
            df = df[df["known_id"].astype(int).isin(valid_known_ids)]
            if df.empty:
                continue

            df.loc[df["predicted_attribute_rank"] == -1, "predicted_attribute_rank"] = 1000
            grouped_rank = df.groupby("layer")["predicted_attribute_rank"].mean()
            grouped_prob = df.groupby("layer")["predicted_prob"].mean()

            total_ranks.append(grouped_rank)
            total_probs.append(grouped_prob)

        if total_ranks:
            avg_ranks_by_language[lang] = pd.concat(total_ranks, axis=1).mean(axis=1)
        if total_probs:
            avg_probs_by_language[lang] = pd.concat(total_probs, axis=1).mean(axis=1)

    # === Plot average rank across languages ===
    plt.figure(figsize=(10, 5))
    for lang, avg_rank in avg_ranks_by_language.items():
        color = language_colors.get(lang.lower(), 'gray')
        linestyle = '-' if lang == LANGUAGE else '--'
        plt.plot(
            avg_rank.index, avg_rank.values,
            label=lang, color=color,
            linestyle=linestyle, linewidth=2 if lang == LANGUAGE else 1.5
        )
    os.makedirs(f"knowLLM_model_results/{MODEL}_results/{TASK}/mean_attr_ranks_plot/", exist_ok=True)
    plt.xlabel("Layer")
    plt.ylabel("Average Rank")
    plt.title(f"{plot_title}, {LANGUAGE}: Mean Object Rank")
    plt.legend()
    plt.grid()
    

    plt.savefig(f"knowLLM_model_results/{MODEL}_results/{TASK}/mean_attr_ranks_plot/{plot_title}_mean_rank_with_comparison_{LANGUAGE}.svg")
    plt.savefig(f"knowLLM_model_results/{MODEL}_results/{TASK}/mean_attr_ranks_plot/{plot_title}_mean_rank_with_comparison_{LANGUAGE}.pdf")
    plt.show()


from collections import defaultdict
import csv, os
def print_results_table(all_results, languages):
    print(f"\n📊 Performance Summary:\n")
    print(f"{'Lang':<10} | {'Total':>6} | {'Acc%':>6} | {'Agn%':>6} | {'Rel%':>6} | {'TransAcc%':>10} | {'ExplTransAcc%':>15}")
    print("-" * 75)

    # Accumulators for non-English summary
    sum_total = 0
    sum_correct = 0
    sum_agnostic = 0
    sum_relation_prop = 0
    sum_translation_acc = 0.0
    sum_explicit_translation_acc = 0.0
    count_translation_acc = 0
    count_explicit_translation_acc = 0

    for lang in languages:
        rec = all_results['check'].get(lang, {})
        total = rec.get('total', 0)
        if total == 0:
            continue

        correct = rec.get('correct', 0)
        agnostic = rec.get('agnostic', 0)
        relation_prop = rec.get('relation_prop', 0)
        translation_acc = rec.get('translation_acc', 0.0)
        explicit_translation_acc = rec.get('explicit_translation_acc', 0.0)

        acc   = 100 * correct / total
        agn   = 100 * agnostic / total
        rel   = 100 * relation_prop / total
        trans = 100 * translation_acc
        expl  = 100 * explicit_translation_acc

        print(f"{lang:<10} | {total:6.0f} | {acc:6.2f} | {agn:6.2f} | {rel:6.2f} | {trans:10.2f} | {expl:15.2f}")

        if lang != "english":
            sum_total += total
            sum_correct += correct
            sum_agnostic += agnostic
            sum_relation_prop += relation_prop
            if translation_acc:
                sum_translation_acc += translation_acc
                count_translation_acc += 1
            if explicit_translation_acc:
                sum_explicit_translation_acc += explicit_translation_acc
                count_explicit_translation_acc += 1

    # Summary for non-English
    if sum_total > 0:
        acc_percent = 100 * sum_correct / sum_total
        agn_percent = 100 * sum_agnostic / sum_total
        rel_percent = 100 * sum_relation_prop / sum_total
        trans_percent = 100 * (sum_translation_acc / count_translation_acc) if count_translation_acc else 0
        expl_percent = 100 * (sum_explicit_translation_acc / count_explicit_translation_acc) if count_explicit_translation_acc else 0

        print("-" * 75)
        print(f"{'non-eng':<10} | {sum_total:6.0f} | {acc_percent:6.10f} | {agn_percent:6.2f} | {rel_percent:6.2f} | {trans_percent:10.2f} | {expl_percent:15.2f}")

def aggregate_results_for_certain_dir(directory,langs):
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    CONDITION='check'
    

    per_total     = defaultdict(int)
    per_final     = defaultdict(int)
    per_agnostic  = defaultdict(int)
    per_relation  = defaultdict(int)

    agn_correct_ids   = defaultdict(lambda: defaultdict(set))
    final_correct_ids = defaultdict(lambda: defaultdict(set))

    dataset_list = [
        'country_currency', 'country_language', 'book_language', 'animal_classification',
        'object_color', 'country_religion', 'language_family', 'musician_country',
        'musician_instruments', 'person_university'
    ]

    for LANG in langs:
        for DATASET in dataset_list:
            csv_path = os.path.join(directory, f"{DATASET}_{LANG}_ag_final_completions.csv")

            if not os.path.exists(csv_path):
                print(f"❌ Missing CSV: {csv_path}")
                continue

            with open(csv_path, encoding='utf-8', newline='') as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    per_total[LANG] += 1


                    if row.get('is_correct', '').lower() == 'true':
                        per_final[LANG] += 1
                        final_correct_ids[LANG][DATASET].add(int(row['known_id']))

                    if row.get('is_agnostic_correct', '').lower() in ['true', '1']:
                        per_agnostic[LANG] += 1
                        agn_correct_ids[LANG][DATASET].add(int(row['known_id']))

                    if row.get('is_relation_prop_correct', '').lower() in ['true', '1']:
                        per_relation[LANG] += 1

    # Compute translation accuracy per language
    for lang in per_total:
        numer, denom = 0, 0
        for dataset in dataset_list:
            correct = agn_correct_ids[lang][dataset]
            converted = final_correct_ids[lang][dataset]
            denom += len(correct)
            numer += len(correct.intersection(converted))
        trans_acc = numer / denom if denom else 0.0

        all_results[CONDITION][lang]['total']           += per_total[lang]
        all_results[CONDITION][lang]['correct']         += per_final[lang]
        all_results[CONDITION][lang]['agnostic']        += per_agnostic[lang]
        all_results[CONDITION][lang]['relation_prop']   += per_relation[lang]
        all_results[CONDITION][lang]['translation_acc'] += trans_acc

        # NEW: explicit translation accuracy
        explicit_translation_correct = 0
        explicit_translation_total = 0

        for dataset in dataset_list:
            intersection_ids = agn_correct_ids[lang][dataset]
            if not intersection_ids:
                continue

            trans_csv = (
                f"knowLLM_model_results/llama-3b_results/translation/"
                f"{dataset}/from_english/{lang}_predictions.csv"
            )

            if os.path.exists(trans_csv):
                with open(trans_csv, encoding='utf-8', newline='') as f:
                    rdr = csv.DictReader(f)
                    for row in rdr:
                        row_id = int(row["known_id"])
                        if row_id in intersection_ids:
                            explicit_translation_total += 1
                            if row.get("is_correct", "").lower() == "true":
                                explicit_translation_correct += 1

        explicit_translation_acc = (
            explicit_translation_correct / explicit_translation_total
            if explicit_translation_total > 0 else 0.0
        )
        all_results[CONDITION][lang]['explicit_translation_acc'] += explicit_translation_acc

    return all_results


def plot_bar_plot_fact_translate_similarity_across_layers(
    cos_similarity_by_language,
    cos_similarity_by_language_INTERVENTION,
    DIFF_LAYER=21,
    model=None  # needed for model.cfg.n_layers
):
    EXAMINE_LAYERS = list(range(DIFF_LAYER, model.cfg.n_layers))
    
    orig_vals = []
    intervention_vals = []

    for L in EXAMINE_LAYERS:
        orig_layer_vals = [
            cos_similarity_by_language[lang][L]
            for lang in cos_similarity_by_language
            if L in cos_similarity_by_language[lang]
        ]
        intv_layer_vals = [
            cos_similarity_by_language_INTERVENTION[lang][L]
            for lang in cos_similarity_by_language_INTERVENTION
            if L in cos_similarity_by_language_INTERVENTION[lang]
        ]

        orig_vals.append(np.mean(orig_layer_vals) if orig_layer_vals else np.nan)
        intervention_vals.append(np.mean(intv_layer_vals) if intv_layer_vals else np.nan)

    x = np.arange(len(EXAMINE_LAYERS))
    layer_labels = [str(L) for L in EXAMINE_LAYERS]
    bar_width = 0.35

    plt.figure(figsize=(6, 4))

    # === Add background horizontal dashed lines
    plt.grid(axis='y', linestyle='--', color='gray', alpha=0.5)

    # === Plot bars
    bars_orig = plt.bar(x - bar_width/2, orig_vals, width=bar_width, label='Original', color='#B5E2FA')
    bars_intv = plt.bar(x + bar_width/2, intervention_vals, width=bar_width, label='Intervention', color='#0CA3B1')

    # === Add labels on each bar
    for bar in bars_orig:
        height = bar.get_height()
        #if not np.isnan(height):
        #   plt.text(bar.get_x() + bar.get_width()/2, height + 0.005, f"{height:.2f}", ha='center', va='bottom', fontsize=11)

    for bar in bars_intv:
        height = bar.get_height()
        #if not np.isnan(height):
            #plt.text(bar.get_x() + bar.get_width()/2, height + 0.005, f"{height:.2f}", ha='center', va='bottom', fontsize=11)

    # === Axis formatting
    plt.xticks(ticks=x, labels=layer_labels, fontsize=14)
    plt.xlabel("Layer", fontsize=14)
    plt.ylabel("Cosine Similarity", fontsize=16)
    plt.title("Neuron Activation Similarity", fontsize=16)
    plt.legend(fontsize=13)
    plt.ylim(0.42, 0.78)

    plt.tight_layout()
    plt.savefig(f'knowLLM_PAPER_PLOTS/Neuron_Sim_{DIFF_LAYER}.svg')
    plt.show()

def agnostic_correct_and_incorrect_list(MODEL, TASK, DATASET, LANGUAGE, EXAMINE_LAYER, MODE='examine_last_tok',
                                        WITH_DIFF_VECTOR=None, WITH_TASK_VECTOR=None,
                                        FIRST_CONVERSION_SUBJECT_TOPK=None, input_directory=None,
                                        specific_known_ids_dict=None):

    if input_directory:
        input_dir = os.path.join(input_directory, f"{DATASET}/attribute_ranks/{MODE}/{LANGUAGE}")
    else:
        if WITH_DIFF_VECTOR is None:
            input_dir = f"knowLLM_model_results/{MODEL}_results/fact_recall/{DATASET}/attribute_ranks/{MODE}/{LANGUAGE}"
        else:
            if WITH_TASK_VECTOR is None:
                input_dir = f"knowLLM_model_results/{MODEL}_results/fact_recall/COMBINED/fact_recall_results_DIFF_LAYER_{WITH_DIFF_VECTOR}/{DATASET}/attribute_ranks/{MODE}/{LANGUAGE}"
            else:
                input_dir = f"knowLLM_model_results/{MODEL}_results/fact_recall/COMBINED/fact_recall_results_DIFF_LAYER_{WITH_DIFF_VECTOR}_and_TASK_vector_LAYER_3/{DATASET}/attribute_ranks/{MODE}/{LANGUAGE}"

    if LANGUAGE == 'english':
        results_csv_fp = os.path.join(input_dir, 'english_rank_prob_results.csv')
    else:
        results_csv_fp = os.path.join(input_dir, 'comparison_english_rank_prob_results.csv')

    if not os.path.exists(results_csv_fp):
        raise FileNotFoundError(f"File not found: {results_csv_fp}")

    df = pd.read_csv(results_csv_fp)

    if specific_known_ids_dict is None:
        filtered_df = df[df['layer'] == EXAMINE_LAYER]
    else:
        filtered_df = df[
            (df['layer'] == EXAMINE_LAYER) &
            (df['known_id'].isin(specific_known_ids_dict[DATASET][LANGUAGE]))
        ]

    if MODE in ['examine_last_tok', 'examine_relation']:
        correct_df = filtered_df[filtered_df['predicted_attribute_rank'] == 0]
    elif MODE == 'examine_subject_tok':
        if FIRST_CONVERSION_SUBJECT_TOPK is None:
            correct_df = filtered_df[filtered_df['predicted_attribute_rank'] == 0]
        else:
            correct_df = filtered_df[
                (filtered_df['predicted_attribute_rank'] > 0) &
                (filtered_df['predicted_attribute_rank'] < FIRST_CONVERSION_SUBJECT_TOPK)
            ]
    else:
        correct_df = pd.DataFrame(columns=filtered_df.columns)  # Empty fallback

    # If MODE is examine_last_tok and language is not English, do additional merging
    if MODE == 'examine_last_tok' and LANGUAGE != 'english':
        alt_fp = os.path.join(input_dir, f'{LANGUAGE}_rank_prob_results.csv')
        if not os.path.exists(alt_fp):
            raise FileNotFoundError(f"File not found: {alt_fp}")
        df_lang = pd.read_csv(alt_fp)
        lang_filtered = df_lang[df_lang['layer'] == EXAMINE_LAYER]
        lang_correct = lang_filtered[lang_filtered['predicted_attribute_rank'] == 0]
        correct_df = pd.concat([correct_df, lang_correct])

    list_of_agnostic_correct = correct_df['known_id'].tolist()
    all_known_ids = filtered_df['known_id'].tolist()
    list_of_agnostic_incorrect = [kid for kid in all_known_ids if kid not in list_of_agnostic_correct]

    return list_of_agnostic_correct, list_of_agnostic_incorrect
#Figure 2: Breakdown of Failures (final / agnostic correct and incorrect)
import re
import spacy
import difflib
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
def correct_incorrect_predictions_ids(MODEL,DATASET,LANGUAGE,TASK='fact_recall'):
    file_path = f'knowLLM_model_results/{MODEL}_results/{TASK}/{DATASET}/{LANGUAGE}_predictions.csv'
    if TASK =='translation':
        file_path = f"knowLLM_model_results/{MODEL}_results/{TASK}/{DATASET}/from_english/{LANGUAGE}_predictions.csv"
    true_known_ids = None
    false_known_ids = None
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        true_df = df[df['is_correct'] == True]
        true_known_ids = set(true_df['known_id'].tolist())
        false_df = df[df['is_correct'] == False]
        false_known_ids = set(false_df['known_id'].tolist())
        
    else:
        print(f"{file_path} does not exist")
    return true_known_ids,false_known_ids


def check_agnostic_correct_from_input_directory(
    MODEL, TASK, DATASET, LANGUAGE, EXAMINE_LAYER,
    MODE='examine_last_tok',
    WITH_DIFF_VECTOR=None, WITH_TASK_VECTOR=None,
    FIRST_CONVERSION_SUBJECT_TOPK=None,
    input_directory=None,
    specific_known_ids_dict=None
):
    # Load English ground truth data
    with open(f"knowLLM_DATASETS/fact_recall/{DATASET}/{DATASET}_english.json", "r") as f:
        english_data = json.load(f)

    # Load prediction file
    input_path =os.path.join(input_directory,f"topk_preds/{MODE}/{LANGUAGE}/{LANGUAGE}_top_1000_{EXAMINE_LAYER}.csv")
    
    df = pd.read_csv(input_path)

    ag_correct_known_ids = []
    ag_false_known_ids = []

    for _, row in df.iterrows():
        known_id = row['known_id']
        predictions_str = row['predictions']
        try:
            predictions = eval(predictions_str)
        except:
            continue  # Skip malformed entries

        if not predictions:
            continue

        # Match with English attribute
        english_entry = next((e for e in english_data if e["known_id"] == known_id), None)
        if english_entry is None:
            continue  # Skip if known_id not found

        english_attribute = english_entry['attribute']
        if check_matching(predictions[0], english_attribute):
            ag_correct_known_ids.append(known_id)
        else:
            ag_false_known_ids.append(known_id)

    return ag_correct_known_ids, ag_false_known_ids