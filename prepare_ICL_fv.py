import json
import numpy as np
class ICLSequence:
    """
    Class to store a single antonym sequence.

    Uses the default template "Q: {x}\nA: {y}" (with separate pairs split by "\n\n").
    """

    def __init__(self, word_pairs: list[list[str]]):
        self.word_pairs = word_pairs
        self.x, self.y = zip(*word_pairs)

    def __len__(self):
        return len(self.word_pairs)

    def __getitem__(self, idx: int):
        return self.word_pairs[idx]

    def prompt(self):
        """Returns the prompt, which contains all but the second element in the last word pair."""
        p = "\n\n".join([f"Q: {x}\nA: {y}" for x, y in self.word_pairs])
        return p[: -len(self.completion())]

    def completion(self):
        """Returns the second element in the last word pair (with padded space)."""
        return " " + self.y[-1]

    def __str__(self):
        """Prints a readable string representation of the prompt & completion (indep of template)."""
        return f"{', '.join([f'({x}, {y})' for x, y in self[:-1]])}, {self.x[-1]} ->".strip(", ")


class ICLDataset:
    def __init__(
        self,
        word_pairs: list[list[str]],  # each = [known_id, subject, attribute]
        size: int,
        n_prepended: int,
        bidirectional: bool = False,
        seed: int = 0,
        corrupted: bool = False,
        instruction: str = None
    ):
        assert n_prepended + 1 <= len(word_pairs), "Not enough word pairs to create prompt."

        # Extract known_ids and word_pairs from input
        self.known_ids = [pair[0] for pair in word_pairs] 
        self.word_pairs = [pair[1:] for pair in word_pairs] 
        self.word_list = [word for pair in self.word_pairs for word in pair]
        self.size = size
        self.n_prepended = n_prepended
        self.bidirectional = bidirectional
        self.corrupted = corrupted
        self.seed = seed
        self.seqs = []
        self.prompts = []
        self.completions = []
        self.subjects = []
        


        for n in range(size):
            np.random.seed(seed + n)

            # Pick one word pair to be the "test" pair
            test_idx = np.random.choice(len(self.word_pairs), 1, replace=False)
            order = np.random.choice([1, -1], 1)
            if not bidirectional:
                order[:] = 1
            test_pair = [self.word_pairs[test_idx[0]][::order[0]]]
            self.subjects.append(test_pair[0][0])  # first element in the selected test pair

            # Generate context (prepended) word pairs
            if n_prepended > 0:
                np.random.seed(seed + n + 1)
                prepend_idxs = np.random.choice(len(self.word_pairs), n_prepended, replace=False)
                prepend_orders = np.random.choice([1, -1], n_prepended)
                if not bidirectional:
                    prepend_orders[:] = 1

                context_pairs = [self.word_pairs[i][::o] for i, o in zip(prepend_idxs, prepend_orders)]

                if corrupted:
                    for i in range(len(context_pairs)):
                        context_pairs[i][1] = np.random.choice(self.word_list)

                full_sequence = context_pairs + test_pair
            else:
                full_sequence = test_pair

            seq = ICLSequence(full_sequence)
            self.seqs.append(seq)

            # Format prompt
            if instruction is None:
                self.prompts.append(seq.prompt())
            elif '{' in instruction:
                self.prompts.append(instruction.format(test_pair[0][0]))
            else:
                self.prompts.append(instruction + seq.prompt())

            self.completions.append(seq.completion())

    def create_corrupted_dataset(self):
        """Creates a corrupted version of the dataset (with same random seed)."""
        return ICLDataset(
            self.word_pairs,
            self.size,
            self.n_prepended,
            self.bidirectional,
            corrupted=True,
            seed=self.seed
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return self.seqs[idx]

import json

def construct_n_shot_zero_shot_datasets(DATASET, LAN, n_shot=5, instruction='',seed=0):
    file_path = f'knowLLM_DATASETS/fact_recall/{DATASET}/{DATASET}_{LAN}.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Format: [known_id, subject, attribute]
    FACT_RECALL_PAIRS = [[entry["known_id"], entry["subject"], entry["attribute"]] for entry in data]

    dataset = ICLDataset(
        word_pairs=FACT_RECALL_PAIRS,
        size=len(FACT_RECALL_PAIRS),
        n_prepended=n_shot,
        corrupted=False,
        instruction=instruction,
        seed=seed
    )

    zero_shot_dataset = ICLDataset(
        word_pairs=FACT_RECALL_PAIRS,
        size=len(FACT_RECALL_PAIRS),
        n_prepended=0,
        corrupted=False,
        instruction=instruction,
        seed=seed
    )

    return dataset, zero_shot_dataset
