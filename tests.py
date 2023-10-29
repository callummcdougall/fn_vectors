from typing import List, Union, Tuple, Optional, Callable
import numpy as np
import torch as t
from torch import Tensor
from pathlib import Path
import string

N_LAYERS = 28
N_HEADS = 16
D_MODEL = 4096
D_HEAD = D_MODEL // N_HEADS

# root should be the "function_vectors" path
root = Path(__file__).parent



class AntonymSequence:
    '''
    Class to store a single antonym sequence.

    Uses the default template "Q: {x}\nA: {y}" (with separate pairs split by "\n\n").
    '''
    def __init__(self, word_pairs: List[List[str]]):
        self.word_pairs = word_pairs
        self.x, self.y = zip(*word_pairs)

    def __len__(self):
        return len(self.word_pairs)

    def __getitem__(self, idx: int):
        return self.word_pairs[idx]

    def prompt(self):
        '''Returns the prompt, which contains all but the second element in the last word pair.'''
        p = "\n\n".join([f"Q: {x}\nA: {y}" for x, y in self.word_pairs])
        return p[:-len(self.completion())]

    def completion(self):
        '''Returns the second element in the last word pair (with padded space).'''
        return " " + self.y[-1]

    def __str__(self):
        '''Prints a readable string representation of the prompt & completion (indep of template).'''
        return f"{', '.join([f'({x}, {y})' for x, y in self[:-1]])}, {self.x[-1]} ->".strip(", ")

class AntonymDataset:
    '''
    Dataset to create antonym pair prompts, in ICL task format. We use random seeds for consistency
    between the corrupted and clean datasets.

    Inputs:
        word_pairs: list of antonym pairs, e.g. [["old", "young"], ["top", "bottom"], ...]
        corrupted: if True, then the second word in each pair is replaced with a random word
        size = number of prompts to generate
        n_prepended = number of antonym pairs before the single word
    '''

    def __init__(
        self,
        word_pairs: List[List[str]],
        size: int,
        n_prepended: int,
        corrupted: bool = False,
        seed: int = 0,
    ):
        assert n_prepended+1 <= len(word_pairs), "Not enough antonym pairs in dataset to create prompt."
        
        self.seed = seed
        self.size = size
        self.n_prepended = n_prepended
        self.corrupted = corrupted
        self.word_pairs = word_pairs
        self.word_list = [word for word_pair in word_pairs for word in word_pair]

        self.seqs = []
        self.prompts = []
        self.completions = []

        # Generate the dataset (by choosing random antonym pairs, and constructing `AntonymSequence` objects)
        for n in range(size):
            np.random.seed(seed + n)
            random_pairs = np.random.choice(len(self.word_pairs), n_prepended+1, replace=False)
            random_orders = np.random.choice([1, -1], n_prepended+1)
            word_pairs = [self.word_pairs[pair][::order] for pair, order in zip(random_pairs, random_orders)]
            if corrupted:
                for i in range(len(word_pairs) - 1):
                    word_pairs[i][1] = np.random.choice(self.word_list)
            seq = AntonymSequence(word_pairs)

            self.seqs.append(seq)
            self.prompts.append(seq.prompt())
            self.completions.append(seq.completion())

    def create_corrupted_dataset(self):
        '''Creates a corrupted version of the dataset (with same random seed).'''
        return AntonymDataset(self.word_pairs, self.size, self.n_prepended, True, self.seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return self.seqs[idx]



def test_calculate_h(calculate_h: Callable[[AntonymDataset, int], Tuple[List[str], Tensor]]) -> None:

    # This same code was used to initially calculate the fn vector (everything is deterministic)
    word_pairs = list(zip(string.ascii_lowercase, string.ascii_uppercase))
    dataset = AntonymDataset(word_pairs, size=5, n_prepended=5)
    model_completions, h = calculate_h(dataset, 9)

    # Check model completions
    assert model_completions == [" L", " K", " m", " Q", " K"], "Unexpected model completions."

    # Check h-vector against expected vector (which was saved using this same code)
    assert h.shape == (D_MODEL,), f"Expected shape (d_model,), got {h.shape}"
    test_h = t.load(root / "data" / "test_h.pt")
    t.testing.assert_close(h, test_h, msg="Correct shape, but incorrect values.")

    print("All tests in `test_calculate_h` passed.")



def test_calculate_fn_vector(calculate_fn_vector: Callable[[AntonymDataset, List[Tuple[int, int]]], Tensor]) -> None:

    # This same code was used to initially calculate the fn vector (everything is deterministic)
    word_pairs = list(zip(string.ascii_lowercase, string.ascii_uppercase))
    dataset = AntonymDataset(word_pairs, size=3, n_prepended=2)
    fn_vector = calculate_fn_vector(dataset, [(8, 1)])

    # Check shape
    assert fn_vector.shape == (D_MODEL,), f"Expected shape (d_model,), got {fn_vector.shape}"

    # Load in the previously calculated fn vector, and test for equality
    fn_vector_expected = t.load(root / "data" / "test_fn_vector.pt")
    t.testing.assert_close(fn_vector, fn_vector_expected, msg="Correct shape, but incorrect values.")

    print("All tests in `test_calculate_fn_vector` passed.")