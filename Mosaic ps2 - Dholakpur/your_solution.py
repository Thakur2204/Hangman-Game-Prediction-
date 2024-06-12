from my_model import model
from n_gram import unigram_probs, bigram_probs, trigram_probs, fourgram_probs, fivegram_probs
from n_gram import build_n_grams
import random
import numpy as np

file_path = "training.txt" 
with open(file_path, "r") as file:
    data = file.read().splitlines()

#lowercase the corpus and remove the word which contain non-alphabetic characters
processed_words = []
for word in data:
    if word.isalpha():
        processed_words.append(word.lower())

#unique words in corpus
unique_words = list(set(processed_words))

unique_words = np.array(unique_words).tolist()
# training_set = unique_words[:-1000]
training_set = unique_words

letter_set = sorted(set("".join(training_set)))

unigram_og, bigram_og, trigram_og, fourgram_og, fivegram_og = build_n_grams(training_set)

def suggest_next_letter_sol(displayed_word, guessed_letters):
    """_summary_

    This function takes in the current state of the game and returns the next letter to be guessed.
    displayed_word: str: The word being guessed, with underscores for unguessed letters.
    guessed_letters: list: A list of the letters that have been guessed so far.
    Use python hangman.py to check your implementation.
    """
    
    incorrect_guessed = set(guessed_letters) - set(displayed_word)
    unigram, bigram, trigram, fourgram, fivegram = unigram_og, bigram_og, trigram_og, fourgram_og, fivegram_og

    # only recalibrate if last guess was incorrect and running low on guesses
    if len(incorrect_guessed) >= 3 and guessed_letters[-1] in incorrect_guessed:
        new_dict = [word for word in training_set if not set(word).intersection(incorrect_guessed)]
        unigram, bigram, trigram, fourgram, fivegram = build_n_grams(new_dict)

    guess = model(displayed_word, guessed_letters, unigram, 
                  bigram, trigram, fourgram, fivegram, letter_set)

    return guess

    raise NotImplementedError