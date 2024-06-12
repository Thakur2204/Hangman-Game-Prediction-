from n_gram import fivegram_probs, fourgram_probs, trigram_probs, bigram_probs, unigram_probs
import random

def model(word, guessed_letters, unigram, 
          bigram, trigram, fourgram, fivegram,  letter_set):
    
    probabilities = [0] * len(letter_set)
    
    probabilities = fivegram_probs(word, letter_set, guessed_letters, probabilities, fivegram)
    probabilities = fourgram_probs(word, letter_set, guessed_letters, probabilities, fourgram)
    probabilities = trigram_probs(word, letter_set, guessed_letters, probabilities, trigram)
    probabilities = bigram_probs(word, letter_set, guessed_letters, probabilities, bigram)
    probabilities = unigram_probs(word, letter_set, guessed_letters, probabilities, unigram)

    # adjust probabilities so they sum to one (not necessary but looks better)
    final_probs = [0] * len(letter_set)
    if sum(probabilities) > 0:
        for i in range(len(probabilities)):
            final_probs[i] = probabilities[i] / sum(probabilities)
        
    probabilities = final_probs
    
    # find letter with largest probability
    max_prob = 0
    guess_letter = ''
    for i, letter in enumerate(letter_set):
        if probabilities[i] > max_prob:
            max_prob = probabilities[i]
            guess_letter = letter
    
    # if no letter chosen from above, pick a random one (extra weight on vowels)
    if guess_letter == '':
        letters = letter_set.copy()
        random.shuffle(letters)
        letters_shuffled = ['e','a','i','o','u'] + letters
        for letter in letters_shuffled:
            if letter not in guessed_letters:
                return letter
        
    return guess_letter