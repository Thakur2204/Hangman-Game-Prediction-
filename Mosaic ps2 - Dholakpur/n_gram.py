import numpy as np
import collections

def build_n_grams(dictionary):
        '''
        build nested dictionary containing occurences for n (1-5) sequences of letters
        unigrams and bigrams have an extra level for length of the word
        for unigram, take only unique letters within each word  
        '''
        uni_gram = collections.defaultdict(lambda: collections.defaultdict(int))
        bi_gram = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
        tri_gram = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
        four_gram = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int))))
        five_gram = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))))

        for word in dictionary:
            n = len(word)

            # Fill unigram
            for char in set(word):
                uni_gram[n][char] += 1

            # Fill n-grams
            for i in range(n - 1):
                bi_gram[n][word[i]][word[i+1]] += 1
                if i < n - 2:
                    tri_gram[word[i]][word[i+1]][word[i+2]] += 1
                    if i < n - 3:
                        four_gram[word[i]][word[i+1]][word[i+2]][word[i+3]] += 1
                        if i < n - 4:
                            five_gram[word[i]][word[i+1]][word[i+2]][word[i+3]][word[i+4]] += 1

        return uni_gram, bi_gram, tri_gram, four_gram, five_gram

def fivegram_probs(word, letter_set, guessed_letters, probabilities, fivegram):
    ''' 
    Input: the word with a '_' if letter has not been guessed
    Flow: uses fivegram to calculate the probability of a certain letter appearing in a five-letter sequence for a word of given length
    Output: Updated probabilities for each letter
    '''
            
    # vector of probabilities for each letter
    probs = [0] * len(letter_set)
    
    total_count = 0
    letter_count = [0] * len(letter_set)

    # traverse the word and find patterns that have three consecutive letters where one of them is blank
    for i in range(len(word) - 4):
                    
        # case 1: "letter letter letter letter blank"
        if word[i] != '_' and word[i+1] != '_' and word[i+2] != '_' and word[i+3] != '_' and word[i+4] == '_':
            letter_1 = word[i]
            letter_2 = word[i+1]
            letter_3 = word[i+2]
            letter_4 = word[i+3]
            
            # calculate occurrences of "letter_1 letter_2 letter_3 blank" and for each letter not guessed yet
            for j, letter in enumerate(letter_set):
                if fivegram[letter_1][letter_2][letter_3][letter_4][letter] > 0 and letter not in guessed_letters:
                    total_count += fivegram[letter_1][letter_2][letter_3][letter_4][letter]
                    letter_count[j] += fivegram[letter_1][letter_2][letter_3][letter_4][letter]
    
        # case 2: "letter letter letter blank letter"
        elif word[i] != '_' and word[i+1] != '_' and word[i+2] != '_' and word[i+3] == '_' and word[i+4] != '_':
            letter_1 = word[i]
            letter_2 = word[i+1]
            letter_3 = word[i+2]
            letter_4 = word[i+4]
            
            # calculate occurrences of "letter_1 letter_2 letter_3 blank" and for each letter not guessed yet
            for j, letter in enumerate(letter_set):
                if fivegram[letter_1][letter_2][letter_3][letter][letter_4] > 0 and letter not in guessed_letters:
                    total_count += fivegram[letter_1][letter_2][letter_3][letter][letter_4]
                    letter_count[j] += fivegram[letter_1][letter_2][letter_3][letter][letter_4]
           
        # case 3: letter letter blank letter letter
        elif word[i] != '_' and word[i+1] != '_' and word[i+2] == '_' and word[i+3] != '_' and word[i+4] != '_':
            letter_1 = word[i]
            letter_2 = word[i+1]
            letter_3 = word[i+3]
            letter_4 = word[i+4]
            
            # calculate occurrences of "letter_1 letter_2 blank letter_3" and for each letter not guessed yet
            for j, letter in enumerate(letter_set):
                if fivegram[letter_1][letter_2][letter][letter_3][letter_4] > 0 and letter not in guessed_letters:
                    total_count += fivegram[letter_1][letter_2][letter][letter_3][letter_4]
                    letter_count[j] += fivegram[letter_1][letter_2][letter][letter_3][letter_4]
           
        # case 4: letter blank letter letter letter
        elif word[i] != '_' and word[i+1] == '_' and word[i+2] != '_' and word[i+3] != '_' and word[i+4] != '_':
            letter_1 = word[i]
            letter_2 = word[i+2]
            letter_3 = word[i+3]
            letter_4 = word[i+4]
            
            # calculate occurrences of "letter_1 blank letter_2 letter_3" and for each letter not guessed yet
            for j, letter in enumerate(letter_set):
                if fivegram[letter_1][letter][letter_2][letter_3][letter_4] > 0 and letter not in guessed_letters:
                    total_count += fivegram[letter_1][letter][letter_2][letter_3][letter_4]
                    letter_count[j] += fivegram[letter_1][letter][letter_2][letter_3][letter_4]
    
        # case 5: blank letter letter letter letter
        elif word[i] == '_' and word[i+1] != '_' and word[i+2] != '_' and word[i+3] != '_' and word[i+4] != '_':
            letter_1 = word[i+1]
            letter_2 = word[i+2]
            letter_3 = word[i+3]
            letter_4 = word[i+4]
            
            # calculate occurrences of "blank letter_1 letter_2 letter_3" and for each letter not guessed yet
            for j, letter in enumerate(letter_set):
                if fivegram[letter][letter_1][letter_2][letter_3][letter_4] > 0 and letter not in guessed_letters:
                    total_count += fivegram[letter][letter_1][letter_2][letter_3][letter_4]
                    letter_count[j] += fivegram[letter][letter_1][letter_2][letter_3][letter_4]
    
    # calculate the probabilities of each letter appearing
    if total_count > 0:
        for i in range(len(letter_set)):
            probs[i] = letter_count[i] / total_count
    
    # interpolate probabilities
    for i, p in enumerate(probabilities):
        probabilities[i] = p + probs[i] * (0.40)
    
    return probabilities

def fourgram_probs(word, letter_set, guessed_letters, probabilities, fourgram):
    ''' 
    Calculate probabilities for each letter to be used in the next level of guessing.
    
    Inputs:
        word: The word with '_' if the letter has not been guessed
        letter_set: Set of possible letters
        guessed_letters: Letters that have been guessed
        probabilities: Probabilities of each letter based on previous guesses
        fourgram: Fourgram model containing probabilities of four-letter sequences
    
    Output:
        Updated probabilities for each letter
    '''
    # Initialize a list to hold probabilities for each letter
    probs = [0] * len(letter_set)
    
    total_count = 0
    letter_count = [0] * len(letter_set)

    # Traverse the word and find patterns that have three consecutive letters where one of them is blank
    for i in range(len(word) - 3):
        # Case 1: "letter letter letter blank"
        if word[i] != '_' and word[i+1] != '_' and word[i+2] != '_' and word[i+3] == '_':
            first_letter = word[i]
            second_letter = word[i+1]
            third_letter = word[i+2]
            
            # Calculate occurrences of "first_letter second_letter blank" for each letter not guessed yet
            for j, letter in enumerate(letter_set):
                if fourgram[first_letter][second_letter][third_letter][letter] > 0 and letter not in guessed_letters:
                    total_count += fourgram[first_letter][second_letter][third_letter][letter]
                    letter_count[j] += fourgram[first_letter][second_letter][third_letter][letter]
    
        # Case 2: "letter letter blank letter"
        elif word[i] != '_' and word[i+1] != '_' and word[i+2] == '_' and word[i+3] != '_':
            first_letter = word[i]
            second_letter = word[i+1]
            fourth_letter = word[i+3]
            
            # Calculate occurrences of "first_letter blank second_letter" for each letter not guessed yet
            for j, letter in enumerate(letter_set):
                if fourgram[first_letter][second_letter][letter][fourth_letter] > 0 and letter not in guessed_letters:
                    total_count += fourgram[first_letter][second_letter][letter][fourth_letter]
                    letter_count[j] += fourgram[first_letter][second_letter][letter][fourth_letter]
           
        # Case 3: "letter blank letter letter"
        elif word[i] != '_' and word[i+1] == '_' and word[i+2] != '_' and word[i+3] != '_':
            first_letter = word[i]
            third_letter = word[i+2]
            fourth_letter = word[i+3]
            
            # Calculate occurrences of "blank first_letter third_letter" for each letter not guessed yet
            for j, letter in enumerate(letter_set):
                if fourgram[first_letter][letter][third_letter][fourth_letter] > 0 and letter not in guessed_letters:
                    total_count += fourgram[first_letter][letter][third_letter][fourth_letter]
                    letter_count[j] += fourgram[first_letter][letter][third_letter][fourth_letter]
           
        # Case 4: "blank letter letter letter"
        elif word[i] == '_' and word[i+1] != '_' and word[i+2] != '_' and word[i+3] != '_':
            second_letter = word[i+1]
            third_letter = word[i+2]
            fourth_letter = word[i+3]
            
            # Calculate occurrences of "blank second_letter third_letter fourth_letter" for each letter not guessed yet
            for j, letter in enumerate(letter_set):
                if fourgram[letter][second_letter][third_letter][fourth_letter] > 0 and letter not in guessed_letters:
                    total_count += fourgram[letter][second_letter][third_letter][fourth_letter]
                    letter_count[j] += fourgram[letter][second_letter][third_letter][fourth_letter]
    
    # Calculate the probabilities of each letter appearing
    if total_count > 0:
        for i in range(len(letter_set)):
            probs[i] = letter_count[i] / total_count
    
    # Interpolate probabilities
    for i, p in enumerate(probabilities):
        probabilities[i] = p + probs[i] * (0.25)
    
    # Return the updated probabilities
    return probabilities

def trigram_probs(word, letter_set, guessed_letters, probabilities, trigram):
    ''' 
    Input: the word with a '_' if letter has not been guessed
    Flow: uses trigram to calculate the probability of a certain letter appearing in a three-letter sequence for a word of given length
    Output: updated probabilities for each letter 
    '''
            
    # vector of probabilities for each letter
    probs = [0] * len(letter_set)
    
    total_count = 0
    letter_count = [0] * len(letter_set)

    # traverse the word and find patterns that have three consecutive letters where one of them is blank
    for i in range(len(word) - 2):
                    
        # case 1: "letter letter blank"
        if word[i] != '_' and word[i+1] != '_' and word[i+2] == '_':
            letter_1 = word[i]
            letter_2 = word[i+1]
            
            # calculate occurrences of "letter_1 letter_2 blank" and for each letter not guessed yet
            for j, letter in enumerate(letter_set):
                if trigram[letter_1][letter_2][letter] > 0 and letter not in guessed_letters:
                    total_count += trigram[letter_1][letter_2][letter]
                    letter_count[j] += trigram[letter_1][letter_2][letter]
    
        # case 2: "letter blank letter"
        elif word[i] != '_' and word[i+1] == '_' and word[i+2] != '_':
            letter_1 = word[i]
            letter_2 = word[i+2]
            
            # calculate occurrences of "letter_1 blank letter_2" and for each letter not guessed yet
            for j, letter in enumerate(letter_set):
                if trigram[letter_1][letter][letter_2] > 0 and letter not in guessed_letters:
                    total_count += trigram[letter_1][letter][letter_2]
                    letter_count[j] += trigram[letter_1][letter][letter_2]
       
        # case 3: blank letter letter
        elif word[i] == '_' and word[i+1] != '_' and word[i+2] != '_':
            letter_1 = word[i+1]
            letter_2 = word[i+2]
            
            # calculate occurrences of "blank letter_1 letter_2" and for each letter not guessed yet
            for j, letter in enumerate(letter_set):
                if trigram[letter][letter_1][letter_2] > 0 and letter not in guessed_letters:
                    total_count += trigram[letter][letter_1][letter_2]
                    letter_count[j] += trigram[letter][letter_1][letter_2]
    
    # calculate the probabilities of each letter appearing
    if total_count > 0:
        for i in range(len(letter_set)):
            probs[i] = letter_count[i] / total_count
    
    # interpolate probabilities
    for i, p in enumerate(probabilities):
        probabilities[i] = p + probs[i] * (0.20)
     
    return probabilities

def bigram_probs(word, letter_set, guessed_letters, probabilities, bigram):
    ''' 
    Input: the word with a '_' if letter has not been guessed
    Flow: uses bigram to calculate the probability of a certain letter appearing in a two-letter sequence for a word of given length
    Output: Updated probabilities for each letter 
    '''
    # vector of probabilities for each letter
    probs = [0] * len(letter_set)
    
    total_count = 0
    letter_count = [0] * len(letter_set)
    
    # traverse the word and find either patterns of "letter blank" or "blank letter"
    for i in range(len(word) - 1):
        # case 1: "letter blank"
        if word[i] != '_' and word[i+1] == '_':
            letter_1 = word[i]
            
            # calculate occurrences of "letter_1 blank" and each letter not guessed yet
            for j, letter in enumerate(letter_set):
                if bigram[len(word)][letter_1][letter] > 0 and letter not in guessed_letters:
                    total_count += bigram[len(word)][letter_1][letter]
                    letter_count[j] += bigram[len(word)][letter_1][letter]
                        
        # case 2: "blank letter"
        elif word[i] == '_' and word[i+1]!= '_':
            letter_2 = word[i+1]
            
            # calculate occurrences of "blank letter_2" and each letter not guessed yet
            for j, letter in enumerate(letter_set):
                if bigram[len(word)][letter][letter_2] > 0 and letter not in guessed_letters:
                    total_count += bigram[len(word)][letter][letter_2]
                    letter_count[j] += bigram[len(word)][letter][letter_2]
                                                                
    # calculate the probabilities of each letter appearing
    if total_count > 0:
        for i in range(len(letter_set)):
            probs[i] = letter_count[i] / total_count

    # interpolate probabilities 
    for i, p in enumerate(probabilities):
        probabilities[i] = p + probs[i] * (0.10)
    
    return probabilities

def unigram_probs(word, letter_set, guessed_letters, probabilities, unigram):
    ''' 
    Input: the word with a '_' if letter has not been guessed
    Flow: uses unigram to calculate the probability of a certain letter appearing in any blank space
    Output: Updated probabilities for each letter 
    '''
            
    # vector of probabilities for each letter
    probs = [0] * len(letter_set)
    
    total_count = 0
    letter_count = [0] * len(letter_set)
    
    # traverse the word and find blank spaces
    for i in range(len(word)):
        # case 1: "letter blank"
        if word[i] == '_':
                            
            # calculate occurrences of pattern and each letter not guessed yet
            for j, letter in enumerate(letter_set):
                if unigram[len(word)][letter] > 0 and letter not in guessed_letters:
                    total_count += unigram[len(word)][letter]
                    letter_count[j] += unigram[len(word)][letter]
                   
    # calculate the probabilities of each letter appearing
    if total_count > 0:
        for i in range(len(letter_set)):
            probs[i] = letter_count[i] / total_count
            
    # interpolate probabilities
    for i, p in enumerate(probabilities):
        probabilities[i] = p + probs[i] * (0.05)

    return probabilities