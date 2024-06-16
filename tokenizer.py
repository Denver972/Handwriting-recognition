# Tokenize the words and characters to prep for classification
"""
Class with methods to create the tokens for
"""

import pandas as pd


class Tokenize():
    """
    Creates word tokens and character tokens for the dataset
        INPUT: csv file containing the hand labeled words
        OUTPUT: word tokens and character tokens
        Special words/characters: ->64/@
                                  EMPTY represents an empty cell -> 
    """

    def __init__(self, file, pad_value=0):
        """
        Allowed characters
        """
        self.file = file
        self.pad_value = pad_value
        self.df = pd.read_csv(self.file)
        self.unique_words = self.df["Label"].unique()
        self.max_word_length = self.get_max_word_length()
        self.word_tokens = self.get_word_token()
        self.word_to_number = self.word_to_token()
        self.number_to_word = self.token_to_word()
        self.characters = self.get_allowed_characters()
        self.char_tokens = self.get_char_token()
        self.char_to_number = self.char_to_token()
        self.number_to_char = self.token_to_char()

    def create_csv(self, file_name):
        """
        INPUT: File name of the csv that will be outputed
        OUTPUT: Csv containing the path and tokens required for classification
        """
        new_df = self.df.copy()

        new_df["Class"] = new_df["Label"].map(self.word_to_number)
        new_df["CharList"] = new_df["Label"].apply(self.word_to_char_list)
        new_df["TokenList"] = new_df["CharList"].apply(
            self.char_list_to_token_list)

        new_df.to_csv(file_name, index=False)

    def get_max_word_length(self):
        """
        INPUT: csv containing labels
        OUTPUT: length of the longest word
        """
        self.max_word_length = -1
        for word in self.unique_words:
            if len(word) > self.max_word_length:
                self.max_word_length = len(word)
        return self.max_word_length

    def get_word_token(self):
        """
        Asign a number or "token" to each unique word
        """
        number = []
        for ix, word in enumerate(self.unique_words):
            number.append(ix)

        return number

    def word_to_token(self):
        """
        convert the words to a number 
        """
        classes = dict(
            map(lambda i, j: (i, j), self.unique_words, self.word_tokens))

        return classes

    def token_to_word(self):
        """
        Convert from the number to the word
        """
        inv_classes = {i: j for j, i in self.word_to_number.items()}

        return inv_classes

    def get_allowed_characters(self):
        """
        Get the unique characters in the labels and add them to a list
        Remove the special words ENPTY and CIRCLE from the unique list
        before passing to the character level as these characters may not
        be present in the dataset. Then sort the characters by their ascii
        value. Let @ represent the CIRCLE in a single character
        """
        characters = []
        modified_unique_words = self.unique_words.copy()
        modified_unique_words = modified_unique_words[modified_unique_words != "EMPTY"]
        # modified_unique_words = modified_unique_words[modified_unique_words != "CIRCLE"]

        modified_unique_words = "".join(map(str, modified_unique_words))
        characters = list(set(modified_unique_words))
        characters = sorted(characters, key=lambda s: list(map(ord, s)))
        return characters

    def get_char_token(self):
        """
        Asign a number or "token" to each unique char. Use the ASCII value
        for the letters. Bounded by 32 and 126. Need to come up with a way to 
        symbolise an empty charatcter and the circle character. Let empty 
        character_token be 0 and circle be 1
        """
        number = []
        for char in self.characters:
            number.append(ord(char))

        return number

    def char_to_token(self):
        """
        convert the character to a number
        """
        classes = dict(
            map(lambda i, j: (i, j), self.characters, self.char_tokens))

        return classes

    def token_to_char(self):
        """
        convert the token back into the character. Just takes the inverse
        of the char_to_token dictionary
        """
        inv_classes = {i: j for j, i in self.char_to_number.items()}

        return inv_classes

    def word_to_char_list(self, word):
        """
        Converts the word to a list of characters
        """
        char_list = list(word)
        return char_list

    def char_list_to_word(self, char_list):
        """
        Convert a list of characters to a string 
        """
        word = "".join(char_list)
        return word

    def char_list_to_token_list(self, char_list):
        """
        Converts the list of characters to a list of tokens and adds padding
        INPUT: List of characters
        OUTPUT: List of tokens
        Note ascii of @ is 64
        """

        if char_list == ["E", "M", "P", "T", "Y"]:
            token_list = [0]
        else:
            token_list = [self.char_to_number[i] for i in char_list]

        while self.max_word_length - len(token_list) > 0:
            token_list.append(0)

        return token_list

    def token_list_to_char_list(self, token_list):
        """
        Converts the list of characters to a list of tokens and adds padding
        INPUT: List of tokens
        OUTPUT: List of characters with padding removed
        """

        if token_list == [0]*self.max_word_length:
            char_list = ["E", "M", "P", "T", "Y"]
        else:
            # remove the padding value of 0
            mod_token_list = [i for i in token_list if i != 0]
            char_list = [self.number_to_char[i] for i in mod_token_list]

        return char_list
