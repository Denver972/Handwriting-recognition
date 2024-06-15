# Tokenize the words and characters to prep for classification


import pandas as pd


class Tokenize():
    """
    Creates word tokens and character tokens for the dataset
        INPUT: csv file containing the hand labeled words
        OUTPUT: word tokens and character tokens
        Special words/characters: CIRCLE represents a circle split into quadrants->98
                                  EMPTY represents an empty cell -> 99
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
        self.characters = self.get_allowed_characters()
        self.char_tokens = self.get_char_token()
        self.char_to_number = self.char_to_token()
        self.number_to_char = self.token_to_char()

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
        Asign a number or "token" to each unique word.
        """

    def get_allowed_characters(self):
        """
        Get the unique characters in the labels and add them to a list
        Remove the special words ENPTY and CIRCLE from the unique list
        before passing to the character level as these characters may not
        be present in the dataset. Then sort the characters by their ascii
        value
        """
        characters = []
        modified_unique_words = self.unique_words.copy()
        modified_unique_words = modified_unique_words[modified_unique_words != "EMPTY"]
        modified_unique_words = modified_unique_words[modified_unique_words != "CIRCLE"]

        modified_unique_words = "".join(map(str, modified_unique_words))
        characters = list(set(modified_unique_words))
        characters = sorted(characters, key=lambda s: list(map(ord, s)))
        return characters

    def get_char_token(self):
        """
        Asign a number or "token" to each unique word. Use the ASCII value
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
        inv_classes = {v: k for k, v in self.char_to_number.items()}

        return inv_classes
