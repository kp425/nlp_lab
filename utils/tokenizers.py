from abc import ABC, abstractmethod
import nltk
from nltk.tokenize import word_tokenize 
nltk.download('punkt')

class Tokenizer(ABC):
    def __init__(self):
        self.vocab_size = 0
        self.vocab2id, self.id2vocab = {}, {}

    def add_tokens(self, tokens):
        count = self.vocab_size
        for tkn in tokens:
            if tkn not in self.vocab2id:
                self.vocab2id.update({tkn:count})
                count+=1
        self.id2vocab = dict((v,k) for k,v in self.vocab2id.items())
        self.vocab_size = count
  
    def encode(self, list_of_words):
        return [self.vocab2id[i] for i in list_of_words]
    
    def decode(self, list_of_ints):
        return ''.join([self.id2vocab[i] for i in list_of_ints])
    
    def __getitem__(self, key):
        if type(key)==str:
            return self.vocab2id[key]
        elif type(key)==int:
            return self.id2vocab[key]
    
    @abstractmethod
    def tokenize(self):
        pass


class WordTokenizer(Tokenizer):
    def __init__(self, language):
        super(WordTokenizer, self).__init__()
        self.language = language

    def tokenize(self, text):
        if isinstance(text, list):
            text = " ".join(text)
        tokens = word_tokenize(text, language = self.language)
        self.add_tokens(tokens)

class CharTokenizer(Tokenizer):
    def __init__(self):
        super(CharTokenizer, self).__init__()
    def tokenize(self, text):
        if isinstance(text, list):
            text = " ".join(text)
        tokens = set(text)
        self.add_tokens(tokens)



