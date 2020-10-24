from abc import ABC, abstractmethod
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize 
nltk.download('punkt')

# This tokenizer doesn't work with tf.data.Dataset.map
# But helps preprocess data and store them in tfrecords
class Tokenizer(ABC):
    def __init__(self, maxlen = None, teacher_force=False):
        self.vocab_size = 0
        self.vocab2id, self.id2vocab = {}, {}
        self.maxlen = maxlen
        self.teacher_force = teacher_force
        #adding out-of-vocabulary token
        self.add_tokens(['<oov_token>'])
        if teacher_force:
            self.add_tokens(['<sos>', '<eos>'])

    def add_tokens(self, tokens):
        count = self.vocab_size
        for tkn in tokens:
            if tkn not in self.vocab2id:
                self.vocab2id.update({tkn:count})
                count+=1
        self.id2vocab = dict((v,k) for k,v in self.vocab2id.items())
        self.vocab_size = count
  
    def encode(self, list_of_tokens):
        enc_seq = []
        for tkn in list_of_tokens:
            if tkn not in self.vocab2id:
                enc_seq.append(self.vocab2id['<oov_token>'])
            else:
                enc_seq.append(self.vocab2id[tkn])
        return enc_seq
    
    def decode(self, list_of_ids):
        return ''.join([self.id2vocab[i] for i in list_of_ids])
    
    def __getitem__(self, key):
        if type(key)==str:
            return self.vocab2id[key]
        elif type(key)==int:
            return self.id2vocab[key]
    
    @abstractmethod
    def tokenize(self, string):
        pass

    def encode_n_pad(self, list_of_seqs):
        enc_seqs = []
        for seq in list_of_seqs:
            tkns = self.tokenize(seq)
            self.add_tokens(tkns)
            enc_seq = list(map(lambda x: self[x], tkns))
            if self.teacher_force:
                enc_seq = [self['<sos>']] + enc_seq + [self['<eos>']]
            enc_seqs.append(enc_seq)
        return pad_sequences(enc_seqs, maxlen = self.maxlen, padding="post")


class WordTokenizer(Tokenizer):
    def __init__(self, language, maxlen=None, teacher_force = False):
        super(WordTokenizer, self).__init__(maxlen=maxlen, teacher_force=teacher_force)
        self.language = language
        
    def tokenize(self, string):
        tokens = word_tokenize(string, language = self.language)
        return tokens


class CharTokenizer(Tokenizer):
    def __init__(self, maxlen=None, teacher_force = False):
        super(CharTokenizer, self).__init__(maxlen=maxlen, 
                                teacher_force=teacher_force)

    def tokenize(self, string):
        tokens = set(string)
        return tokens


