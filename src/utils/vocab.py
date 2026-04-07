class Vocab:
    def __init__(self):
        self.word2idx = {"<pad>": 0}
        self.idx2word = {0: "<pad>"}

    def add_sentence(self, sentence):
        for word in sentence.lower().split():
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, sentence):
        return [self.word2idx[word] for word in sentence.lower().split()]

    def __len__(self):
        return len(self.word2idx)