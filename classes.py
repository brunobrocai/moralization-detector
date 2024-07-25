from dataclasses import dataclass
import spacy
import torch


@dataclass
class MetaData:
    title: str
    author: str
    date: str
    source: str
    other: str


class PossibleMoralization:

    def __str__(self):
        return self.__full_text

    def __repr__(self):
        return f'PossibleMoralization Object around: {self.__focus_sentence}'

    @property
    def full_text(self):
        return self.__full_text

    @property
    def focus_sentence(self):
        return self.__focus_sentence

    def update_full_text(self):
        self.__full_text = ' '.join((
            ' '.join(self.__precontext),
            self.__focus_sentence,
            ' '.join(self.__postcontext)
        ))

    @focus_sentence.setter
    def focus_sentence(self, value):
        self.__focus_sentence = value
        self.update_full_text()

    @property
    def precontext(self):
        return self.__precontext

    @precontext.setter
    def precontext(self, value):
        if len(value) > self.__precontext_len:
            print('Precontext too long, truncating')
            self.__precontext = value[-self.__precontext_len:]
        self.__precontext = value
        self.update_full_text()

    @property
    def postcontext(self):
        return self.__postcontext

    @postcontext.setter
    def postcontext(self, value):
        if len(value) > self.__postcontext_len:
            print('Postcontext too long, truncating')
            self.__postcontext = value[:self.__postcontext_len]
        self.__postcontext = value
        self.update_full_text()

    @property
    def metadata(self):
        return self.__metadata

    @metadata.setter
    def metadata(self, value):
        self.__metadata = value

    def __init__(
        self, precontext_len=2, postcontext_len=2,
        metadata=MetaData('', '', '', '', '')
    ):
        self.__full_text = ''
        self.__focus_sentence = ''
        self.__precontext = ''
        self.__postcontext = ''
        self.__precontext_len = precontext_len
        self.__postcontext_len = postcontext_len
        self.__metadata = metadata


class PossibleMoralizationDimi(PossibleMoralization):

    @property
    def dimi_words(self):
        return self.__dimi_words

    @dimi_words.setter
    def dimi_words(self, value):
        if value is None:
            value = []
        self.__dimi_words = value

    def find_dimi_words(self, dimi, spacy_model='de_core_news_lg'):
        nlp = spacy.load(spacy_model)
        doc = nlp(self.full_text)
        self.__dimi_words = [
            {
                'lemma': token.lemma_,
                'text': token.text
            }
            for token in doc if token.lemma_ in dimi
        ]

    def __init__(
        self, precontext_len=2, postcontext_len=2,
        metadata=MetaData('', '', '', '', ''), dimi_words=None
    ):
        super().__init__(precontext_len, postcontext_len, metadata)
        self.__dimi_words = dimi_words


class PossibleMoralizationModelled(PossibleMoralizationDimi):

    @property
    def logits(self):
        return self.__logits

    @logits.setter
    def logits(self, value):
        self.__logits = value
        self.label_from_logits()
        self.probabilities_from_logits()

    @property
    def label(self):
        return self.__label

    @property
    def probabilities(self):
        return self.__probabilities

    def label_from_logits(self):
        self.__label = self.__logits.argmax().item()

    def probabilities_from_logits(self):
        self.__probabilities = torch.nn.functional.softmax(
            self.__logits, dim=0
        )

    def __init__(
        self,
        super_instance,
        logits=None
    ):
        super().__init__(
            super_instance.__precontext_len,
            super_instance.__postcontext_len,
            super_instance.__metadata,
            super_instance.__dimi_words
        )
        self.__logits = logits
        self.__label = label
        self.__probabilities = None
