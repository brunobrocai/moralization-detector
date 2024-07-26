from dataclasses import dataclass
import json
import spacy


@dataclass
class MetaData:
    """Dataclass for metadata of a text."""
    title: str
    author: str
    date: str
    source: str
    other: str

    def to_dict(self):
        return {
            'title': self.title,
            'author': self.author,
            'date': self.date,
            'source': self.source,
            'other': self.other
        }

    def from_json(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.title = data.get('title', None)
        self.author = data.get('author', None)
        self.date = data.get('date', None)
        self.source = data.get('source', None)
        self.other = data.get('other', None)


class PossibleMoralization:
    """Class for representing possible moralization in a text."""

    # +++ Getters & Setters +++

    @property
    def full_text(self):
        return self.__full_text

    @property
    def focus_sentence(self):
        return self.__focus_sentence

    @focus_sentence.setter
    def focus_sentence(self, value):
        self.__focus_sentence = value
        self.update_full_text()

    @property
    def precontext(self):
        return self.__precontext

    @precontext.setter
    def precontext(self, value):
        if len(value) > self.__context_window:
            print('Precontext too long, truncating')
            self.__precontext = value[-self.__context_window:]
        self.__precontext = value
        self.update_full_text()

    @property
    def postcontext(self):
        return self.__postcontext

    @postcontext.setter
    def postcontext(self, value):
        if len(value) > self.__context_window:
            print('Postcontext too long, truncating')
            self.__postcontext = value[:self.__context_window]
        self.__postcontext = value
        self.update_full_text()

    @property
    def metadata(self):
        return self.__metadata

    @metadata.setter
    def metadata(self, value):
        self.__metadata = value

    @property
    def context_window(self):
        return self.__context_window

    @context_window.setter
    def context_window(self, value):
        self.__context_window = value

    # +++ Magic Methods +++

    def __str__(self):
        return self.__full_text

    def __repr__(self):
        return f'PossibleMoralization Object around: {self.__focus_sentence}'

    def __init__(
        self, context_window=2,
        metadata=MetaData('', '', '', '', '')
    ):
        self.__full_text = ''
        self.__focus_sentence = ''
        self.__precontext = ''
        self.__postcontext = ''
        self.__context_window = context_window
        self.__metadata = metadata

    # +++ Methods +++

    def update_full_text(self):
        self.__full_text = ' '.join((
            ' '.join(self.__precontext),
            self.__focus_sentence,
            ' '.join(self.__postcontext)
        ))

    def to_dict(self):
        return {
            'focus_sentence': self.__focus_sentence,
            'precontext': self.__precontext,
            'postcontext': self.__postcontext,
            'metadata': self.metadata.to_dict(),
            'full_text': self.__full_text
        }


class PossibleMoralizationDimi(PossibleMoralization):
    """Class for representing possible moralization, incl. DIMI words."""

    # +++ Getters & Setters +++

    @property
    def dimi_words(self):
        return self.__dimi_words

    @dimi_words.setter
    def dimi_words(self, value):
        if value is None:
            value = []
        self.__dimi_words = value

    # +++ Magic Methods +++

    def __init__(
        self, context_window=2,
        metadata=MetaData('', '', '', '', ''), dimi_words=None
    ):
        super().__init__(context_window, metadata)
        self.__dimi_words = dimi_words

    # +++ Methods +++

    def to_dict(self):
        return {
            **super().to_dict(),
            'dimi_words': self.__dimi_words
        }

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


class PossibleMoralizationModelled(PossibleMoralizationDimi):
    """Class for representing possible moralization, incl. model output."""

    # +++ Getters & Setters +++

    @property
    def logits(self):
        return self.__logits

    @logits.setter
    def logits(self, value):
        self.__logits = value
        self.probabilities_from_logits()
        self.label_from_probabilities()

    @property
    def label(self):
        return self.__label

    @property
    def probabilities(self):
        return self.__probabilities

    # +++ Magic Methods +++

    def __repr__(self):
        return f'PossibleMoralization Object around: {self.__focus_sentence}'

    def __str__(self):
        return (
            f'"{self.__focus_sentence}" ...with '
            f'moralizing content {"not " if self.__label == 0 else ""} '
            f'detected by the classification model.'
        )

    def __init__(
        self,
        super_instance,
        logits=None
    ):
        super().__init__(
            super_instance.context_window,
            super_instance.metadata,
            super_instance.dimi_words
        )
        self.precontext = super_instance.precontext
        self.postcontext = super_instance.postcontext
        self.focus_sentence = super_instance.focus_sentence
        self.__label = None
        self.__probabilities = None
        self.logits = logits

    # +++ Methods +++

    def label_from_probabilities(self):
        self.__label = int(self.__probabilities.argmax().item())

    def probabilities_from_logits(self):
        prob_tensor = torch.sigmoid(self.__logits)
        self.__probabilities = prob_tensor.cpu().numpy().flatten()

    def to_dict(self):
        return {
            **super().to_dict(),
            'label': self.__label,
            'probabilities': self.__probabilities.tolist()
        }
