import re
import string
import gensim
import nltk
from nltk.tree import Tree
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

class TextPreprocessing():
    """Provides ways of preprocessing text data."""

    def __init__(self):
        self.punctuation = list(string.punctuation)
        self.stop_words = stopwords.words('english') + self.punctuation + ['\\n'] + ['quot']

    def tokenize_document(self, text):
        """Preprocess a whole raw document.

        Args:
            text (str): Raw string of text.

        Return:
            Nested lists where each list is a tokenized sentence of the document. Example: [['foo', 'bar', 'hello', 'world'], ['one', 'two', 'three'], ... ].

        """
        return [self.tokenize_sentence(sentence) for sentence in self.text2sentences(text)]

    def tokenize_sentence(self, text):
        """Preprocess a raw string/sentence of text.

        Args:
            text (str): Raw string of text.

        Return:
            tokens (list, str): Preprocessed tokens.

        """
        tokens = self.create_tokens(text)
        tokens = self.remove_stop_words(tokens)      
        return tokens

    def text2sentences(self, text):
        """Split raw text to sentences.

        Args:
            text (str): Raw text data.

        Return:
            List of strings, where each element is a sentence.

        """
        return nltk.sent_tokenize(text)

    def create_tokens(self, text):
        """Split a string into tokens.

        Args:
            text (str): Raw text data

        Return:
            tokens (str): The pieces of the string.

        """
        regex_str = ["http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),](?:%[0-9a-f][0-9a-f]))+", 
                     "(?:\w+-\w+){2}",
                     "(?:\w+-\w+)", 
                     "(?:\\\+n+)", 
                     "(?:@[\w_]+)", 
                     "<[^>]+>", 
                     "(?:\w+'\w)", 
                     "(?:[\w_]+)", 
                     "(?:\S)"]

        # Create the tokenizer which will be case insensitive and will ignore space.
        tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

        # Find all patterns and tokenize them
        tokens = tokens_re.findall(text)
        return tokens

    def remove_stop_words(self, tokens):
        """Remove the stop words and punctuation of a tokenized sentence.

        Args:
            tokens (list, str): Collection of the entities that make up a sentence.

        Return:
            A filtered list of tokens.

        """
        filtered_tokens = [self.fix_underscores(token.lower()) for token in tokens 
                            if token not in self.stop_words 
                            and len(token) > 2
                            and any(x in token for x in ('1234567890')) == False
                            and any(x in token for x in ('qwertyuioplkjhgfdaszxcvbnm-'))]

        return filtered_tokens

    def fix_underscores(self, token):
        """Replace '-' with an underscore."""
        return token.replace('-', '_')

    def tagged_documents(self, documents, labels):
        """Tag preprocessed document with its name.

        Args:
            documents (list, str): Nested lists with preprocessed tokens.
            labels (list, str): The tag for each nested list.

        Return:
            TaggedDocument object of the form (words=[tokens], tag=[label]).

        """

        return [gensim.models.doc2vec.TaggedDocument(documents[i], [labels[i]]) for i in range(len(labels))]

    def bigrams(self, documents):
        """Create bigrams using Gensim's phrases."""
        phrases = gensim.models.Phrases(documents)
        bigram = gensim.models.phrases.Phraser(phrases)
        return list(bigram[documents])