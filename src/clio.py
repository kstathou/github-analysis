import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KDTree

class Tfidf():
    """Calculates the TF-IDF weighting of a corpus."""
    def __init__(self, docs):
        """Args:
            docs (list, str): Nested lists with preprocessed tokens. Example: [['foo', 'bar'], ['alpha', 'beta']]

        """
        self.documents = docs

    def tfidf_transformer(self):
        """Calculates the TF-IDF. The TfidfVectorizer() passes the tokens as they are, without preprocessing.
        
        Args:
            self.documents (list, str): Nested lists with preprocessed tokens.
            
        Returns:
            tfidf: Trained instance of the sklearn TfidfVectorizer.
            X (csr_matrix): Sparse matrix with the IDF weighting of the tokens and shape (documents, tokens).
            feature_names (list, str): Unique tokens used in TfidfVectorizer. 
        
        """
        tfidf = TfidfVectorizer(tokenizer=lambda x:x, lowercase=False)
        X = tfidf.fit_transform(self.documents)
        feature_names = tfidf.get_feature_names()
        return tfidf, X, feature_names

class WordQueries(Tfidf):
    """Expand the query term list with words similar to the original input. The word similarity is found based on pretrained vectors and TFIDF weighting is used to filter out rare and very frequent terms from the list."""
    def __init__(self, word2vec, docs):
        """Args:
            w2v (word2vec): Pretrained word vectors.
            tfidf: Trained instance of the sklearn TfidfVectorizer.
            
        """
        super().__init__(docs)
        self.w2v = word2vec
        self.tfidf, _, _ = self.tfidf_transformer()

    def similar_tokens(self, token, sim_tokens):
        """Find the most similar tokens using a pretrained word2vec model. The top 25 words are considered.

        Args:
            token (str): Query token.

        Returns:
            tokens (set, str): Top 25 most similar to the input token. The input term is also added to the set.

        """
        # Add the query word directly into the list because it will not be in the similar ones.
        tokens = [token]
        
#         for token in [tup[0] for tup in self.w2v.most_similar([token], topn=20)]:
#             tokens.extend([tup[0] for tup in self.w2v.most_similar([token], topn=25) if tup[1] > 0.55])

        tokens.extend([tup[0] for tup in self.w2v.most_similar([token], topn=sim_tokens)])
        return set(tokens)

    def word2ids(self, tokens):
        """Find the TF-IDF IDs of tokens. 

        Args:
            tokens (set, str): Unique, preprocessed tokens.

        Returns:
            token_ids(list, int): Token IDs of the TF-IDF dictionary.

        """
        token_ids = []
        for token in set(tokens):
            try:
                token_ids.append(self.tfidf.vocabulary_[token])
            except Exception as e:
                continue
                
        return token_ids

    def high_idf_ids(self, token_ids, bottom_lim, upper_lim):
        """Filter out tokens with extreme TF-IDF weights (both rare and frequent tokens).

        Args:
            token_ids: (list, int): Token IDs of the TF-IDF dictionary.

        Returns:
            ids: (list, int): Token IDs of the TF-IDF dictionary.

        """
        idfs = {token_id:self.tfidf.idf_[token_id] for token_id in token_ids}
        # Keeping only IDs of tokens that have score between the [35,95] of the IDF distribution.
        ids = [k for k, v in idfs.items() 
               if v >= np.percentile([val for val in idfs.values()], [bottom_lim])[0]
               and v < np.percentile([val for val in idfs.values()], [upper_lim])[0]]
        return ids

    def tfidf_id2word(self, id_):
        """Find a token using its TF-IDF ID.

        Args:
            id_ (int): TF-IDF id of a token.

        Returns:
            Token (str) that corresponds to the TF-IDF id.

        """
        return list(self.tfidf.vocabulary_.keys())[list(self.tfidf.vocabulary_.values()).index(id_)]

    def query_word(self, token, sim_tokens=25, bottom_lim=35, upper_lim=95):
        """Wrapper function around the WordQueries class.

        Args:
            token (str): Token to query the engine with.

        Returns:
            queries (set, str): Unique tokens that are similar to the queried token and do not belong into the extremes.

        """
        sim_tokens = self.similar_tokens(token, sim_tokens)
        token_ids = self.word2ids(sim_tokens)
        high_ids = self.high_idf_ids(token_ids, bottom_lim, upper_lim)
        queries = [self.tfidf_id2word(id_) for id_ in high_ids]
        queries.append(token)
        return set(queries)

class Clio(WordQueries):
    """Information retrieval based on word embeddings. 

    TODO: 
    * Change the names of the composed classes from feature, word_queries to something else.
    * Top 25 words in word2vec might be a bit restrictive. How to find the optimal number of similar tokens?
    * The portion of the IDF distribution we keep was selected on a qualitative basis. How to optimise the filtering process?
    * The intersection (>=1) makes the querying flexible, but what about the non-related words that might be passing through?

    """
    def __init__(self, dataframe, documents, word2vec, d2v):
        """Args:
            tfidf_model: Instance of the Tfidf class.
            word_queries: Instance of the WordQueries class.
            df: Pandas dataframe that contains the InnovateUK data.

        """
        self.tfidf_model = Tfidf(documents)
        self.word_queries = WordQueries(word2vec, documents)
        self.df = dataframe
        self.df['paragraph vectors'] = [d2v.docvecs[id_] for id_ in self.df['project_id']]
        self.kdt = KDTree(self.df['paragraph vectors'].tolist(), leaf_size=3)
        self.df = self.df.reindex(columns=(list([a for a in self.df.columns if a != 'project_id']) + ['project_id']))

    def query_kdt(self, query, num_results):
        """Create a KDTree using the pretrained paragraph vectors and query it with a project title.

        Args:
            query (str): Project ID.

        Returns:
            dist (array, float): Euclidean distance between the query term and the results.
            idx (array, int): Index of the results, ordered by their distance to the query (ascending order).

        """
        return self.kdt.query(self.df['paragraph vectors'][self.df['project_id'] == query].tolist(), k=num_results)

    def relevant_doc_ids(self, idx, keywords):
        """Find the relevant documents IDs based on their intersection with the query words.

        Args:
            idx (array, int): Index of relevant paragraph vectors, ordered by their distance to the query (ascending order).
            keywords (set, str): Query terms.

        Returns:
            List of document indexes (int) that have at least one word the same with the query terms.

        """
        return [id_ for id_ in idx[0] if len(keywords.intersection(self.tfidf_model.documents[id_])) >= 1]
    
    def search_queries(self, token=None, custom_list=None, sim_tokens=25, bottom_lim=35, upper_lim=95):
        """Wrapper function around the GtrSearch class.

        Args:
            word (str): Query term to search the dataset with.

        Returns:
            Subset of the dataframe with unique projects that are relevant to the query term.

        """
        if token:
            queries = self.word_queries.query_word(token, sim_tokens, bottom_lim, upper_lim)
        else:
            queries = set(custom_list)
            
        print(queries)
        
        # Dictionary with the following format {doc index: number of words from the queries in it}.
        doc_intersection = {i:len(queries.intersection(doc)) for i, doc in enumerate(self.tfidf_model.documents)}
        
        # Find the ID of the document that contains the maximum number of query terms.
        max_id = list(doc_intersection.keys())[list(doc_intersection.values()).index(max([v for k,v in doc_intersection.items()]))]
        
        # -6 shows the index of the paragraph vector column in df.
        dist, idx = self.query_kdt(query=self.df.iloc[max_id, -1], num_results=self.df.shape[0])
        
        # Indexes of all the relevant documents sorted by their similarity to the document with the max_id
        idx = self.relevant_doc_ids(idx, queries)
        
        return self.df.iloc[idx, :]