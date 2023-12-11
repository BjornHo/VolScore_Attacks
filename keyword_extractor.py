import multiprocessing
from collections import Counter
from functools import reduce
from typing import List, Dict

import nltk
import numpy as np
import pandas as pd
from nltk import PorterStemmer, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm


# Callable class used to parallelize occurrence matrix computation
class OccRowComputer:

    # Initialize and create vocabulary
    def __init__(self, sorted_voc_with_occ):
        self.voc = [word for word, occ in sorted_voc_with_occ]

    # Checks if a list of words are inside a document.
    # It returns a row that consists of zeroes and ones.
    # One if the word is inside the vocabulary and inside the word list.
    # And else it is zero.
    # This is creates a row inside a document to keyword matrix.
    def __call__(self, word_list):
        return [int(voc_word in word_list) for voc_word in self.voc]


# Keyword extractor class that will create the keywords from a given dataframe and vocabulary size.
# The min_freq indicates the minimum frequency of a keyword, by default = 1.
class KeywordExtractor:
    def __init__(self, df, voc_size, min_freq=1, **kwargs):

        # Dictionary that stores {keyword: count}.
        # This dictionary is in sorted in descending order on the count value.
        glob_freq_dict = {}

        # Dictionary that stores {file_path: [keywords]} for the Enron case.
        # For the apache case it stores {Message-ID: [keywords]}.
        freq_dict = {}

        # Multi processing part to parallelize stemming and counting messages.
        nr_cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=nr_cores) as pool:
            results = pool.starmap(self.stem_count_all_msg, enumerate(np.array_split(df, nr_cores)))
            freq_dict, glob_freq_dict = reduce(self._merge_results, results)

        # Creation of the vocabulary
        # It transforms the dictionary into a list of pairs.
        # glob_freq_list contains (keyword, occurrence) pairs for example ('john', 2)
        glob_freq_list = nltk.FreqDist(glob_freq_dict)

        # Choose the "voc_size" most common keywords if specified, else choose all.
        glob_freq_list = (glob_freq_list.most_common(voc_size) if voc_size else glob_freq_list.most_common())

        # Sort the vocabulary based on the count, and only choose words for the voc that are >= min_freq.
        self.sorted_voc_with_occ = sorted(
            [(word, count) for word, count in glob_freq_list if count >= min_freq],
            key=lambda d: d[1],
            reverse=True,
        )

        # Creation of the occurrence matrix (document to keyword). Documents are ordered.
        self.occ_array = self.build_occurrence_array(sorted_voc_with_occ=self.sorted_voc_with_occ, freq_dict=freq_dict)

        # Array with all documents
        self.docs_array = list(freq_dict.keys())

        # Set the file_path as index for the dataframe.
        df = df.set_index('file_path')

        # Store the volume of each document in the order of self.docs_array
        self.vol_array = [df.loc[doc]['volume'] for doc in self.docs_array]

        # # Some statistics for the report.
        # # Remember to input the whole dataframe into the keyword extractor, and not a split dataframe.
        # nr_docs = len(freq_dict)
        # print("nr. docs", len(freq_dict))
        # print("nr. kws", len(glob_freq_dict))
        # print("nr. unique volumes", len(set(self.vol_array)))
        # print("avg nr. kw/doc", sum([len(x) for x in freq_dict.values()]) / nr_docs)
        # sys.exit()

        # Initialize inverted index
        self.inv_index = {}

        # Create inverted index
        self.gen_inv_index()

        # Remove them to save space
        del df
        del glob_freq_dict
        del glob_freq_list

        # Something went wrong, raise exception
        if not self.occ_array.any():
            raise ValueError("occurrence array is empty")

    @staticmethod
    def _merge_results(res1, res2):
        merge_results2 = Counter(res1[1]) + Counter(res2[1])
        merge_results1 = res1[0].copy()
        merge_results1.update(res2[0])
        return merge_results1, merge_results2

    # For a given message we stem it and remove any word that is part of the ban list.
    # We count the amount of occurrences of each word and return that.
    @staticmethod
    def stem_count_single_msg(msg):
        ban_list = stopwords.words('english')
        ban_list.extend(['from', 'to', 'subject', 'cc', 'forward'])
        ps = PorterStemmer()

        # Stem all words and check if alphanumeric and not in ban_list
        stemmed_word_list = [
            ps.stem(word.lower())
            for sentence in sent_tokenize(msg)
            for word in word_tokenize(sentence)
            if word.lower() not in ban_list and word.isalnum()
        ]

        # It contains (keyword, occurrence) pairs for example ('john', 2)
        return nltk.FreqDist(stemmed_word_list)

    # Stem and count all messages
    # It returns glob_kw_freq which is map from keyword to count,
    # and path_to_keyword_map is {file_path: [keywords]}, the keywords are inside that document.
    @staticmethod
    def stem_count_all_msg(ind, df):
        glob_kw_freq_map = {}
        path_to_keyword_map = {}
        for row_tuple in tqdm(iterable=df.itertuples(), desc=f"Extracting corpus vocabulary (Core {ind})",
                              total=len(df), position=ind):

            keywords_freq = KeywordExtractor.stem_count_single_msg(row_tuple.content)
            path_to_keyword_map[row_tuple.file_path] = []
            # For each message/file store keywords that are in that file, and also the number of occurrences.
            for keyword, count in keywords_freq.items():
                path_to_keyword_map[row_tuple.file_path].append(keyword)
                try:
                    glob_kw_freq_map[keyword] += 1
                except KeyError:
                    # 1 Because we only allow single occurrence keywords in the document
                    glob_kw_freq_map[keyword] = 1

        return path_to_keyword_map, glob_kw_freq_map

    # Builds the occurrence matrix where the rows are documents and columns are keywords.
    # Each cell consists of zeroes and ones if the document contains the keyword.
    @staticmethod
    def build_occurrence_array(sorted_voc_with_occ: List, freq_dict: Dict) -> pd.DataFrame:
        occ_list = []

        # Multi processing
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for row in tqdm(
                    # ORDERED. The original code used unordered, which will give us trouble when using volume...
                    pool.imap(OccRowComputer(sorted_voc_with_occ), freq_dict.values()),
                    desc="Computing the occurrence array",
                    total=len(freq_dict.values()),
            ):
                occ_list.append(row)
        return np.array(occ_list, dtype=np.float64)

    # Returns the sorted vocabulary without the occurrences.
    def get_sorted_voc(self) -> List[str]:
        return list(dict(self.sorted_voc_with_occ).keys())

    def gen_inv_index(self):
        # Transpose the occurrence array so that we get keywords as rows and documents as columns.
        # Each cell is a zero or 1, indicating if the keyword is inside the document.
        kw_to_docs = self.occ_array.T

        # Retrieve the sorted vocabulary of keywords
        sorted_voc = self.get_sorted_voc()

        # Create the inverted index {keyword: [doc_indices]}. So each keyword will give us a list of documents that
        # contain this keyword.
        self.inv_index = {}
        for kw_ind, doc_list in enumerate(kw_to_docs):
            doc_indices = np.where(doc_list == 1)[0]
            self.inv_index[sorted_voc[kw_ind]] = list(doc_indices)

        print("DONE INV INDEX!!")


# Splits the dataframe, one part the attacker knows. The other part is stored on the server.
def split_df(dframe, frac=0.5):
    first_split = dframe.sample(frac=frac)
    second_split = dframe.drop(first_split.index)
    return first_split, second_split