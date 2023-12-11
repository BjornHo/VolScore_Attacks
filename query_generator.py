import math
import random
import warnings
from typing import List, Tuple, Dict

import numpy as np
from scipy import stats

from keyword_extractor import KeywordExtractor


# A keyword extractor augmented with a query generator.
# It corresponds to the index in the server. The fake queries are seen by the attacker.
class QueryResultExtractor(KeywordExtractor):
    def __init__(self, *args, distribution="uniform", **kwargs):
        super().__init__(*args, **kwargs)

        # n is the amount of keywords in the vocabulary
        n = len(self.sorted_voc_with_occ)

        # Creates an array [1, 2, ..., n]
        x = np.arange(1, n + 1)

        # Every weight of each keyword is equal, and the sum of all weights is equal to 1.
        if distribution == "uniform":
            weights = np.ones(n) / n

        # The keywords are ordered on occurrences. The 1st keyword should have a higher weight than the 2nd.
        # And the second higher than the 3rd, and so on.
        # We do this by using x^(-a). And finally normalize the weights.
        elif distribution == "zipfian":
            a = 1.0
            weights = x ** (-a)
            weights /= weights.sum()

        # Inverse of zipfian
        elif distribution == "inv_zipfian":
            a = 1.0
            weights = (n - x + 1) ** (-a)
            weights /= weights.sum()
        else:
            raise ValueError("Distribution not supported.")

        self._rv = stats.rv_discrete(name="query_distribution", values=(x, weights))

    # This creates a sample by randomly selecting indices that are used for generating queries.
    def _generate_random_sample(self, size: int = 1) -> List[int]:
        # Creates a unique sample set with given size, which contains list of indices
        # that are randomly selected based on the distribution.
        sample_set = set(self._rv.rvs(size=size) - 1)
        nr_queries_remaining = size - len(sample_set)

        # The process is repeated until having the correct size.
        # It is because we could have had selected duplicates which are removed using set().
        while nr_queries_remaining > 0:
            # Create new sample_set and merge using union
            sample_set = sample_set.union(self._rv.rvs(size=nr_queries_remaining) - 1)
            nr_queries_remaining = size - len(sample_set)

        # Cast needed to index np.array
        sample_list = list(sample_set)

        # Returns a list of integers that correspond to selected indices for the queries.
        return sample_list

    # Function to generate the queries
    def get_fake_queries(self, size=1, hide_nb_files=True) -> Tuple[np.array, List[str]]:
        # Retrieve a list of randomly chosen indices that are used for the queries.
        sample_list = self._generate_random_sample(size=size)

        # Create the query vocabulary
        query_voc = [self.sorted_voc_with_occ[index][0] for index in sample_list]

        # Creates a document to trapdoor occurrence array.
        # We do this by retrieving a sub matrix from the original occurrence array (doc to keywords),
        # by only selecting column indices from the sample_list which are the trapdoor indices.
        query_arr = self.occ_array[:, sample_list]

        # We remove every line containing only zeros, so we hide the nr of documents stored.
        # i.e. we remove every document not involved in the queries.
        if hide_nb_files:
            query_arr = query_arr[~np.all(query_arr == 0, axis=1)]
        return query_arr, query_voc


# Generate a number of known queries from keywords that are in both similar_wordlist and stored_wordlist.
# It returns dictionary {keyword: keyword}. Note that we do not have {trapdoor: keyword}.
# This will be replaced later on when the keywords are encrypted.
def generate_known_queries(similar_wordlist: List[str], stored_wordlist: List[str], nr_queries: int) -> Dict[str, str]:
    # Possible candidate keywords that are in both sets
    candidates = list(set(similar_wordlist).intersection(stored_wordlist))

    if len(candidates) == 0:
        raise ValueError("Did not find any candidates for gen known queries")

    # Returns a number of known queries in the form of a dictionary {keyword: keyword}.
    return {word: word for word in random.sample(candidates, nr_queries)}


# Class that inherits the QueryResultExtractor and adds the access pattern padding countermeasure.
# As described by D.Cash, P.Grubbs, J.Perry and T. Ristenpart. Leakage-abuse attacks against searchable encryption. 2015

# The idea is to add fake documents to disguise the exact amount of documents that are returned when doing a query.
# The client can easily filter out these fake documents, and get the results he wants.
# For each keyword, you can pad the occurrences to an integer multiple of amount "n". We can do this by either
# adding rows(fake documents), or changing existing zeroes to ones.
class PaddedResultExtractor(QueryResultExtractor):
    def __init__(self, *args, n_pad=500, padding_and_volume_hiding=False, **kwargs):
        self.occ_array = np.array([])
        super().__init__(*args, **kwargs)

        # n is the padding size.
        self._n = n_pad

        # Retrieve the amount of rows and columns.
        nrows, ncol = self.occ_array.shape

        # Store total number of occurrences to check overhead after padding.
        self._number_real_entries = np.sum(self.occ_array)

        # Loop through all columns(which are keywords).
        for j in range(ncol):

            # Total number of occurrences for a keyword.
            nr_entries = sum(self.occ_array[:, j])

            # Note that if nb_entries < self._n, when using ceil the result is 1. So self._n - nr_entries
            # is the amount fake entries to add.
            nr_fake_entries_to_add = int(math.ceil(nr_entries / self._n) * self._n - nr_entries)

            # Create a list where the occurrence values equal to 0, and collapse this 2D array into a 1D array.
            # Equal to zero, because those can then be used to pad, by changing the value zero to 1.
            possible_fake_entries = list(np.argwhere(self.occ_array[:, j] == 0).flatten())

            # We don't have enough documents. So we generate fake document IDs.
            if len(possible_fake_entries) < nr_fake_entries_to_add:

                # Create a new array that matches the column size and rows should be the remaining amount
                # that did not fit the original array using possible_fake_entries.
                fake_documents = np.zeros((nr_fake_entries_to_add - len(possible_fake_entries), ncol))

                # Concatenate and extend the original occurrence array.
                self.occ_array = np.concatenate((self.occ_array, fake_documents))

                # Get all indices in our new occurrence array that has a value of zero in the j'th column.
                # Because we can then choose from this list, which to change to 1 for padding.
                possible_fake_entries = list(np.argwhere(self.occ_array[:, j] == 0).flatten())

            # Choose the indices randomly and those will be changed to 1.
            fake_entries = random.sample(possible_fake_entries, nr_fake_entries_to_add)
            self.occ_array[fake_entries, j] = 1

        # Check if we had created any fake documents
        total_docs, _ = self.occ_array.shape
        total_fake_entries = total_docs - nrows

        # Edge case when we don't have enough documents for padding, thus we created fake documents.
        if total_fake_entries > 0:
            warnings.warn("Total_fake_entries was larger than 0")
            warnings.warn("We will create fake volumes...")
            list_fake_vols = random.choices(self.vol_array, k=total_fake_entries)
            self.vol_array.extend(list_fake_vols)

        # If we want to apply padding + volume hiding
        if padding_and_volume_hiding:
            VolumeHidingResultExtractor.apply_vol_hiding(self)

        # Re-create the inverted index, when we have a new occurrence array.
        self.gen_inv_index()

        print("PADDING APPLIED")

        #print("real nr entries: ", self._number_real_entries)
        #print("OVERHEAD", np.sum(self.occ_array) - self._number_real_entries)

    def __str__(self):
        return "Padded"


# Class that inherits the QueryResultExtractor and only adds an access pattern obfuscation countermeasure.
# Ref: G.Chen, T.Lai, M.K.Reiter and Y. Zhang. Differentially private access patterns for searchable
# symmetric encryption. 2018.

# The idea of the obfuscation is to use shards. From each document there are m shards created. And inside each shard
# the original keyword list is also there. And then on these shards there is obfuscation applied such that, some
# documents that do not contain the keyword are returned (false positives), and some documents are not returned that
# do contain the keyword (false negatives).
#
# When the client has grouped the shards from the same original document,
# he can reconstruct the original file when he has at least k out of m shards. For the purpose of this code, we are
# not interested in reconstructing. All we need to do is to apply obfuscation without reconstruction.
class ObfuscatedResultExtractor(QueryResultExtractor):
    # Initialize the obfuscator. The obfuscation parameters are those presented as optimal in the paper from Chen et al.
    def __init__(self, *args, m=6, p=0.88703, q=0.04416, **kwargs):
        self.occ_array = np.array([])
        super().__init__(*args, **kwargs)

        # p is the probability that a value 1 stays 1.
        self._p = p

        # q is the probability that a value 0, is flipped to 1.
        self._q = q

        # m is the amount of shards that we create for each document.
        self._m = m

        # Get amount of rows and columns from the original occurrence array (document to keyword).
        nrow, ncol = self.occ_array.shape

        # Duplicate each row in self.occ_array m times to create m shards.
        # Remember occ_array is a document to keyword matrix, documents as rows and keyword as columns.
        self.occ_array = np.repeat(self.occ_array, self._m, axis=0)

        # Now apply obfuscation
        for i in range(nrow):
            for j in range(ncol):
                # Document i contains keyword j, which means the value in the cell is 1.
                if self.occ_array[i, j]:
                    # Probability of 1-p for flipping value 1 to 0.
                    if random.random() < 1 - self._p:
                        self.occ_array[i, j] = 0
                else:
                    # Probability of q for flipping value 0 to 1.
                    if random.random() < self._q:
                        self.occ_array[i, j] = 1

    def __str__(self):
        return "Obfuscated"


# Class that inherits the QueryResultExtractor and adds the volume hiding counter measure.
# We are applying the naive approach.
# Ref: S. Kamara and T. Moatez . Computationally Volume-Hiding Structured Encryption 2019
class VolumeHidingResultExtractor(QueryResultExtractor):
    def __init__(self, *args, **kwargs):
        self.occ_array = np.array([])
        super().__init__(*args, **kwargs)
        VolumeHidingResultExtractor.apply_vol_hiding(self)

    @staticmethod
    def apply_vol_hiding(self):
        # All volumes of our documents
        all_volume_list = self.vol_array

        # Retrieve max volume
        max_vol = max(all_volume_list)

        # We are simulating as if we are padding the original document with spaces to fill up the volume to max_vol.
        # Here we just replace all volumes by the max volume to hide the volume.
        self.vol_array = [max_vol for _ in range(len(all_volume_list))]

        print("VOLUME HIDING APPLIED")
