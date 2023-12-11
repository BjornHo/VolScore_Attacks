import multiprocessing
import sys
import warnings
from functools import partial, reduce
from typing import List, Dict, Optional, Tuple

import numpy as np
from joblib._multiprocessing_helpers import mp
from tqdm import tqdm


# ScoreAttacker Class
# Keyword_occ_array np.array -- Keyword occurrence (row: similar documents; columns: keywords)
# Trapdoor_occ_array np.array -- Trapdoor occurrence (row: stored documents; columns: trapdoors)
# The documents are unknown (just the identifier has been seen by the attacker)
# Keyword_sorted_voc {List[str]} -- Keyword vocabulary extracted from similar documents.
# Known_queries {Dict[str, str]} -- Queries known by the attacker {trapdoor: keyword}.
# Trapdoor_sorted_voc {Optional[List[str]]} -- The trapdoor voc can be a sorted list of hashes
# to hide the underlying keywords.
# sim_inv_index -- dict {keyword: List[int]} -- Inverted index for the similar dataset. Keyword to document indices.
# real_inv_index  -- dict {keyword: List[int]} -- Inverted index for the real dataset. Keyword to document indices.
# sim_docs_vol_array -- volume array List[int] for similar documents. Each number indicates the volume of that document.
# real_docs_vol_array -- volume array List[int] for real documents.
#
# Keyword Arguments:
# norm_ord {int} -- Order of the norm used by the matchmaker (default: {2} which is Euclidean norm)
class ScoreAttacker:
    def __init__(
            self,
            keyword_occ_array,
            trapdoor_occ_array,
            keyword_sorted_voc: List[str],
            known_queries: Dict[str, str],
            trapdoor_sorted_voc: Optional[List[str]],
            sim_inv_index,
            real_inv_index,
            sim_docs_vol_array,
            real_docs_vol_array,
            norm_ord=2
    ):
        print("INIT SCORE ATTACKER")

        # Set the norm
        self.set_norm_ord(norm_ord=norm_ord)

        # Some error checking.
        if not known_queries:
            raise ValueError("Known queries are mandatory.")
        if len(known_queries.keys()) != len(set(known_queries.keys())):
            raise ValueError("Found duplicate trapdoors as known queries.")
        if len(known_queries.values()) != len(set(known_queries.values())):
            raise ValueError("Several trapdoors are linked to the same keyword.")

        # Dictionary known_queries {trapdoor: keyword}
        self._known_queries = known_queries.copy()

        # Store number of similar documents.
        # Obtained by getting the number of rows in keyword_occ_array (doc to keyword).
        self.number_similar_docs = keyword_occ_array.shape[0]

        # Keyword dictionary that stores the index which is the position where the keyword occurs
        # in the keyword_sorted_voc. And also the occurrence of a keyword across all documents.
        self.kw_voc_info = {
            word: {"vector_ind": ind, "word_occ": sum(keyword_occ_array[:, ind])}
            for ind, word in enumerate(keyword_sorted_voc)
        }

        # Same as previous, but now we do it for trapdoors.
        self.td_voc_info = {
            word: {"vector_ind": ind, "word_occ": sum(trapdoor_occ_array[:, ind])}
            for ind, word in enumerate(trapdoor_sorted_voc)
        }

        # Store the estimated number of real documents
        self.number_real_docs = self._estimate_nb_real_docs()

        # Store the frequency of each keyword inside kw_voc_info dictionary.
        # The frequency is the avg occurrence per document.
        for kw in self.kw_voc_info.keys():
            self.kw_voc_info[kw]["word_freq"] = (
                    self.kw_voc_info[kw]["word_occ"] / self.number_similar_docs
            )

        # Same as previous but now for trapdoors. We store the trapdoor frequency inside td_voc_info.
        for td in self.td_voc_info.keys():
            self.td_voc_info[td]["word_freq"] = (
                    self.td_voc_info[td]["word_occ"] / self.number_real_docs
            )

        # Computes 2 co-occurrence matrices.
        # The keyword co-occurrence matrix self.kw_coocc, and the trapdoor co-occurrence matrix self.td_coocc.
        self._compute_coocc_matrices(keyword_occ_array, trapdoor_occ_array)

        # Computes 2 reduced co-occurrence sub matrices.
        # The keyword co-occurrence sub matrix is self.kw_reduced_coocc
        # and the trapdoor co-occurrence sub matrix is self.td_reduced_coocc
        self._refresh_reduced_coocc()

        # Compute 2 co-vol matrices.
        # The keyword co-vol matrix self.kw_covol, and the trapdoor co-occurrence matrix self.td_covol
        self._compute_covol_matrices(sim_inv_index, sim_docs_vol_array, real_inv_index, real_docs_vol_array)

        # Computes 2 reduced co-vol sub matrices.
        # The keyword co-vol sub matrix is self.kw_reduced_covol
        # and the trapdoor co-vol sub matrix is self.td_reduced_covol
        self._refresh_reduced_covol()

        # Store dynamic refinement speeds for statistical reason. We can see what the average refinement speed is.
        self.dynamic_ref_speeds = []

        print("END INIT")

    # Set the order of the norm used to compute the scores.
    def set_norm_ord(self, norm_ord: int):
        self._norm = partial(np.linalg.norm, ord=norm_ord)

    # Estimates the number of documents stored.
    def _estimate_nb_real_docs(self):
        nb_doc_ratio_estimator = np.mean(
            [self.td_voc_info[td]["word_occ"] / self.kw_voc_info[kw]["word_occ"]
             for td, kw in self._known_queries.items()])
        return self.number_similar_docs * nb_doc_ratio_estimator

    # Computes the co-occurrence matrix by using the keyword and trapdoor occurrence arrays.
    def _compute_coocc_matrices(self, keyword_occ_array: np.array, trapdoor_occ_array: np.array):
        # Transpose of keyword_occ_array gives kw to document matrix.
        # The dot product between kw to doc and doc to keyword gives the co-occurrence matrix. We also divide
        # by number of similar documents to normalize.
        self.kw_coocc = (np.dot(keyword_occ_array.T, keyword_occ_array) / self.number_similar_docs)

        # Fill zero, we don't consider same keywords.
        np.fill_diagonal(self.kw_coocc, 0)

        # Same as keywords but now for trapdoors.
        self.td_coocc = (np.dot(trapdoor_occ_array.T, trapdoor_occ_array) / self.number_real_docs)

        # Fill zero again.
        np.fill_diagonal(self.td_coocc, 0)

    # Computes the covolume matrix.
    # sim_inv_index is a dictionary {keyword: [doc_indices]} for the similar document set.
    # real_inv_index is a dictionary {keyword: [doc_indices]} for the similar document set.
    def _compute_covol_matrices(self, sim_inv_index, sim_vol_array, real_inv_index, real_vol_array):

        # Build the keyword covolume.
        self.kw_covol = self.build_co_vol(list(self.kw_voc_info.keys()), sim_inv_index,
                                          sim_vol_array, self.number_similar_docs)

        # Fill zero on the diagonal.
        np.fill_diagonal(self.kw_covol, 0)

        # We need to know the trapdoors, and they have "enc_" as prefix.
        # So we remove it. We do this because the real_inv_index does not have the enc_ prefix, due to the order
        # of how the code is run.
        unenc_trapdoors = [td[4:] for td in list(self.td_voc_info.keys())]

        # Build trapdoor covolume.
        self.td_covol = self.build_co_vol(unenc_trapdoors, real_inv_index, real_vol_array, self.number_real_docs)

        # Fill zero again.
        np.fill_diagonal(self.td_covol, 0)

    # Refresh the co-occurrence matrix based on the known queries.
    # This function creates a sub matrix called kw_reduced_coocc and a submatrix called td_reduced_coocc.
    # In the end we can use the 2 sub matrices to compare. The columns have the same order and are known queries.
    # So by comparing the rows of the 2 sub matrices, we can determine if a keyword matches a trapdoor.
    def _refresh_reduced_coocc(self):
        # All indices of known keywords
        ind_known_kw = [self.kw_voc_info[kw]["vector_ind"] for kw in self._known_queries.values()]

        # Take a slice. We take a sub matrix from the original kw_cooc matrix where the rows and columns are keywords.
        # This means, we choose all rows and only choose columns of keywords that we know.
        self.kw_reduced_coocc = self.kw_coocc[:, ind_known_kw]

        # Similar to before, now we determine all trapdoor indices for slicing.
        ind_known_td = [self.td_voc_info[td]["vector_ind"] for td in self._known_queries.keys()]

        # The td_cooc matrix where rows and columns are trapdoors.
        # We select all rows, and only the columns of known trapdoors.
        self.td_reduced_coocc = self.td_coocc[:, ind_known_td]

    # Same as previous, but now we do it for refreshing the co-vol matrices.
    def _refresh_reduced_covol(self):
        # All indices of known keywords
        ind_known_kw = [self.kw_voc_info[kw]["vector_ind"] for kw in self._known_queries.values()]

        # Take a slice. We take a sub matrix from the original kw_covol matrix where the rows and columns are keywords.
        # This means, we choose all rows and only choose columns of keywords that we know.
        self.kw_reduced_covol = self.kw_covol[:, ind_known_kw]

        # Similar to before, now we determine all trapdoor indices for slicing.
        ind_known_td = [self.td_voc_info[td]["vector_ind"] for td in self._known_queries.keys()]

        # The td_covol matrix where rows and columns are trapdoors.
        # We select all rows, and only the columns of known trapdoors.
        self.td_reduced_covol = self.td_covol[:, ind_known_td]

    # Global variable to store co-vol results between processes.
    co_vol_results = []

    # Callback function used to store the results calculated inside a parallelized process into co_vol_results.
    def collect_co_vol_result(self, result):
        # Imagine creating a nxn keyword to keyword co-vol matrix. We want to split the rows between processes.
        # chunks is an array of row numbers allocated to this process.
        # For example [0 1 2 3 4]. Each process has a different chunk.
        # chunk_results contain all the results that should be inside this nxn matrix,
        # for the rows this process was responsible for.
        # matrix_size is simply the size of the matrix, and is the same between all processes.
        chunks, chunk_results, matrix_size = result
        global co_vol_results

        # Start with the first index of chunks.
        current_chunk = 0

        # j is the column of the matrix. We start with the first column.
        j = 0

        # Process each co_vol_value, but we also need to keep track when we need to move to the next row. Therefore,
        # we use the counter with the modulo and matrix_size.
        for counter, co_vol_value in enumerate(chunk_results):
            # Check if we reached the last value of the matrix in a row, so we need to move to the next row.
            if counter % matrix_size == 0 and counter != 0:
                # We move to the next chunk/next row.
                current_chunk += 1
                # We reset the column back to 0.
                j = 0
            # Store all the results in a global variable that can be accessed by each process.
            co_vol_results[chunks[current_chunk]][j] = co_vol_value

            # After storing the value, we move to the next column to process the next result
            j += 1

    # Create chunks to parallelize co-vol calculations
    # "a" is the list of row numbers to split.
    # "n" is the cpu amount to split the chunks among.
    def create_chunks(self, a, n):
        # Division mod len(a)/n and gives the quotient and the remainder.
        q, r = divmod(len(a), n)

        # A trick to distribute "a" between "n" processes. "a" can be divided "q" amount of times.
        # But we still have the remainder "r". We can increase the size of "r" chunks by 1 to distribute this.
        # This is done using min(i,r).
        # i+1 is for the boundary.
        # In the end we get chunks that are roughly equal in size.
        return (a[i * q + min(i, r):(i + 1) * q + min(i + 1, r)] for i in range(n))

    # Calculates the co-vol for keyword to keyword or query to query, depending on input.
    def calc_co_vol(self, chunks, matrix_size, kq_set, inv_index, vol_array, nr_docs):

        # Store the results for input chunks
        chunk_results = []
        for i in tqdm(chunks, total=len(chunks), desc="co_vol"):
            # keyword or query i
            keyword_i = kq_set[i]

            # The documents in which "i" occurs.
            kw_i_docs = inv_index[keyword_i]

            # Make it a set.
            kw_i_docs_set = set(kw_i_docs)

            # For all columns, we want to calculate each result in row i
            for j in range(matrix_size):
                # Keyword or query j
                keyword_j = kq_set[j]
                # Retrieve all documents that contain a keyword or query j
                kw_j_docs = inv_index[keyword_j]

                # Use set to determine the intersection of the two which contains the documents that matched
                # both keywords. We are interested in which documents are the same, as co_docs.
                # We retrieve the volume of each document, and sum all of them up.
                # And finally we divide by the total amount of documents to calculate co-volume.
                co_docs = kw_i_docs_set & set(kw_j_docs)

                if len(co_docs) > 0:
                    co_vol_result = sum([vol_array[doc] for doc in co_docs]) / nr_docs
                else:
                    co_vol_result = float(0)

                # Store the result
                chunk_results.append(co_vol_result)
        return chunks, chunk_results, matrix_size

    # Generates co-volume either keyword to keyword or query to query.
    # kq_set is the set of keywords or queries.
    # vol_array is an array that contains the volume of each document (ordered).
    # nr_docs, the amount of documents in the similar dataset or on the server dataset.
    def build_co_vol(self, kq_set, inv_index, vol_array, nr_docs):
        matrix_size = len(kq_set)

        # Used to store the results in one global variable that can be accessed by each process.
        global co_vol_results

        # Initialize each cell as 0
        co_vol_results = [[0 for x in range(matrix_size)] for y in range(matrix_size)]

        # Multiprocessing pool with all CPUs
        pool = mp.Pool(mp.cpu_count())

        # Create chunk list for each CPU. Each CPU gets a set of rows to calculate, which it is responsible for.
        chunk_list = list(self.create_chunks(range(matrix_size), mp.cpu_count()))

        # Calculate co-vol in parallel, and use callback to collect the results.
        for chunks in chunk_list:
            pool.apply_async(self.calc_co_vol,
                             args=(chunks, matrix_size, kq_set, inv_index, vol_array, nr_docs),
                             callback=self.collect_co_vol_result)
        pool.close()
        pool.join()
        return np.array(co_vol_results, dtype=np.float64)

    # The base score attack
    # It returns a dictionary {trapdoor: [keyword]}.
    # It only assigns one keyword to each trapdoor, so the length of the keyword list is 1.
    def predict(self, trapdoor_list: List[str]) -> Dict[str, List[str]]:

        # Multi processing
        nb_cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=nb_cores) as pool:
            pred_func = partial(self._sub_pred)
            results = pool.starmap(
                pred_func,
                enumerate([trapdoor_list[i::nb_cores] for i in range(nb_cores)]),
            )

            # Transforms nested list into a single prediction list
            pred_list = reduce(lambda x, y: x + y, results)

            # Save predictions. We do not use certainty for the base score attack even though it is returned.
            predictions = {td: kw for td, kw, _certainty in pred_list}
        return predictions

    # Sub function used for multiprocessing to make predictions for all score attacks using co-occurrence.
    # It returns (trapdoor, [keyword prediction], certainty)
    def _sub_pred(self, _ind: int, td_list: List[str], assigned_kw_list=None) -> List[Tuple[str, List[str], float]]:

        # Prediction, stores tuples (trapdoor, [keyword prediction], certainty)
        prediction = []
        for trapdoor in td_list:

            # Check if trapdoor is in our vocabulary
            try:
                trapdoor_ind = self.td_voc_info[trapdoor]["vector_ind"]
            except KeyError:
                warnings.warn("Unknown trapdoor: " + trapdoor)
                prediction.append((trapdoor, [""], 0))
                continue

            # Retrieve trapdoor vector (row inside td_reduced_coocc)
            trapdoor_vec = self.td_reduced_coocc[trapdoor_ind]

            # List to store (keyword, score) tuples.
            score_list = []

            # Computes the matching with each keyword of the vocabulary extracted from similar documents
            for keyword, kw_info in self.kw_voc_info.items():

                # We skip keywords that are already assigned (used during predict_with_cluster_refinement)
                if assigned_kw_list is not None and keyword in assigned_kw_list:
                    continue

                # The keyword vector, obtained from getting the row from self.kw_reduced_coocc.
                keyword_vec = self.kw_reduced_coocc[kw_info["vector_ind"]]

                # Calculate difference
                vec_diff = keyword_vec - trapdoor_vec

                # Distance between the keyword point and the trapdoor point in the known-queries sub-vector space
                td_kw_distance = self._norm(vec_diff)
                if td_kw_distance:
                    score = -np.log(td_kw_distance)

                # Perfect match when distance is zero
                else:
                    score = np.inf
                score_list.append((keyword, score))

            # Sort (keyword, score) list on score in descending order
            score_list.sort(key=lambda tup: tup[1], reverse=True)

            # Best candidate is the first item in the list which has the highest score
            best_candidate = score_list[0][0]

            # Certainty is difference between the first and the second item's score in the list.
            certainty = score_list[0][1] - score_list[1][1]

            # Add prediction
            prediction.append((trapdoor, [best_candidate], certainty))
        return prediction

    # Sub function used for multiprocessing to make predictions for score attack using co volume.
    # It returns (trapdoor, [keyword prediction], certainty)
    def _sub_pred_VOL(self, _ind: int, td_list: List[str], assigned_kw_list=None) -> List[Tuple[str, List[str], float]]:

        # Prediction, stores tuples (trapdoor, [keyword prediction], certainty)
        prediction = []
        for trapdoor in td_list:

            # Check if trapdoor is in our vocabulary
            try:
                trapdoor_ind = self.td_voc_info[trapdoor]["vector_ind"]
            except KeyError:
                warnings.warn("Unknown trapdoor: " + trapdoor)
                prediction.append((trapdoor, [""], 0))
                continue

            # Retrieve trapdoor vector (row inside td_reduced_covol)
            trapdoor_vec = self.td_reduced_covol[trapdoor_ind]

            # List to store (keyword, score) tuples.
            score_list = []

            # Computes the matching with each keyword of the vocabulary extracted from similar documents
            for keyword, kw_info in self.kw_voc_info.items():

                # We skip keywords that are already assigned (used during predict_with_cluster_refinement)
                if assigned_kw_list is not None and keyword in assigned_kw_list:
                    continue

                # The keyword vector, obtained from getting the row from self.kw_reduced_covol.
                keyword_vec = self.kw_reduced_covol[kw_info["vector_ind"]]
                vec_diff = keyword_vec - trapdoor_vec

                # Distance between the keyword point and the trapdoor point in the known-queries sub-vector space
                td_kw_distance = self._norm(vec_diff)
                if td_kw_distance:
                    score = -np.log(td_kw_distance)

                # Perfect match when distance is zero
                else:
                    score = np.inf
                score_list.append((keyword, score))

            # Sort (keyword, score) list on score in descending order
            score_list.sort(key=lambda tup: tup[1], reverse=True)

            # Best candidate is the first item in the list which has the highest score
            best_candidate = score_list[0][0]

            # Certainty is difference between the first and the second item's score in the list.
            certainty = score_list[0][1] - score_list[1][1]

            # Add prediction
            prediction.append((trapdoor, [best_candidate], certainty))
        return prediction

    # Refined score attack
    # The ref_speed indicates how many queries are added to known queries.
    # It returns a prediction dictionary {trapdoor: [keyword]} so that for each trapdoor
    # we have made a keyword prediction.
    def predict_with_refinement(self, trapdoor_list: List[str], ref_speed=0) -> Dict[str, List[str]]:

        # Default refinement speed: 5% of the total number of trapdoors
        if ref_speed < 1:
            ref_speed = int(0.05 * len(self.td_voc_info))

        # Store original known queries
        old_known = self._known_queries.copy()

        # Number of cores for multiprocessing
        nr_cores = multiprocessing.cpu_count()

        # List of unknown trapdoors
        unknown_td_list = list(trapdoor_list)

        # Dictionary {trapdoor: [keyword]} that stores the final results.
        final_results = []

        # Multiprocessing
        with multiprocessing.Pool(processes=nr_cores) as pool, tqdm(
                total=len(trapdoor_list), desc="Refining predictions"
        ) as pbar:
            while True:
                prev_td_nr = len(unknown_td_list)

                # Unknown trapdoors
                unknown_td_list = [td for td in unknown_td_list if td not in self._known_queries.keys()]

                # Removes the known trapdoors to update progress bar
                pbar.update(prev_td_nr - len(unknown_td_list))

                # Launch parallel predictions
                pred_func = partial(self._sub_pred)
                results = pool.starmap(
                    pred_func,
                    enumerate([unknown_td_list[i::nr_cores] for i in range(nr_cores)]),
                )

                # Transforms nested list into a single prediction list
                results = reduce(lambda x, y: x + y, results)

                # Sort the results
                results.sort(key=lambda tup: tup[2], reverse=True)

                # Exit condition, we stop refining.
                if len(unknown_td_list) < ref_speed:
                    final_results = [(td, candidates) for td, candidates, _sep in results]
                    break

                # Add the pseudo-known queries.
                new_known = {
                    td: candidates[0]
                    # Take a slice of the array, only the ref_speed amount is added to new_known
                    for td, candidates, _sep in results[:ref_speed]
                }
                #print("ADDED: ", new_known)

                self._known_queries.update(new_known)
                self._refresh_reduced_coocc()

        # Concatenate known queries and last results
        prediction = {td: [kw] for td, kw in self._known_queries.items() if td in trapdoor_list}
        prediction.update(dict(final_results))

        # Reset the known queries to the original known
        self._known_queries = old_known
        self._refresh_reduced_coocc()
        return prediction

    # VolScore Attack
    # We use the volume pattern as co-volume.
    def predict_with_refinement_VOL(self, trapdoor_list: List[str], ref_speed=0) -> Dict[str, List[str]]:

        # Default refinement speed: 5% of the total number of trapdoors
        if ref_speed < 1:
            ref_speed = int(0.05 * len(self.td_voc_info))

        # Store original known queries
        old_known = self._known_queries.copy()

        # Number of cores for multiprocessing
        nr_cores = multiprocessing.cpu_count()

        # List of unknown trapdoors
        unknown_td_list = list(trapdoor_list)

        # Dictionary {trapdoor: [keyword]} that stores the final results.
        final_results = []

        # Multiprocessing
        with multiprocessing.Pool(processes=nr_cores) as pool, tqdm(
                total=len(trapdoor_list), desc="Refining predictions"
        ) as pbar:
            while True:
                prev_td_nr = len(unknown_td_list)

                # Unknown trapdoors
                unknown_td_list = [td for td in unknown_td_list if td not in self._known_queries.keys()]

                # Removes the known trapdoors to update progress bar
                pbar.update(prev_td_nr - len(unknown_td_list))

                # Launch parallel predictions
                pred_func = partial(self._sub_pred_VOL)
                results = pool.starmap(
                    pred_func,
                    enumerate([unknown_td_list[i::nr_cores] for i in range(nr_cores)]),
                )

                # Transforms nested list into a single prediction list
                results = reduce(lambda x, y: x + y, results)

                # Sort the results
                results.sort(key=lambda tup: tup[2], reverse=True)

                # Exit condition, we stop refining.
                if len(unknown_td_list) < ref_speed:
                    final_results = [(td, candidates) for td, candidates, _sep in results]
                    break

                # Add the pseudo-known queries.
                new_known = {
                    td: candidates[0]
                    # Take a slice of the array, only the ref_speed amount is added to new_known
                    for td, candidates, _sep in results[:ref_speed]
                }
                #print("ADDED: ", new_known)

                self._known_queries.update(new_known)
                self._refresh_reduced_covol()

        # Concatenate known queries and last results
        prediction = {td: [kw] for td, kw in self._known_queries.items() if td in trapdoor_list}
        prediction.update(dict(final_results))

        # Reset the known queries to the original known
        self._known_queries = old_known
        self._refresh_reduced_covol()
        return prediction

    # From a list of sorted certainties, find the best index number for the maximum difference of certainties.
    # This is the index that gives the highest difference in score when compared to the next index.
    @staticmethod
    def index_max_diff(
            sorted_certainties: List[Tuple[str, List[str], float]],
            max_ref_speed=1,
    ) -> int:

        # Take a subset of all sorted certainties.
        # We add 1 element so that we can make a comparison for the last element.
        sorted_certainties = sorted_certainties[:(max_ref_speed + 1)]

        # This stores a list of differences of certainties between first and second element, second and third, etc.
        diff_list = [
            (i, score[2] - sorted_certainties[i + 1][2])
            # Loop through scores but without the additional element we added before.
            for i, score in enumerate(sorted_certainties[:-1])
        ]
        #print("DIFF LIST:", diff_list)

        # Get maximum diff and its index.
        if len(diff_list) > 0:
            ind_max_diff, maximum_diff = max(diff_list, key=lambda tup: tup[1])
            #print("MAX diff IS:", maximum_diff)
            #print("index max diff is:", ind_max_diff)
        else:
            ind_max_diff = 0
            #print("Final element")
            #print("index map diff is: ", ind_max_diff)

        return ind_max_diff

    # Make the prediction but use clustering with a max refinement speed to have a dynamic refinement speed.
    # Returns a dictionary {trapdoor: [keyword]} as prediction.
    def predict_with_cluster_refinement(self, trapdoor_list: List[str], max_ref_speed=0) -> Dict[str, List[str]]:

        # Default refinement speed: 5% of the total number of trapdoors
        if max_ref_speed < 1:
            max_ref_speed = int(0.05 * len(self.td_voc_info))

        # Store original known queries
        old_known = self._known_queries.copy()

        # Number of cores for multiprocessing
        nr_cores = multiprocessing.cpu_count()

        # List of unknown trapdoors
        unknown_td_list = list(trapdoor_list)

        # List of already assigned keywords
        assigned_kw_list = []

        # Add known keywords to assigned_kw_list
        for kw in self._known_queries.values():
            assigned_kw_list.append(kw)

        # Dictionary {trapdoor: [keyword]} that stores the final results.
        final_results = []

        # Multiprocessing
        with multiprocessing.Pool(processes=nr_cores) as pool, tqdm(
                total=len(trapdoor_list), desc="Cluster Refining predictions"
        ) as pbar:
            while True:
                prev_td_nb = len(unknown_td_list)

                # Unknown trapdoors
                unknown_td_list = [td for td in unknown_td_list if td not in self._known_queries.keys()]

                # Removes the known trapdoors to update progress bar
                pbar.update(prev_td_nb - len(unknown_td_list))

                # Launch parallel predictions
                pred_func = partial(self._sub_pred, assigned_kw_list=assigned_kw_list)
                results = pool.starmap(
                    pred_func,
                    enumerate([unknown_td_list[i::nr_cores] for i in range(nr_cores)]),
                )

                # Transforms nested list into a single prediction list
                results = reduce(lambda x, y: x + y, results)

                # Sort the results
                results.sort(key=lambda tup: tup[2], reverse=True)

                #print("Nr cluster candidates: ", results[:max_ref_speed])

                # We need to add 1 for array splicing. Otherwise, you get the wrong subset.
                # If the result is index 0, this means we have a refinement speed of 1.
                new_ref_speed = self.index_max_diff(results, max_ref_speed) + 1

                # Exit condition, we stop refining.
                if len(unknown_td_list) < max_ref_speed:
                    final_results = [(td, candidates) for td, candidates, _sep in results]
                    break

                # Store new refinement speed for statistical purposes. We do it here because we know we use it.
                # Otherwise, we do the exit condition, and still store the result...
                self.dynamic_ref_speeds.append(int(new_ref_speed))

                new_known = {}
                # Add chosen keywords to assigned_kw_list
                for td, candidates, _sep in results[:new_ref_speed]:
                    if candidates[0] not in assigned_kw_list:
                        new_known[td] = candidates[0]
                        assigned_kw_list.append(candidates[0])
                    else:
                        warnings.warn("Keyword: " + candidates[0] + "was already assigned, skipping")
                #print("ADDED: ", new_known)

                # Update known queries and co-occurrence sub matrices.
                self._known_queries.update(new_known)
                self._refresh_reduced_coocc()

        # Concatenate known queries and last results
        prediction = {
            td: [kw] for td, kw in self._known_queries.items() if td in trapdoor_list
        }
        prediction.update(dict(final_results))

        # Reset the known queries
        self._known_queries = old_known
        self._refresh_reduced_coocc()
        return prediction