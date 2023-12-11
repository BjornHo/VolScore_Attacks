# Implementation of the score attack as described in
# A Highly Accurate Query-Recovery Attack against Searchable Encryption using Non-Indexed Documents
# by Damie et al. 2021

import numpy as np
import pandas as pd
from numpy import log as ln


# We need to create a submatrix of query to query (trapdoor to trapdoor) from the original query to query matrix.
# The rows remain the same, however for the columns we are only interested in the queries that we know.
def gen_td_sub_matrix(known_queries, queries, co_occ_td, q_to_index):
    df = pd.DataFrame(index=queries, columns=known_queries)

    # For each query retrieve the co occ values from co_occ_td (the query to query co-occ matrix)
    for i, query in enumerate(queries):
        row = []

        # Idea is to build each row for the submatrix.
        # Get every value from a row in the original co-occ matrix from columns that are known queries
        for known_query in known_queries:
            co_occ_value = co_occ_td[i][q_to_index.get(known_query)]
            row.append(co_occ_value)

        # Set the row inside the dataframe
        df.loc[queries[i]] = row
    return df

# Same idea as gen_td_sub_matrix
def gen_kw_sub_matrix(known_kws, kws, co_occ_kw, k_to_index):
    df = pd.DataFrame(index=kws, columns=known_kws)
    for i, kw in enumerate(kws):
        row = []

        for known_kw in known_kws:
            co_occ_value = co_occ_kw[i][k_to_index.get(known_kw)]
            row.append(co_occ_value)

        df.loc[kws[i]] = row
    return df

def score_attack(k_sim, co_occ_kw, known_queries, queries, co_occ_td, k_to_index, q_to_index):

    # remove Q_ prefix
    known_kws = [kw[2:] for kw in known_queries]

    df_kw_sub = gen_kw_sub_matrix(known_kws, k_sim, co_occ_kw, k_to_index)
    df_td_sub = gen_td_sub_matrix(known_queries, queries, co_occ_td, q_to_index)

    # Predictions
    pred = []

    for td in queries:
        # Create dataframe with k_sim row, which will be the keywords, and as column the score of each keyword
        candidates = pd.DataFrame(index=k_sim, columns=['score'])
        for kw in k_sim:
            # Subtract the co-occ vector of a trapdoor with the co-occ vector of a keyword as diff
            diff = df_kw_sub.loc[kw, :].values - df_td_sub.loc[td, :].values

            # Norm 2, euclidean distance. Square of each element of the vector and sum, and then take the square root.
            s = -ln(np.sqrt(np.square(diff).sum()))

            # Set score for the current keyword as candidate
            candidates.loc[kw] = s

        # Sort candidates descending order
        candidates = candidates.sort_values(['score'], ascending=False)

        # Choose the candidate with the highest score
        highest_score_kw = candidates.iloc[0].name

        # Assign the highest scored keyword to the current trapdoor as prediction
        pred.append((td, highest_score_kw))

    return pred
