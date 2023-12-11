import copy

# Initializes the known query mapping K that have unique result lengths.
# count_q consists of a list of result length counts for each query.
# count_k consists of a list of result length counts for each keyword.
# The index in count_q matches the index in the query list, so we also know which query it is.
# And also for count_k it matches the index in the keyword list.
#
# Basically in this function we create a known query mapping K = [(Query, Keyword), ...]
# We find queries that have a unique result length count. This means that there are no multiple
# keywords that share the same result length count in the keyword set. Since both keyword and query have
# the exact same result length count and are unique, we have found the correct match and map for a query to keyword.
def init_K(q_set, kw_set, count_q, count_k):
    # Store (Query, Keyword) pairs
    K = []
    # Loop through all result length counts of the queries
    for i in range(len(count_q)):
        # Keep track of the number of matches
        match_counts = 0
        # Initialize and this will store the matching index of the keyword
        match_index = -1
        # Loop through all result length counts of the keywords
        for j in range(len(count_k)):
            # Check if result length counts matches
            if count_q[i] == count_k[j]:
                match_counts += 1
                match_index = j
            # No need to look further, more than 1 match means it is not unique
            if match_counts > 1:
                break
        # Equal to 1 means unique result length count, so we found a match and a map of query to keyword.
        if match_counts == 1:
            K.append((q_set[i], kw_set[match_index]))
    return K


# Implementation of Count attack
# As described in Leakage-Abuse Attacks Against Searchable Encryption
#
# q_set is query set
# kw_set is keyword set
# count_k consists of a list of result length counts for each keyword.
# count_q consists of a list of result length counts for each query.
# c_q is the co-occurrence array for the queries.
# q_to_index is a map that maps a query to an index number (used for co-occ)
# c_i is the co-occurrence array for keywords.
# k_to_index is a map that maps a keyword to an index number (used for co-occ)
def count_attack(q_set, kw_set, count_k, count_q, c_q, q_to_index, c_i, k_to_index):
    # Initialize the current size
    current_size_k = 0

    # Store mapping of known (Query, Keyword) pairs in K. These are known based on the result length count.
    K = init_K(q_set, kw_set, count_q, count_k)

    # While we obtain new information of query to keyword pairs, we "scan" again. And hopefully find new pairs.
    while len(K) > current_size_k:
        current_size_k = len(K)

        # Known queries
        q_known = [q_i for (q_i, k_i) in K]

        # Unknown queries
        q_unknown = [q_i for q_i in q_set if q_i not in q_known]

        # For each unknown query, find candidate keywords. And ideally have 1 candidate at the end to obtain
        # a query to keyword mapping.
        for q_i in q_unknown:
            # Known keywords
            k_known = [k_i for (q_i, k_i) in K]

            # Unknown keywords
            k_unknown = [kw for kw in kw_set if kw not in k_known]

            # Candidate keywords are keywords that have the same result count as q_i and are unknown keywords
            candidate_keywords = [kw for kw in k_unknown if count_k[k_to_index.get(kw)] == count_q[q_to_index.get(q_i)]]

            # Make deepcopy to prepare for removal during iteration
            candidate_keywords_copy = copy.deepcopy(candidate_keywords)

            # For each candidate keyword, filter out keywords using co-occurrence
            for s in candidate_keywords_copy:

                # A known (query, keyword) pair in K
                for (q_prime, k) in K:
                    # Use known information of K and use co-occurrence to see if the mapping of
                    # (q_i, s) makes sense or not.
                    # It checks co-occurrence of (q_i, q_prime) with co-occurrence of (s, k).
                    # If it does not match, then it means the keyword is wrong, so we remove it.
                    if c_q[q_to_index.get(q_i)][q_to_index.get(q_prime)] != c_i[k_to_index.get(s)][k_to_index.get(k)]:
                        candidate_keywords.remove(s)
                        break

            # If there is only 1 candidate left, then we have found the correct (query,keyword) mapping
            if len(candidate_keywords) == 1:
                K.append((q_i, candidate_keywords[0]))
    return K
