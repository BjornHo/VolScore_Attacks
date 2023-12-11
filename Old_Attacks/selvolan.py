# Implementation of selective volume analysis as described in Revisiting Leakage Abuse Attacks (Blackstone et. al 2019)
# q_set is a set of queries.
# kw_set is a set of keywords.
# tvol_q is an array that contains the total volume observed from each query.
# tvol_k is an array that contains the total volume from each known keyword.
# count_q is an array that contains the observed count from each query.
# k_to_index is used when given a keyword, we want to know the index number of the keyword.
# count_k is an array that contains the counts of known keywords.
# d_known_factor is a factor from 0 to 1, the known data rate.
# epsilon is an error parameter.
def selvolan_attack(q_set, kw_set, tvol_q, tvol_k, count_q, k_to_index, count_k, d_known_factor, epsilon, theta):

    # Initialize empty candidate sets
    candidate_sets = [[] for i in range(len(q_set))]

    # Initialize to store query to keyword mappings
    solution_map = []

    # Loop through observed total volume of each query
    for i, v_i in enumerate(tvol_q):

        # Possible volumes in window range [d * v_i, v_i].
        # The idea is that we might only know a fraction of the data. Therefore, tvol_k which represents the total
        # volume of a keyword from the data we know, can be smaller than the observed volume v_i
        # when given the same keyword. Every keyword that is within this window could be a candidate keyword.
        possible_volumes = (v_index for v_index, v in enumerate(tvol_k) if d_known_factor * v_i <= v <= v_i)
        for v_index in possible_volumes:
            candidate_sets[i].append(kw_set[v_index])

        # No candidates in the window range, so we choose a keyword that is the closest d * v_i in volume.
        if len(candidate_sets[i]) == 0:
            keyword_vol_max = 0
            keyword_vol_max_index = -1
            for k_index, keyword_vol in enumerate(tvol_k):
                if keyword_vol_max < keyword_vol <= d_known_factor * v_i:
                    keyword_vol_max = keyword_vol
                    keyword_vol_max_index = k_index
            candidate_sets[i].append(kw_set[keyword_vol_max_index])

        # Filter the candidate keywords using count
        for w in candidate_sets[i]:
            # A linear error value used for the count estimation
            lambda_val = (1 - d_known_factor) * count_q[i] / epsilon

            # v_i - tvol_k, means observed volume minus volume that we know.
            # This results in the volume that we don't know. if we divide that by theta which is the avg volume
            # of a document, then we obtain an estimated document counts that we don't know.
            # So est_r_i is observed volume count minus count that we don't know. It basically maps the observed count
            # to a count number of our partial knowledge data count number.
            est_r_i = count_q[i] - (v_i - tvol_k[k_to_index.get(w)]) / theta

            # We use the estimated count for filtering candidate keywords.
            # Remove candidate keyword if the estimated count is a lambda away from the count of the current keyword,
            # or if the count of the keyword is larger (this should never happen unless we have the wrong keyword)
            if est_r_i - lambda_val > count_k[k_to_index.get(w)] or count_k[k_to_index.get(w)] > count_q[i]:
                candidate_sets[i].remove(w)

        # Choose the first candidate keyword in the candidate set as solution
        if len(candidate_sets[i]) > 0:
            solution_map.append((q_set[i], candidate_sets[i][0]))
        else:
            solution_map.append((q_set[i], None))
    return solution_map
