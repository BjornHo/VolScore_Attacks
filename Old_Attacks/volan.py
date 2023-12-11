
def volan_attack(q_set, kw_set, tvol_q, tvol_k):
    solution_map = []
    for q_index, query_vol in enumerate(tvol_q):
        keyword_vol_max = 0
        keyword_vol_max_index = -1
        for k_index, keyword_vol in enumerate(tvol_k):
            if query_vol >= keyword_vol > keyword_vol_max:
                keyword_vol_max = keyword_vol
                keyword_vol_max_index = k_index
        solution_map.append((q_set[q_index], kw_set[keyword_vol_max_index]))
    return solution_map
