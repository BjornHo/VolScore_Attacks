import csv
import statistics
import sys
import time
import warnings

import nltk
import numpy as np

from keyword_extractor import split_df, KeywordExtractor
from parser import enron_parser, apache_parser, wiki_parser
from query_generator import QueryResultExtractor, generate_known_queries, PaddedResultExtractor, \
    VolumeHidingResultExtractor, ObfuscatedResultExtractor
from score_attacker import ScoreAttacker


# Prints out how many queries we could have possibly guessed correctly.
def max_possible_correct(similar_extractor, query_voc):
    counter = 0
    sorted_voc = similar_extractor.get_sorted_voc()
    for q in query_voc:
        if q in sorted_voc:
            counter += 1
    print("total q", len(query_voc), "max possible correct: ", counter)


# Determine the intersection of results from refined score and vol score.
# That will be our new known queries, and we will use that to extend our initial knowledge.
def get_new_known_queries(results_ref_score, results_vol_score, known_queries):
    # Retrieve the trapdoors
    tds1 = set(list(results_ref_score.keys()))
    tds2 = set(list(results_vol_score.keys()))
    intersect_tds = tds1.intersection(tds2)

    # Store new known queries
    new_known_queries = {}

    for td in intersect_tds:
        # Check if the keyword matches on the same trapdoor, and  if the keyword is not already assigned.
        if results_ref_score.get(td)[0] == results_vol_score.get(td)[0] and \
                results_ref_score.get(td)[0] not in new_known_queries.values() and \
                results_ref_score.get(td)[0] not in known_queries.values():
            print("FOUND MATCH")
            print("trapdoor: ", td)
            print("results_ref_score KW: ", results_ref_score.get(td)[0])
            print("results_vol_score KW: ", results_vol_score.get(td)[0])
            new_known_queries[td] = results_ref_score.get(td)[0]
    return new_known_queries


# Dataset name to nr
dataset_nr = {"enron": 1, "apache": 2, "wiki": 3}


# Modify when adding new dataset to parse
def parse_dataset(dataset_name):
    match dataset_nr.get(dataset_name):
        case 1:
            return enron_parser()
        case 2:
            return apache_parser()
        case 3:
            return wiki_parser()
        case _:
            warnings.warn("Invalid dataset")
            sys.exit()


def reproduce_original(dataset_name):
    df = parse_dataset(dataset_name)

    # Setup parameters
    sim_docs, server_docs = split_df(dframe=df, frac=0.4)
    sim_voc_size = 1200
    real_voc_size = 1000

    # Number of experiment runs
    nr_runs = 20

    nr_known_queries = [5, 10, 20, 40]

    total_runs = nr_runs * len(nr_known_queries)

    queryset_size = int(real_voc_size * 0.15)
    ref_speed = 10  # int(0.05 * queryset_size)

    with open("reproduce_original.csv", "w", newline="") as csv_file:
        fieldnames = [
            "Nr similar docs",
            "Nr server docs",
            "Similar voc size",
            "Server voc size",
            "Nr queries",
            "Nr queries known",
            "Base Score Acc",
            "Refined Score attack Acc"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        current_run = 0
        for current_nr_known_query in nr_known_queries:
            for i in range(nr_runs):
                current_run += 1
                print("RUN ", current_run, " / ", total_runs)

                # Extract the data
                similar_extractor = KeywordExtractor(sim_docs, sim_voc_size, 1)
                real_extractor = QueryResultExtractor(server_docs, real_voc_size, 1)

                # Create queries
                query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

                # Known queries dictionary {keyword: keyword}
                known_queries = generate_known_queries(
                    similar_wordlist=similar_extractor.get_sorted_voc(),
                    stored_wordlist=query_voc,
                    nr_queries=current_nr_known_query,
                )

                # Trapdoor vocabulary
                td_voc = []

                # Temporary dictionary to replace known queries. Known queries only has (keyword, keyword) pairs. Since
                # the trapdoors were not encrypted yet.
                temp_known = {}

                # Dictionary {trapdoor: keyword}
                eval_dico = {}

                # Do the fake encryption for trapdoors
                for keyword in query_voc:
                    fake_trapdoor = "enc_" + keyword
                    td_voc.append(fake_trapdoor)

                    # If it is a known query, we need to save which keyword corresponds to which trapdoor.
                    if known_queries.get(keyword):
                        temp_known[fake_trapdoor] = keyword

                    # Save inside {trapdoor: keyword} for evaluation.
                    eval_dico[fake_trapdoor] = keyword

                # Replace known queries
                known_queries = temp_known

                attacker = ScoreAttacker(
                    keyword_occ_array=similar_extractor.occ_array,
                    keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                    trapdoor_occ_array=query_array,
                    trapdoor_sorted_voc=td_voc,
                    known_queries=known_queries,
                    sim_inv_index=similar_extractor.inv_index,
                    real_inv_index=real_extractor.inv_index,
                    sim_docs_vol_array=similar_extractor.vol_array,
                    real_docs_vol_array=real_extractor.vol_array
                )

                # Trapdoor list are trapdoors that are not known.
                td_list = list(set(eval_dico.keys()).difference(attacker._known_queries.keys()))

                # Base Score
                results_base_score = attacker.predict(td_list)
                base_score_acc = np.mean(
                    [eval_dico[td] == candidates[0] for td, candidates in results_base_score.items()])

                # Refined score
                results_ref_score = attacker.predict_with_refinement(td_list, ref_speed=ref_speed)
                ref_score_acc = np.mean(
                    [eval_dico[td] == candidates[0] for td, candidates in results_ref_score.items()])

                writer.writerow(
                    {
                        "Nr similar docs": sim_docs.shape[0],
                        "Nr server docs": server_docs.shape[0],
                        "Similar voc size": sim_voc_size,
                        "Server voc size": real_voc_size,
                        "Nr queries": queryset_size,
                        "Nr queries known": current_nr_known_query,
                        "Base Score Acc": base_score_acc,
                        "Refined Score attack Acc": ref_score_acc
                    }
                )
                csv_file.flush()


# refScore, volScore, RefVolScore, ClusterVolScore 5, 10, 20 knownQ. |Q|=150, Refspeed = 10 = maxrefspeed
def comparison_general(dataset_name):
    df = parse_dataset(dataset_name)

    # Setup parameters
    sim_docs, server_docs = split_df(dframe=df, frac=0.4)
    sim_voc_size = 1200
    real_voc_size = 1000

    # Number of experiment runs
    nr_runs = 20

    nr_known_queries = [5, 10, 20]

    total_runs = nr_runs * len(nr_known_queries)

    queryset_size = int(real_voc_size * 0.15)
    ref_speed = 10  # int(0.05 * queryset_size)

    with open("comparison_general_" + dataset_name + ".csv", "w", newline="") as csv_file:
        fieldnames = [
            "Nr similar docs",
            "Nr server docs",
            "Similar voc size",
            "Server voc size",
            "Nr queries",
            "Nr queries known",
            "Refined Score attack Acc",
            "Vol Score Acc",
            "Ref Vol Score Acc",
            "Cluster Vol Score Acc"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        current_run = 0
        for current_nr_known_query in nr_known_queries:
            for i in range(nr_runs):
                current_run += 1
                print("RUN ", current_run, " / ", total_runs)

                # Extract the data
                similar_extractor = KeywordExtractor(sim_docs, sim_voc_size, 1)
                real_extractor = QueryResultExtractor(server_docs, real_voc_size, 1)
                # real_extractor = PaddedResultExtractor(server_docs, real_voc_size, 1, n=500, volume_hiding=True)
                # real_extractor = VolumeHidingResultExtractor(server_docs, real_voc_size, 1)

                # Create queries
                query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

                # Known queries dictionary {keyword: keyword}
                known_queries = generate_known_queries(
                    similar_wordlist=similar_extractor.get_sorted_voc(),
                    stored_wordlist=query_voc,
                    nr_queries=current_nr_known_query,
                )

                # Trapdoor vocabulary
                td_voc = []

                # Temporary dictionary to replace known queries. Known queries only has (keyword, keyword) pairs. Since
                # the trapdoors were not encrypted yet.
                temp_known = {}

                # Dictionary {trapdoor: keyword}
                eval_dico = {}

                # Do the fake encryption for trapdoors
                for keyword in query_voc:
                    fake_trapdoor = "enc_" + keyword
                    td_voc.append(fake_trapdoor)

                    # If it is a known query, we need to save which keyword corresponds to which trapdoor.
                    if known_queries.get(keyword):
                        temp_known[fake_trapdoor] = keyword

                    # Save inside {trapdoor: keyword} for evaluation.
                    eval_dico[fake_trapdoor] = keyword

                # Replace known queries
                known_queries = temp_known

                attacker = ScoreAttacker(
                    keyword_occ_array=similar_extractor.occ_array,
                    keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                    trapdoor_occ_array=query_array,
                    trapdoor_sorted_voc=td_voc,
                    known_queries=known_queries,
                    sim_inv_index=similar_extractor.inv_index,
                    real_inv_index=real_extractor.inv_index,
                    sim_docs_vol_array=similar_extractor.vol_array,
                    real_docs_vol_array=real_extractor.vol_array
                )

                # Trapdoor list are trapdoors that are not known.
                td_list = list(set(eval_dico.keys()).difference(attacker._known_queries.keys()))

                # Refined score
                results_ref_score = attacker.predict_with_refinement(td_list, ref_speed=ref_speed)
                ref_score_acc = np.mean(
                    [eval_dico[td] == candidates[0] for td, candidates in results_ref_score.items()])

                # VolScore
                results_vol_score = attacker.predict_with_refinement_VOL(td_list, ref_speed=ref_speed)
                vol_score_acc = np.mean(
                    [eval_dico[td] == candidates[0] for td, candidates in results_vol_score.items()])

                # Retrieve new known queries
                new_known_queries = get_new_known_queries(results_ref_score, results_vol_score, known_queries)
                print("KNOWN:", known_queries)
                known_queries.update(new_known_queries)
                print("NEW KNOWN", known_queries)

                # Init new score attack with new known queries.
                attacker = ScoreAttacker(
                    keyword_occ_array=similar_extractor.occ_array,
                    keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                    trapdoor_occ_array=query_array,
                    trapdoor_sorted_voc=td_voc,
                    known_queries=known_queries,
                    sim_inv_index=similar_extractor.inv_index,
                    real_inv_index=real_extractor.inv_index,
                    sim_docs_vol_array=similar_extractor.vol_array,
                    real_docs_vol_array=real_extractor.vol_array
                )

                # Refined score (aka ref Vol Score) with new known queries
                results_ref_vol_score = attacker.predict_with_refinement(td_list, ref_speed=ref_speed)
                ref_vol_score_acc = np.mean(
                    [eval_dico[td] == candidates[0] for td, candidates in results_ref_vol_score.items()])

                # Cluster Vol Score
                results_cluster_vol_score = attacker.predict_with_cluster_refinement(td_list, max_ref_speed=10)
                cluster_vol_score_acc = np.mean(
                    [eval_dico[td] == candidates[0] for td, candidates in results_cluster_vol_score.items()])

                print("ref_score_acc", ref_score_acc)
                print("vol_score_acc", vol_score_acc)
                print("ref_vol_score_acc", ref_vol_score_acc)
                print("cluster_vol_score_acc", cluster_vol_score_acc)

                writer.writerow(
                    {
                        "Nr similar docs": sim_docs.shape[0],
                        "Nr server docs": server_docs.shape[0],
                        "Similar voc size": sim_voc_size,
                        "Server voc size": real_voc_size,
                        "Nr queries": queryset_size,
                        "Nr queries known": current_nr_known_query,
                        "Refined Score attack Acc": ref_score_acc,
                        "Vol Score Acc": vol_score_acc,
                        "Ref Vol Score Acc": ref_vol_score_acc,
                        "Cluster Vol Score Acc": cluster_vol_score_acc
                    }
                )
                csv_file.flush()


# refScore, volScore, RefVolScore, ClusterVolScore 2, 3, 4 knownQ. |Q|=150, Refspeed = 10 = maxrefspeed
def comparison_low_knownq(dataset_name):
    df = parse_dataset(dataset_name)

    # Setup parameters
    sim_docs, server_docs = split_df(dframe=df, frac=0.4)
    sim_voc_size = 1200
    real_voc_size = 1000

    # Number of experiment runs
    nr_runs = 20

    nr_known_queries = [2, 3, 4]

    total_runs = nr_runs * len(nr_known_queries)

    queryset_size = int(real_voc_size * 0.15)
    ref_speed = 10  # int(0.05 * queryset_size)

    with open("comparison_low_knownq_" + dataset_name + ".csv", "w", newline="") as csv_file:
        fieldnames = [
            "Nr similar docs",
            "Nr server docs",
            "Similar voc size",
            "Server voc size",
            "Nr queries",
            "Nr queries known",
            "Nr new known queries",
            "Combined known queries acc",
            "Refined Score attack Acc",
            "Vol Score Acc",
            "Ref Vol Score Acc",
            "Cluster Vol Score Acc"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        current_run = 0
        for current_nr_known_query in nr_known_queries:
            for i in range(nr_runs):
                current_run += 1
                print("RUN ", current_run, " / ", total_runs)

                # Extract the data
                similar_extractor = KeywordExtractor(sim_docs, sim_voc_size, 1)
                real_extractor = QueryResultExtractor(server_docs, real_voc_size, 1)
                # real_extractor = PaddedResultExtractor(server_docs, real_voc_size, 1, n=500, volume_hiding=True)
                # real_extractor = VolumeHidingResultExtractor(server_docs, real_voc_size, 1)

                # Create queries
                query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

                # Known queries dictionary {keyword: keyword}
                known_queries = generate_known_queries(
                    similar_wordlist=similar_extractor.get_sorted_voc(),
                    stored_wordlist=query_voc,
                    nr_queries=current_nr_known_query,
                )

                # Trapdoor vocabulary
                td_voc = []

                # Temporary dictionary to replace known queries. Known queries only has (keyword, keyword) pairs. Since
                # the trapdoors were not encrypted yet.
                temp_known = {}

                # Dictionary {trapdoor: keyword}
                eval_dico = {}

                # Do the fake encryption for trapdoors
                for keyword in query_voc:
                    fake_trapdoor = "enc_" + keyword
                    td_voc.append(fake_trapdoor)

                    # If it is a known query, we need to save which keyword corresponds to which trapdoor.
                    if known_queries.get(keyword):
                        temp_known[fake_trapdoor] = keyword

                    # Save inside {trapdoor: keyword} for evaluation.
                    eval_dico[fake_trapdoor] = keyword

                # Replace known queries
                known_queries = temp_known

                attacker = ScoreAttacker(
                    keyword_occ_array=similar_extractor.occ_array,
                    keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                    trapdoor_occ_array=query_array,
                    trapdoor_sorted_voc=td_voc,
                    known_queries=known_queries,
                    sim_inv_index=similar_extractor.inv_index,
                    real_inv_index=real_extractor.inv_index,
                    sim_docs_vol_array=similar_extractor.vol_array,
                    real_docs_vol_array=real_extractor.vol_array
                )

                # Trapdoor list are trapdoors that are not known.
                td_list = list(set(eval_dico.keys()).difference(attacker._known_queries.keys()))

                # Refined score
                results_ref_score = attacker.predict_with_refinement(td_list, ref_speed=ref_speed)
                ref_score_acc = np.mean(
                    [eval_dico[td] == candidates[0] for td, candidates in results_ref_score.items()])

                # VolScore
                results_vol_score = attacker.predict_with_refinement_VOL(td_list, ref_speed=ref_speed)
                vol_score_acc = np.mean(
                    [eval_dico[td] == candidates[0] for td, candidates in results_vol_score.items()])

                # Retrieve new known queries
                new_known_queries = get_new_known_queries(results_ref_score, results_vol_score, known_queries)
                print("KNOWN:", known_queries)
                known_queries.update(new_known_queries)
                print("NEW KNOWN", known_queries)

                # Calculate combined known query accuracy for statistical purposes.
                combined_knownq_acc = np.mean([eval_dico[td] == kw for td, kw in known_queries.items()])

                # Init new score attack with new known queries.
                attacker = ScoreAttacker(
                    keyword_occ_array=similar_extractor.occ_array,
                    keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                    trapdoor_occ_array=query_array,
                    trapdoor_sorted_voc=td_voc,
                    known_queries=known_queries,
                    sim_inv_index=similar_extractor.inv_index,
                    real_inv_index=real_extractor.inv_index,
                    sim_docs_vol_array=similar_extractor.vol_array,
                    real_docs_vol_array=real_extractor.vol_array
                )

                # Refined score (aka ref Vol Score) with new known queries
                results_ref_vol_score = attacker.predict_with_refinement(td_list, ref_speed=ref_speed)
                ref_vol_score_acc = np.mean(
                    [eval_dico[td] == candidates[0] for td, candidates in results_ref_vol_score.items()])

                # Cluster Vol Score
                results_cluster_vol_score = attacker.predict_with_cluster_refinement(td_list, max_ref_speed=10)
                cluster_vol_score_acc = np.mean(
                    [eval_dico[td] == candidates[0] for td, candidates in results_cluster_vol_score.items()])

                print("ref_score_acc", ref_score_acc)
                print("vol_score_acc", vol_score_acc)
                print("ref_vol_score_acc", ref_vol_score_acc)
                print("cluster_vol_score_acc", cluster_vol_score_acc)

                writer.writerow(
                    {
                        "Nr similar docs": sim_docs.shape[0],
                        "Nr server docs": server_docs.shape[0],
                        "Similar voc size": sim_voc_size,
                        "Server voc size": real_voc_size,
                        "Nr queries": queryset_size,
                        "Nr queries known": current_nr_known_query,
                        "Nr new known queries": len(new_known_queries),
                        "Combined known queries acc": combined_knownq_acc,
                        "Refined Score attack Acc": ref_score_acc,
                        "Vol Score Acc": vol_score_acc,
                        "Ref Vol Score Acc": ref_vol_score_acc,
                        "Cluster Vol Score Acc": cluster_vol_score_acc
                    }
                )
                csv_file.flush()


# refScore, volScore, RefVolScore, ClusterVolScore 4 knownQ. |Q|=150, Refspeed = 2, 4, 8  maxrefspeed = 10
def comparison_refspeed(dataset_name):
    df = parse_dataset(dataset_name)

    # Setup parameters
    sim_docs, server_docs = split_df(dframe=df, frac=0.4)
    sim_voc_size = 1200
    real_voc_size = 1000

    # Number of experiment runs
    nr_runs = 20

    nr_known_queries = [4]

    total_runs = nr_runs * len(nr_known_queries)

    queryset_size = int(real_voc_size * 0.15)
    ref_speed = [2, 4, 8]  # int(0.05 * queryset_size)

    with open("comparison_refspeed_" + dataset_name + ".csv", "w", newline="") as csv_file:
        fieldnames = [
            "Nr similar docs",
            "Nr server docs",
            "Similar voc size",
            "Server voc size",
            "Nr queries",
            "Nr queries known",
            "Nr new known queries",
            "Combined known queries acc",
            "Current refspeed",
            "Dynamic refspeeds",
            "Runtime RefVolScore",
            "Runtime ClusterVolScore",
            "Refined Score attack Acc",
            "Vol Score Acc",
            "Ref Vol Score Acc",
            "Cluster Vol Score Acc"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        current_run = 0
        for current_nr_known_query in nr_known_queries:
            for current_refspeed in ref_speed:
                for i in range(nr_runs):
                    current_run += 1
                    print("RUN ", current_run, " / ", total_runs)

                    # Extract the data
                    similar_extractor = KeywordExtractor(sim_docs, sim_voc_size, 1)
                    real_extractor = QueryResultExtractor(server_docs, real_voc_size, 1)
                    # real_extractor = PaddedResultExtractor(server_docs, real_voc_size, 1, n=500, volume_hiding=True)
                    # real_extractor = VolumeHidingResultExtractor(server_docs, real_voc_size, 1)

                    # Create queries
                    query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

                    # Known queries dictionary {keyword: keyword}
                    known_queries = generate_known_queries(
                        similar_wordlist=similar_extractor.get_sorted_voc(),
                        stored_wordlist=query_voc,
                        nr_queries=current_nr_known_query,
                    )

                    # Trapdoor vocabulary
                    td_voc = []

                    # Temporary dictionary to replace known queries. Known queries only has (keyword, keyword) pairs. Since
                    # the trapdoors were not encrypted yet.
                    temp_known = {}

                    # Dictionary {trapdoor: keyword}
                    eval_dico = {}

                    # Do the fake encryption for trapdoors
                    for keyword in query_voc:
                        fake_trapdoor = "enc_" + keyword
                        td_voc.append(fake_trapdoor)

                        # If it is a known query, we need to save which keyword corresponds to which trapdoor.
                        if known_queries.get(keyword):
                            temp_known[fake_trapdoor] = keyword

                        # Save inside {trapdoor: keyword} for evaluation.
                        eval_dico[fake_trapdoor] = keyword

                    # Replace known queries
                    known_queries = temp_known

                    attacker = ScoreAttacker(
                        keyword_occ_array=similar_extractor.occ_array,
                        keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                        trapdoor_occ_array=query_array,
                        trapdoor_sorted_voc=td_voc,
                        known_queries=known_queries,
                        sim_inv_index=similar_extractor.inv_index,
                        real_inv_index=real_extractor.inv_index,
                        sim_docs_vol_array=similar_extractor.vol_array,
                        real_docs_vol_array=real_extractor.vol_array
                    )

                    # Trapdoor list are trapdoors that are not known.
                    td_list = list(set(eval_dico.keys()).difference(attacker._known_queries.keys()))

                    # Refined score
                    results_ref_score = attacker.predict_with_refinement(td_list, ref_speed=current_refspeed)
                    ref_score_acc = np.mean(
                        [eval_dico[td] == candidates[0] for td, candidates in results_ref_score.items()])

                    # VolScore
                    results_vol_score = attacker.predict_with_refinement_VOL(td_list, ref_speed=current_refspeed)
                    vol_score_acc = np.mean(
                        [eval_dico[td] == candidates[0] for td, candidates in results_vol_score.items()])

                    # Retrieve new known queries
                    new_known_queries = get_new_known_queries(results_ref_score, results_vol_score, known_queries)
                    print("KNOWN:", known_queries)
                    known_queries.update(new_known_queries)
                    print("NEW KNOWN", known_queries)

                    # Calculate combined known query accuracy for statistical purposes.
                    combined_knownq_acc = np.mean([eval_dico[td] == kw for td, kw in known_queries.items()])
                    print("Combined knownQ acc ", combined_knownq_acc)

                    # Init new score attack with new known queries.
                    attacker = ScoreAttacker(
                        keyword_occ_array=similar_extractor.occ_array,
                        keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                        trapdoor_occ_array=query_array,
                        trapdoor_sorted_voc=td_voc,
                        known_queries=known_queries,
                        sim_inv_index=similar_extractor.inv_index,
                        real_inv_index=real_extractor.inv_index,
                        sim_docs_vol_array=similar_extractor.vol_array,
                        real_docs_vol_array=real_extractor.vol_array
                    )

                    # Refined score (aka ref Vol Score) with new known queries
                    start_time = time.time()
                    results_ref_vol_score = attacker.predict_with_refinement(td_list, ref_speed=current_refspeed)
                    runtime_ref_vol_score = time.time() - start_time
                    ref_vol_score_acc = np.mean(
                        [eval_dico[td] == candidates[0] for td, candidates in results_ref_vol_score.items()])

                    # Cluster Vol Score
                    start_time = time.time()
                    results_cluster_vol_score = attacker.predict_with_cluster_refinement(td_list, max_ref_speed=10)
                    runtime_cluster_vol_score = time.time() - start_time
                    cluster_vol_score_acc = np.mean(
                        [eval_dico[td] == candidates[0] for td, candidates in results_cluster_vol_score.items()])

                    print("ref_score_acc", ref_score_acc)
                    print("vol_score_acc", vol_score_acc)
                    print("ref_vol_score_acc", ref_vol_score_acc)
                    print("cluster_vol_score_acc", cluster_vol_score_acc)

                    writer.writerow(
                        {
                            "Nr similar docs": sim_docs.shape[0],
                            "Nr server docs": server_docs.shape[0],
                            "Similar voc size": sim_voc_size,
                            "Server voc size": real_voc_size,
                            "Nr queries": queryset_size,
                            "Nr queries known": current_nr_known_query,
                            "Nr new known queries": len(new_known_queries),
                            "Combined known queries acc": combined_knownq_acc,
                            "Current refspeed": current_refspeed,
                            "Dynamic refspeeds": attacker.dynamic_ref_speeds,
                            "Runtime RefVolScore": runtime_ref_vol_score,
                            "Runtime ClusterVolScore": runtime_cluster_vol_score,
                            "Refined Score attack Acc": ref_score_acc,
                            "Vol Score Acc": vol_score_acc,
                            "Ref Vol Score Acc": ref_vol_score_acc,
                            "Cluster Vol Score Acc": cluster_vol_score_acc
                        }
                    )
                    csv_file.flush()


# refScore, volScore, RefVolScore
# sim_voc_size = real_voc_size
# voc_sizes = 500, 1000, 2000
# 15 knownQ. |Q|=0.15 * real_voc_size, Refspeed = 0.05 * queryset_size,  npad= 500
def comparison_countermeasures(dataset_name):
    df = parse_dataset(dataset_name)

    # Setup parameters
    sim_docs, server_docs = split_df(dframe=df, frac=0.4)

    # No distinction, real and similar dataset now have same vocabulary size
    voc_sizes = [500, 1000, 2000]
    padding_size = 500

    # Number of experiment runs
    nr_runs = 20

    nr_known_queries = 15

    # Maps string to countermeasure function
    counter_measures = {'none': QueryResultExtractor, 'volume_hiding': VolumeHidingResultExtractor,
                        'padding': PaddedResultExtractor}

    total_runs = nr_runs * len(voc_sizes) * len(counter_measures)

    with open("comparison_countermeasures_" + dataset_name + ".csv", "w", newline="") as csv_file:
        fieldnames = [
            "Similar voc size",
            "Server voc size",
            "Counter measure",
            "Nr similar docs",
            "Nr server docs",
            "Nr queries",
            "Nr queries known",
            "Nr new known queries",
            "Combined known queries acc",
            "Current refspeed",
            "Refined Score attack Acc",
            "Vol Score Acc",
            "Ref Vol Score Acc",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        current_run = 0
        for current_counter_measure in counter_measures.keys():
            for current_voc_size in voc_sizes:
                queryset_size = int(current_voc_size * 0.15)
                current_refspeed = int(0.05 * queryset_size)

                for i in range(nr_runs):
                    current_run += 1
                    print("RUN ", current_run, " / ", total_runs)

                    # Extract the data
                    similar_extractor = KeywordExtractor(sim_docs, current_voc_size, 1)
                    real_extractor = counter_measures.get(current_counter_measure)(server_docs, current_voc_size,
                                                                                   1, n_pad=padding_size,
                                                                                   padding_and_volume_hiding=False)

                    # real_extractor = QueryResultExtractor(server_docs, real_voc_size, 1)
                    # real_extractor = PaddedResultExtractor(server_docs, real_voc_size, 1, n_pad=padding_size, padding_and_volume_hiding=False)
                    # real_extractor = VolumeHidingResultExtractor(server_docs, real_voc_size, 1)

                    # Create queries
                    query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

                    # Known queries dictionary {keyword: keyword}
                    known_queries = generate_known_queries(
                        similar_wordlist=similar_extractor.get_sorted_voc(),
                        stored_wordlist=query_voc,
                        nr_queries=nr_known_queries,
                    )

                    # Trapdoor vocabulary
                    td_voc = []

                    # Temporary dictionary to replace known queries. Known queries only has (keyword, keyword) pairs. Since
                    # the trapdoors were not encrypted yet.
                    temp_known = {}

                    # Dictionary {trapdoor: keyword}
                    eval_dico = {}

                    # Do the fake encryption for trapdoors
                    for keyword in query_voc:
                        fake_trapdoor = "enc_" + keyword
                        td_voc.append(fake_trapdoor)

                        # If it is a known query, we need to save which keyword corresponds to which trapdoor.
                        if known_queries.get(keyword):
                            temp_known[fake_trapdoor] = keyword

                        # Save inside {trapdoor: keyword} for evaluation.
                        eval_dico[fake_trapdoor] = keyword

                    # Replace known queries
                    known_queries = temp_known

                    attacker = ScoreAttacker(
                        keyword_occ_array=similar_extractor.occ_array,
                        keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                        trapdoor_occ_array=query_array,
                        trapdoor_sorted_voc=td_voc,
                        known_queries=known_queries,
                        sim_inv_index=similar_extractor.inv_index,
                        real_inv_index=real_extractor.inv_index,
                        sim_docs_vol_array=similar_extractor.vol_array,
                        real_docs_vol_array=real_extractor.vol_array
                    )

                    # Trapdoor list are trapdoors that are not known.
                    td_list = list(set(eval_dico.keys()).difference(attacker._known_queries.keys()))

                    # Refined score
                    results_ref_score = attacker.predict_with_refinement(td_list, ref_speed=current_refspeed)
                    ref_score_acc = np.mean(
                        [eval_dico[td] == candidates[0] for td, candidates in results_ref_score.items()])

                    # VolScore
                    results_vol_score = attacker.predict_with_refinement_VOL(td_list, ref_speed=current_refspeed)
                    vol_score_acc = np.mean(
                        [eval_dico[td] == candidates[0] for td, candidates in results_vol_score.items()])

                    # Retrieve new known queries
                    new_known_queries = get_new_known_queries(results_ref_score, results_vol_score, known_queries)
                    print("KNOWN:", known_queries)
                    known_queries.update(new_known_queries)
                    print("NEW KNOWN", known_queries)

                    # Calculate combined known query accuracy for statistical purposes.
                    combined_knownq_acc = np.mean([eval_dico[td] == kw for td, kw in known_queries.items()])
                    print("Combined knownQ acc ", combined_knownq_acc)

                    # Init new score attack with new known queries.
                    attacker = ScoreAttacker(
                        keyword_occ_array=similar_extractor.occ_array,
                        keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                        trapdoor_occ_array=query_array,
                        trapdoor_sorted_voc=td_voc,
                        known_queries=known_queries,
                        sim_inv_index=similar_extractor.inv_index,
                        real_inv_index=real_extractor.inv_index,
                        sim_docs_vol_array=similar_extractor.vol_array,
                        real_docs_vol_array=real_extractor.vol_array
                    )

                    # Refined score (aka ref Vol Score) with new known queries
                    results_ref_vol_score = attacker.predict_with_refinement(td_list, ref_speed=current_refspeed)
                    ref_vol_score_acc = np.mean(
                        [eval_dico[td] == candidates[0] for td, candidates in results_ref_vol_score.items()])

                    print("ref_score_acc", ref_score_acc)
                    print("vol_score_acc", vol_score_acc)
                    print("ref_vol_score_acc", ref_vol_score_acc)
                    writer.writerow(
                        {
                            "Nr similar docs": sim_docs.shape[0],
                            "Nr server docs": server_docs.shape[0],
                            "Counter measure": current_counter_measure,
                            "Similar voc size": current_voc_size,
                            "Server voc size": current_voc_size,
                            "Nr queries": queryset_size,
                            "Nr queries known": nr_known_queries,
                            "Nr new known queries": len(new_known_queries),
                            "Combined known queries acc": combined_knownq_acc,
                            "Current refspeed": current_refspeed,
                            "Refined Score attack Acc": ref_score_acc,
                            "Vol Score Acc": vol_score_acc,
                            "Ref Vol Score Acc": ref_vol_score_acc,
                        }
                    )
                    csv_file.flush()


# Just an example run
def main():
    #df = enron_parser()
    #df = apache_parser
    df = wiki_parser()
    print(df)

    # Setup parameters
    sim_docs, server_docs = split_df(dframe=df, frac=0.4)
    sim_voc_size = 1200
    real_voc_size = 1000
    nr_known_queries = 4
    queryset_size = int(real_voc_size * 0.15)
    ref_speed = 10  # int(0.05 * queryset_size)

    # Number of experiment runs
    nr_runs = 1

    with open("example.csv", "w", newline="") as csv_file:
        fieldnames = [
            "Nr similar docs",
            "Nr server docs",
            "Similar voc size",
            "Server voc size",
            "Nr queries",
            "Nr queries known",
            "Base Score Acc",
            "Refined Score attack Acc",
            "Vol Score Acc",
            "Ref Vol Score Acc",
            "Cluster Vol Score Acc"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(nr_runs):
            print("CURRENT RUN = ", i)
            # Extract the data
            similar_extractor = KeywordExtractor(sim_docs, sim_voc_size, 1)
            real_extractor = QueryResultExtractor(server_docs, real_voc_size, 1)
            # real_extractor = PaddedResultExtractor(server_docs, real_voc_size, 1, n_pad=500, padding_and_volume_hiding=False)
            #real_extractor = ObfuscatedResultExtractor(server_docs, real_voc_size, 1)
            # real_extractor = VolumeHidingResultExtractor(server_docs, real_voc_size, 1)

            # Create queries
            query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

            # Known queries dictionary {keyword: keyword}
            known_queries = generate_known_queries(
                similar_wordlist=similar_extractor.get_sorted_voc(),
                stored_wordlist=query_voc,
                nr_queries=nr_known_queries,
            )

            # Trapdoor vocabulary
            td_voc = []

            # Temporary dictionary to replace known queries. Known queries only has (keyword, keyword) pairs. Since
            # the trapdoors were not encrypted yet.
            temp_known = {}

            # Dictionary {trapdoor: keyword}
            eval_dico = {}

            # Do the fake encryption for trapdoors
            for keyword in query_voc:
                fake_trapdoor = "enc_" + keyword
                td_voc.append(fake_trapdoor)

                # If it is a known query, we need to save which keyword corresponds to which trapdoor.
                if known_queries.get(keyword):
                    temp_known[fake_trapdoor] = keyword

                # Save inside {trapdoor: keyword} for evaluation.
                eval_dico[fake_trapdoor] = keyword

            # Replace known queries
            known_queries = temp_known

            attacker = ScoreAttacker(
                keyword_occ_array=similar_extractor.occ_array,
                keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                trapdoor_occ_array=query_array,
                trapdoor_sorted_voc=td_voc,
                known_queries=known_queries,
                sim_inv_index=similar_extractor.inv_index,
                real_inv_index=real_extractor.inv_index,
                sim_docs_vol_array=similar_extractor.vol_array,
                real_docs_vol_array=real_extractor.vol_array
            )

            # Trapdoor list are trapdoors that are not known.
            td_list = list(set(eval_dico.keys()).difference(attacker._known_queries.keys()))

            # Base Score
            # results_base_score = attacker.predict(td_list)
            # base_score_acc = np.mean([eval_dico[td] == candidates[0] for td, candidates in results_base_score.items()])

            # Refined score
            results_ref_score = attacker.predict_with_refinement(td_list, ref_speed=ref_speed)
            ref_score_acc = np.mean([eval_dico[td] == candidates[0] for td, candidates in results_ref_score.items()])

            # VolScore
            results_vol_score = attacker.predict_with_refinement_VOL(td_list, ref_speed=ref_speed)
            vol_score_acc = np.mean([eval_dico[td] == candidates[0] for td, candidates in results_vol_score.items()])

            # Retrieve new known queries
            new_known_queries = get_new_known_queries(results_ref_score, results_vol_score, known_queries)
            print("KNOWN:", known_queries)
            known_queries.update(new_known_queries)
            print("NEW KNOWN", known_queries)

            # Init new score attack with new known queries.
            attacker = ScoreAttacker(
                keyword_occ_array=similar_extractor.occ_array,
                keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                trapdoor_occ_array=query_array,
                trapdoor_sorted_voc=td_voc,
                known_queries=known_queries,
                sim_inv_index=similar_extractor.inv_index,
                real_inv_index=real_extractor.inv_index,
                sim_docs_vol_array=similar_extractor.vol_array,
                real_docs_vol_array=real_extractor.vol_array
            )

            start_time = time.time()
            # Refined score (aka ref Vol Score) with new known queries
            results_ref_vol_score = attacker.predict_with_refinement(td_list, ref_speed=ref_speed)
            ref_vol_score_acc = np.mean(
                [eval_dico[td] == candidates[0] for td, candidates in results_ref_vol_score.items()])
            print("--- %s seconds (rev_vol_score) ---" % (time.time() - start_time))

            start_time = time.time()
            # Cluster Vol Score
            results_cluster_vol_score = attacker.predict_with_cluster_refinement(td_list, max_ref_speed=10)
            cluster_vol_score_acc = np.mean(
                [eval_dico[td] == candidates[0] for td, candidates in results_cluster_vol_score.items()])
            print("--- %s seconds (cluster_vol_score) ---" % (time.time() - start_time))

            # print("base_score_acc", base_score_acc)
            print("ref_score_acc", ref_score_acc)
            print("vol_score_acc", vol_score_acc)
            print("ref_vol_score_acc", ref_vol_score_acc)
            print("cluster_vol_score_acc", cluster_vol_score_acc)
            print("amount dynamic ref speeds", len(attacker.dynamic_ref_speeds))
            print("sum dynamic ref speeds", sum(attacker.dynamic_ref_speeds))
            print("MEAN dynamic ref speed", statistics.mean(attacker.dynamic_ref_speeds))

            writer.writerow(
                {
                    "Nr similar docs": sim_docs.shape[0],
                    "Nr server docs": server_docs.shape[0],
                    "Similar voc size": sim_voc_size,
                    "Server voc size": real_voc_size,
                    "Nr queries": queryset_size,
                    "Nr queries known": nr_known_queries,
                    # "Base Score Acc": base_score_acc,
                    "Refined Score attack Acc": ref_score_acc,
                    "Vol Score Acc": vol_score_acc,
                    "Ref Vol Score Acc": ref_vol_score_acc,
                    "Cluster Vol Score Acc": cluster_vol_score_acc
                }
            )


if __name__ == '__main__':
    nltk.download("stopwords")
    nltk.download("punkt")

    #main()

    #reproduce_original("enron")

    #comparison_general("enron")
    #comparison_general("apache")
    #comparison_general("wiki")

    #comparison_low_knownq("enron")
    #comparison_low_knownq("apache")
    #comparison_low_knownq("wiki")

    #comparison_refspeed("enron")
    #comparison_refspeed("apache")
    #comparison_refspeed("wiki")

    #comparison_countermeasures("enron")
    #comparison_countermeasures("apache")
    #comparison_countermeasures("wiki")
