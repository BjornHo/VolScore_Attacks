import csv

import matplotlib.pyplot as plt
import numpy as np
import statistics


def plot_reproduce_original():
    with open("reproduce_original.csv", "r", newline="") as csv_file:
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

        # Skips the column fieldnames by default
        csv_reader = csv.DictReader(csv_file)

        score_results = {5: [], 10: [], 20: [], 40: []}
        ref_score_results = {5: [], 10: [], 20: [], 40: []}

        for row in csv_reader:
            nr_knownQ = row["Nr queries known"]
            score_acc = row["Base Score Acc"]
            refScore_acc = row["Refined Score attack Acc"]

            score_results[int(nr_knownQ)].append(float(score_acc))
            ref_score_results[int(nr_knownQ)].append(float(refScore_acc))

        score_mean_5 = statistics.mean(score_results.get(5))
        score_std_5 = statistics.stdev(score_results.get(5))
        ref_score_mean_5 = statistics.mean(ref_score_results.get(5))
        ref_score_std_5 = statistics.stdev(ref_score_results.get(5))

        score_mean_10 = statistics.mean(score_results.get(10))
        score_std_10 = statistics.stdev(score_results.get(10))
        ref_score_mean_10 = statistics.mean(ref_score_results.get(10))
        ref_score_std_10 = statistics.stdev(ref_score_results.get(10))

        score_mean_20 = statistics.mean(score_results.get(20))
        score_std_20 = statistics.stdev(score_results.get(20))
        ref_score_mean_20 = statistics.mean(ref_score_results.get(20))
        ref_score_std_20 = statistics.stdev(ref_score_results.get(20))

        score_mean_40 = statistics.mean(score_results.get(40))
        score_std_40 = statistics.stdev(score_results.get(40))
        ref_score_mean_40 = statistics.mean(ref_score_results.get(40))
        ref_score_std_40 = statistics.stdev(ref_score_results.get(40))

        N = 2
        ind = np.arange(N)  # the x locations for the groups
        width = 0.2  # the width of the bars

        fig = plt.figure()
        ax = fig.add_subplot(111)

        fivevals = [score_mean_5, ref_score_mean_5]
        rects1 = ax.bar(ind, fivevals, width, yerr=[score_std_5, ref_score_std_5], capsize=8, color='b')
        tenvals = [score_mean_10, ref_score_mean_10]
        rects2 = ax.bar(ind + width, tenvals, width, yerr=[score_std_10, ref_score_std_10], capsize=8, color='orange')
        twentyvals = [score_mean_20, ref_score_mean_20]
        rects3 = ax.bar(ind + width * 2, twentyvals, width, yerr=[score_std_20, ref_score_std_20], capsize=8, color='g')
        fortyvals = [score_mean_40, ref_score_mean_40]
        rects4 = ax.bar(ind + width * 3, fortyvals, width, yerr=[score_std_40, ref_score_std_40], capsize=8, color='r')

        ax.set_ylabel('Accuracy')
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.set_ylim(0, 1)
        ax.set_xticks(ind + width + 0.1)
        ax.set_xticklabels(('Score', 'RefScore'))
        ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ("5 KnownQ", "10 KnownQ", "20 KnownQ", "40 KnownQ"),
                  loc='upper left', ncols=2)
        #plt.savefig('reproduce_original.pdf')
        plt.show()


def plot_comparison_general(dataset_name):
    with open("comparison_general_" + dataset_name + ".csv", "r", newline="") as csv_file:
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

        # Skips the column fieldnames by default
        csv_reader = csv.DictReader(csv_file)

        ref_score_results = {5: [], 10: [], 20: []}
        vol_score_results = {5: [], 10: [], 20: []}
        ref_vol_score_results = {5: [], 10: [], 20: []}
        cluster_vol_score_results = {5: [], 10: [], 20: []}

        for row in csv_reader:
            nr_knownQ = row["Nr queries known"]
            ref_score_acc = row["Refined Score attack Acc"]
            vol_score_acc = row["Vol Score Acc"]
            ref_vol_score_acc = row["Ref Vol Score Acc"]
            cluster_vol_score_acc = row["Cluster Vol Score Acc"]

            ref_score_results[int(nr_knownQ)].append(float(ref_score_acc))
            vol_score_results[int(nr_knownQ)].append(float(vol_score_acc))
            ref_vol_score_results[int(nr_knownQ)].append(float(ref_vol_score_acc))
            cluster_vol_score_results[int(nr_knownQ)].append(float(cluster_vol_score_acc))

        attack_results = [ref_score_results, vol_score_results, ref_vol_score_results, cluster_vol_score_results]
        all_means_5 = [statistics.mean(current_atk.get(5)) for current_atk in attack_results]
        all_std_5 = [statistics.stdev(current_atk.get(5)) for current_atk in attack_results]
        all_means_10 = [statistics.mean(current_atk.get(10)) for current_atk in attack_results]
        all_std_10 = [statistics.stdev(current_atk.get(10)) for current_atk in attack_results]
        all_means_20 = [statistics.mean(current_atk.get(20)) for current_atk in attack_results]
        all_std_20 = [statistics.stdev(current_atk.get(20)) for current_atk in attack_results]

        N = 4
        ind = np.arange(N)  # the x locations for the groups
        width = 0.2  # the width of the bars

        fig = plt.figure()
        ax = fig.add_subplot(111)

        fivevals = all_means_5
        rects1 = ax.bar(ind, fivevals, width, yerr=all_std_5, capsize=6, color='b')
        tenvals = all_means_10
        rects2 = ax.bar(ind + width, tenvals, width, yerr=all_std_10, capsize=6, color='orange')
        twentyvals = all_means_20
        rects3 = ax.bar(ind + width * 2, twentyvals, width, yerr=all_std_20, capsize=6, color='g')

        ax.set_ylabel('Query recovery accuracy')
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.set_ylim(0, 1)
        ax.set_xticks(ind + width)  # + 0.1)
        ax.set_xticklabels(('RefScore', 'VolScore', 'RefVolScore', 'ClusterVolScore'))
        ax.legend((rects1[0], rects2[0], rects3[0]), ("5 KnownQ", "10 KnownQ", "20 KnownQ"), loc='lower right', ncols=1)
        plt.savefig("comparison_general_" + dataset_name + ".pdf")
        plt.show()


def plot_comparison_lowknownq(dataset_name):
    with open("comparison_low_knownq_" + dataset_name + ".csv", "r", newline="") as csv_file:
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

        # Skips the column fieldnames by default
        csv_reader = csv.DictReader(csv_file)

        ref_score_results = {2: [], 3: [], 4: []}
        vol_score_results = {2: [], 3: [], 4: []}
        ref_vol_score_results = {2: [], 3: [], 4: []}
        cluster_vol_score_results = {2: [], 3: [], 4: []}
        nr_new_known_queries_results = {2: [], 3: [], 4: []}
        combined_known_queries_acc_results = {2: [], 3: [], 4: []}

        for row in csv_reader:
            nr_knownQ = row["Nr queries known"]
            ref_score_acc = row["Refined Score attack Acc"]
            vol_score_acc = row["Vol Score Acc"]
            ref_vol_score_acc = row["Ref Vol Score Acc"]
            cluster_vol_score_acc = row["Cluster Vol Score Acc"]
            nr_new_known_queries = row["Nr new known queries"]
            combined_known_queries_acc = row["Combined known queries acc"]

            ref_score_results[int(nr_knownQ)].append(float(ref_score_acc))
            vol_score_results[int(nr_knownQ)].append(float(vol_score_acc))
            ref_vol_score_results[int(nr_knownQ)].append(float(ref_vol_score_acc))
            cluster_vol_score_results[int(nr_knownQ)].append(float(cluster_vol_score_acc))
            nr_new_known_queries_results[int(nr_knownQ)].append(float(nr_new_known_queries))
            combined_known_queries_acc_results[int(nr_knownQ)].append(float(combined_known_queries_acc))

        attack_results = [ref_score_results, vol_score_results, ref_vol_score_results, cluster_vol_score_results]
        all_means_2 = [statistics.mean(current_atk.get(2)) for current_atk in attack_results]
        all_std_2 = [statistics.stdev(current_atk.get(2)) for current_atk in attack_results]
        all_means_3 = [statistics.mean(current_atk.get(3)) for current_atk in attack_results]
        all_std_3 = [statistics.stdev(current_atk.get(3)) for current_atk in attack_results]
        all_means_4 = [statistics.mean(current_atk.get(4)) for current_atk in attack_results]
        all_std_4 = [statistics.stdev(current_atk.get(4)) for current_atk in attack_results]

        nr_new_known_queries_means_2 = [statistics.mean(nr_new_known_queries_results.get(2))]
        nr_new_known_queries_means_3 = [statistics.mean(nr_new_known_queries_results.get(3))]
        nr_new_known_queries_means_4 = [statistics.mean(nr_new_known_queries_results.get(4))]
        all_nr_new_known_queries_means = [nr_new_known_queries_means_2, nr_new_known_queries_means_3,
                                          nr_new_known_queries_means_4]

        combined_known_queries_acc_means_2 = [statistics.mean(combined_known_queries_acc_results.get(2))]
        combined_known_queries_acc_means_3 = [statistics.mean(combined_known_queries_acc_results.get(3))]
        combined_known_queries_acc_means_4 = [statistics.mean(combined_known_queries_acc_results.get(4))]
        all_combined_known_queries_acc_means = [combined_known_queries_acc_means_2, combined_known_queries_acc_means_3
            , combined_known_queries_acc_means_4]

        all_acc_means = [all_means_2, all_means_3, all_means_4]

        print("STATISTICS for table")
        print("all nr new known queries means", all_nr_new_known_queries_means)
        print("all combined known queries acc means", all_combined_known_queries_acc_means)
        print("all acc means with 2 known ", [round((x * 100), 2) for x in all_means_2])
        print("all acc means with 3 known ", [round((x * 100), 2) for x in all_means_3])
        print("all acc means with 4 known ", [round((x * 100), 2) for x in all_means_4])

        with open("table_low_knownq_" + dataset_name + ".csv", "w", newline="") as csv_file:
            fieldnames = [
                "Dataset",
                "Nr queries known",
                "KnownQ acc",
                "Newly found KnownQ",
                "Total KnownQ Acc",
                "RefScore Acc",
                "VolScore Acc",
                "RefVolScore Acc",
                "ClusterVolScore Acc"
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for index, current_n in enumerate([2, 3, 4]):
                writer.writerow(
                    {
                        "Dataset": dataset_name,
                        "Nr queries known": current_n,
                        "KnownQ acc": 100,
                        "Newly found KnownQ": all_nr_new_known_queries_means[index][0],
                        "Total KnownQ Acc": round(100 * all_combined_known_queries_acc_means[index][0], 2),
                        "RefScore Acc": round(100 * all_acc_means[index][0], 2),
                        "VolScore Acc": round(100 * all_acc_means[index][1], 2),
                        "RefVolScore Acc": round(100 * all_acc_means[index][2], 2),
                        "ClusterVolScore Acc": round(100 * all_acc_means[index][3], 2)
                    }
                )
                csv_file.flush()

        N = 4
        ind = np.arange(N)  # the x locations for the groups
        width = 0.2  # the width of the bars

        fig = plt.figure()
        ax = fig.add_subplot(111)

        twovals = all_means_2
        rects1 = ax.bar(ind, twovals, width, yerr=all_std_2, capsize=6, color='tab:pink')
        threevals = all_means_3
        rects2 = ax.bar(ind + width, threevals, width, yerr=all_std_3, capsize=6, color='tab:gray')
        fourvals = all_means_4
        rects3 = ax.bar(ind + width * 2, fourvals, width, yerr=all_std_4, capsize=6, color='tab:purple')

        ax.set_ylabel('Query recovery accuracy')
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.set_ylim(0, 1)
        ax.set_xticks(ind + width)  # + 0.1)
        ax.set_xticklabels(('RefScore', 'VolScore', 'RefVolScore', 'ClusterVolScore'))
        ax.legend((rects1[0], rects2[0], rects3[0]), ("2 KnownQ", "3 KnownQ", "4 KnownQ"), loc='lower right', ncols=1)
        plt.savefig("comparison_low_knownq_" + dataset_name +".pdf")
        plt.show()


def plot_comparison_refspeed(dataset_name):
    with open("comparison_refspeed_" + dataset_name + ".csv", "r", newline="") as csv_file:
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

        # Skips the column fieldnames by default
        csv_reader = csv.DictReader(csv_file)

        ref_score_results = {2: [], 4: [], 8: []}
        vol_score_results = {2: [], 4: [], 8: []}
        ref_vol_score_results = {2: [], 4: [], 8: []}
        cluster_vol_score_results = {2: [], 4: [], 8: []}

        all_dynamic_refspeeds = {2: [], 4: [], 8: []}
        all_runtime_ref_vol_score = {2: [], 4: [], 8: []}
        all_runtime_cluster_vol_score = {2: [], 4: [], 8: []}

        for row in csv_reader:
            current_refspeed = row["Current refspeed"]
            ref_score_acc = row["Refined Score attack Acc"]
            vol_score_acc = row["Vol Score Acc"]
            ref_vol_score_acc = row["Ref Vol Score Acc"]
            cluster_vol_score_acc = row["Cluster Vol Score Acc"]
            dynamic_refspeeds = row["Dynamic refspeeds"]
            runtime_ref_vol_score = row["Runtime RefVolScore"]
            runtime_cluster_vol_score = row["Runtime ClusterVolScore"]

            ref_score_results[int(current_refspeed)].append(float(ref_score_acc))
            vol_score_results[int(current_refspeed)].append(float(vol_score_acc))
            ref_vol_score_results[int(current_refspeed)].append(float(ref_vol_score_acc))
            cluster_vol_score_results[int(current_refspeed)].append(float(cluster_vol_score_acc))

            # Convert to list of strings and append all dynamic refspeed numbers
            temp_list = dynamic_refspeeds.strip('][').split(', ')
            [all_dynamic_refspeeds[int(current_refspeed)].append(int(x)) for x in temp_list]

            all_runtime_ref_vol_score[int(current_refspeed)].append(float(runtime_ref_vol_score))
            all_runtime_cluster_vol_score[int(current_refspeed)].append(float(runtime_cluster_vol_score))

        attack_results = [ref_score_results, vol_score_results, ref_vol_score_results, cluster_vol_score_results]
        all_means_2 = [statistics.mean(current_atk.get(2)) for current_atk in attack_results]
        all_std_2 = [statistics.stdev(current_atk.get(2)) for current_atk in attack_results]
        all_means_4 = [statistics.mean(current_atk.get(4)) for current_atk in attack_results]
        all_std_4 = [statistics.stdev(current_atk.get(4)) for current_atk in attack_results]
        all_means_8 = [statistics.mean(current_atk.get(8)) for current_atk in attack_results]
        all_std_8 = [statistics.stdev(current_atk.get(8)) for current_atk in attack_results]

        # means and stds for refspeed 2, 4 and 8
        refspeed_values = [2, 4, 8]
        all_dyn_refspeed_means = [statistics.mean(all_dynamic_refspeeds.get(i)) for i in refspeed_values]
        all_dyn_refspeed_stds = [statistics.stdev(all_dynamic_refspeeds.get(i)) for i in refspeed_values]

        all_runtime_ref_vol_score_means = [statistics.mean(all_runtime_ref_vol_score.get(i)) for i in refspeed_values]
        all_runtime_cluster_vol_score_means = [statistics.mean(all_runtime_cluster_vol_score.get(i)) for i in
                                               refspeed_values]

        print("STATISTICS for table")
        print("mean dyn refspeed", all_dyn_refspeed_means)
        print("runtime refvolscore", all_runtime_ref_vol_score_means)
        print("runtime clustervolscore", all_runtime_cluster_vol_score_means)

        # print("all nr new known queries means", all_nr_new_known_queries_means)
        # print("all combined known queries acc means", all_combined_known_queries_acc_means)

        print("all acc means with RefSpeed 2 ", [round((x * 100), 2) for x in all_means_2])
        print("all acc means with RefSpeed 4 ", [round((x * 100), 2) for x in all_means_4])
        print("all acc means with RefSpeed 8 ", [round((x * 100), 2) for x in all_means_8])

        all_means = [all_means_2, all_means_4, all_means_8]

        with open("table_refspeed_" + dataset_name + ".csv", "w", newline="") as csv_file:
            fieldnames = [
                "Dataset",
                "Attack",
                "RefSpeed",
                "Mean Dynamic RefSpeed",
                "Runtime (sec)",
                "Accuracy (%)"
            ]

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for index, current_refspeed in enumerate([2, 4, 8]):
                for current_atk in ["RefVolScore", "ClusterVolScore"]:
                    writer.writerow(
                        {
                            "Dataset": dataset_name,
                            "Attack": current_atk,
                            "RefSpeed": current_refspeed,
                            "Mean Dynamic RefSpeed": round(all_dyn_refspeed_means[index], 2) if current_atk == "ClusterVolScore" else "N/A",
                            "Runtime (sec)": round(all_runtime_ref_vol_score_means[index], 2) if current_atk == "RefVolScore" else round(all_runtime_cluster_vol_score_means[index], 2),
                            "Accuracy (%)": round(100 * all_means[index][2], 2) if current_atk == "RefVolScore" else round(100 * all_means[index][3], 2)

                        }
                    )
                    csv_file.flush()


        N = 4
        ind = np.arange(N)  # the x locations for the groups
        width = 0.2  # the width of the bars

        fig = plt.figure()
        ax = fig.add_subplot(111)

        twovals = all_means_2
        rects1 = ax.bar(ind, twovals, width, yerr=all_std_2, capsize=6,
                        color=['royalblue', 'royalblue', 'royalblue', 'seagreen'])
        threevals = all_means_4
        rects2 = ax.bar(ind + width, threevals, width, yerr=all_std_4, capsize=6,
                        color=['gold', 'gold', 'gold', 'dimgrey'])
        fourvals = all_means_8
        rects3 = ax.bar(ind + width * 2, fourvals, width, yerr=all_std_8, capsize=6,
                        color=['crimson', 'crimson', 'crimson', 'rebeccapurple'])

        ax.set_ylabel('Query recovery accuracy')
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.set_ylim(0, 1)
        ax.set_xticks(ind + width)  # + 0.1)
        ax.set_xticklabels(('RefScore', 'VolScore', 'RefVolScore', 'ClusterVolScore'))
        ax.legend((rects1[0], rects2[0], rects3[0], rects1[3], rects2[3], rects3[3]), (
        "RefSpeed 2", "RefSpeed 4", "RefSpeed 8", "RefSpeed 2 + MaxRefSpeed 10", "RefSpeed 4 + MaxRefSpeed 10",
        "RefSpeed 8 + MaxRefSpeed 10"), bbox_to_anchor=(1.04, 0), loc='lower right', ncols=1)
        plt.savefig("comparison_refspeed_" + dataset_name + ".pdf")
        plt.show()


def plot_comparison_countermeasures(dataset_name):
    with open("comparison_countermeasures_" + dataset_name + ".csv", "r", newline="") as csv_file:
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

        # Skips the column fieldnames by default
        csv_reader = csv.DictReader(csv_file)
        countermeasure_types = ["none", "volume_hiding", "padding"]
        ref_score_results = {}

        ref_score_means = {}
        ref_score_stds = {}
        ref_vol_score_results = {}
        ref_vol_score_means = {}
        ref_vol_score_stds = {}
        voc_sizes = [500, 1000, 2000]

        # Initialize maps {countermeasure : {} } and {countermeasure : [] }
        for current_cm in countermeasure_types:
            ref_score_results[current_cm] = {}
            ref_vol_score_results[current_cm] = {}
            ref_score_means[current_cm] = []
            ref_score_stds[current_cm] = []
            ref_vol_score_means[current_cm] = []
            ref_vol_score_stds[current_cm] = []

        # Init map {countermeasure : {voc_size : []}}
        # We do this twice, for ref_score and ref_vol_score.
        # We want to store for each counter measure, a voc size together with list of all accuracies.
        for current_cm in countermeasure_types:
            for current_voc_size in voc_sizes:
                ref_score_results[current_cm][current_voc_size] = []
                ref_vol_score_results[current_cm][current_voc_size] = []


        for row in csv_reader:
            # sim voc_size == server voc_size in this experiment
            voc_size = row["Similar voc size"]
            ref_score_acc = row["Refined Score attack Acc"]
            ref_vol_score_acc = row["Ref Vol Score Acc"]
            countermeasure = row["Counter measure"]

            ref_score_results[countermeasure][int(voc_size)].append(float(ref_score_acc))
            ref_vol_score_results[countermeasure][int(voc_size)].append(float(ref_vol_score_acc))


        # Example:
        # {'none': [0.8941666666666667, 0.8340740740740741, 0.7375438596491228], 'volume_hiding': [0.8741666666666666, 0.8285185185185185, 0.7401754385964913]", ...}
        # 3 values, for voc_size 500, 1000 and 2000. The values are all mean accuracies over 20 runs.
        for current_cm in countermeasure_types:
            for current_voc_size in voc_sizes:
                ref_score_means[current_cm].append(statistics.mean(ref_score_results.get(current_cm).get(current_voc_size)))
                ref_score_stds[current_cm].append(statistics.stdev(ref_score_results.get(current_cm).get(current_voc_size)))
                ref_vol_score_means[current_cm].append(statistics.mean(ref_vol_score_results.get(current_cm).get(current_voc_size)))
                ref_vol_score_stds[current_cm].append(statistics.stdev(ref_vol_score_results.get(current_cm).get(current_voc_size)))


        # Plotting for RefScore and RefVolScore
        attack_names = ["RefScore", "RefVolScore"]
        attack_means = [ref_score_means, ref_vol_score_means]
        attack_stds = [ref_score_stds, ref_vol_score_stds]

        for index, current_atk in enumerate(attack_names):
            N = 3
            ind = np.arange(N)  # the x locations for the groups
            width = 0.2  # the width of the bars

            fig = plt.figure()
            ax = fig.add_subplot(111)
            rects1 = ax.bar(ind, attack_means[index].get("none"), width, yerr=attack_stds[index].get("none"), capsize=6, color='grey')
            rects2 = ax.bar(ind + width, attack_means[index].get("volume_hiding"), width, yerr=attack_stds[index].get("volume_hiding"), capsize=6, color='skyblue')
            rects3 = ax.bar(ind + width * 2, attack_means[index].get("padding"),  width, yerr=attack_stds[index].get("padding"), capsize=6, color='darkorange')

            ax.set_xlabel('Vocabulary size (sim = real)')
            ax.set_ylabel('Query recovery accuracy')
            # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax.set_ylim(0, 1)
            ax.set_xticks(ind + width)  # + 0.1)
            ax.set_xticklabels(('500', '1000', '2000'))
            ax.legend((rects1[0], rects2[0], rects3[0]), ("No countermeasure", "Volume hiding", "Padding"), loc='upper right', ncols=1)
            plt.savefig("comparison_countermeasures_" + current_atk + "_" + dataset_name +".pdf")
            plt.show()

dataset_names = ["enron", "apache", "wiki"]

#plot_reproduce_original()
for current_dataset in dataset_names:
    #plot_comparison_general(current_dataset)
    #plot_comparison_lowknownq(current_dataset)
    #plot_comparison_refspeed(current_dataset)
    #plot_comparison_countermeasures(current_dataset)
    pass