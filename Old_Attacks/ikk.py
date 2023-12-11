import math
import random
import time

from tqdm import tqdm

# Implementation of IKK attack
# As described in Access Pattern disclosure on Searchable Encryption: Ramification, Attack and Mitigation

# Algorithm 1 Optimizer
def optimizer(q_set, q_known, kw_set):
    # V, list of queries that are UNKNOWN to the attacker
    V = [q_i for q_i in q_set if q_i not in q_known]

    # All possible keywords
    keywords = kw_set

    # K, known query to keyword assignments to the attacker.
    # q_known consists of queries that we know that start with Q_ .
    # So we can simply trim the first 2 characters Q_ , to get the actual keyword.
    # K = (Q_john, john)
    K = [(q_i, q_i[2:]) for q_i in q_known]

    # D, list of plaintext keywords that are NOT in K since those keywords are already assigned.
    D = [kw for kw in keywords if kw not in [y for (_, y) in K]]

    # Make copy of plaintext keywords that are not in K
    valList = D.copy()

    # Initialize array. It will hold query to keyword assignment pairs.
    initState = []

    # For all queries, we assign a random keyword and add this (query, keyword) pair to initState.
    for var_i in V:
        val_i = random.choice(valList)
        initState.append((var_i, val_i))
        valList.remove(val_i)

    # Add known assignments to initState
    initState.extend(K)
    print("initState with query to keyword assignment", initState)
    return initState, K, D

def calc_acc(results):
    correct_count = 0
    for q_i, k_i in results:
        if q_i[2:] == k_i:
            correct_count += 1
    return "Current Accuracy:" + str(correct_count/len(results) * 100) + "%"

# Algorithm 2 Simulated Annealing
# The parameters initTemperature, coolingRate and rejectThreshold control how long this algorithm
# runs as condition in the while loop.
# Mp is the keyword co-occurrence matrix
# k_to_index is a map that maps a keyword to an index number (used for co-occ)
# Mc is the query co-occurence matrix
# q_to_index is a map that maps a query to an index number (used for co-occ)
# D list of plaintext keywords that are NOT in K.
# K is list of known query to keyword assignments
# stop_in_x_sec is an optional timer to run simulated annealing for x amount of seconds so that it is possible to
# run the algorithm even when the temperature drops near zero. Setting it to zero means we do not use the timer.
def sim_annealing(initTemperature, coolingRate, rejectThreshold, initState, Mp, k_to_index, Mc, q_to_index, D, K, stop_in_x_sec):

    currentState = initState
    succReject = 0
    currT = initTemperature

    # Just for logging purposes to keep track of the counts.
    loop_count = 0
    accepted_count = 0

    # Timer variables if we use the timer.
    time_elapsed = 0
    start_time = time.time()

    # Search for the solution until the temperature reaches near zero or if we have too many fails
    # We have an initial configuration, and we make a slight change to the configuration to see if we improve.
    # Some bounds on the algorithm. We stop if we reach temperature is zero or if we have too many rejects.
    # Also, a timer can be used to run this for a specific amount of time.
    # The 2.5e-323 is chosen to be a very small number, as close as possible to zero since the current temperature
    # is cooled down exponentially.
    with tqdm(total=float(2.5e-323), desc="sim annealing") as pbar:
        while currT > 2.5e-323 and succReject < rejectThreshold or time_elapsed < stop_in_x_sec:
            loop_count += 1
            time_elapsed = time.time() - start_time

            currentCost = 0
            nextCost = 0
            nextState = currentState.copy()

            # In the original algorithm it is possible to accidentally modify a (query, keyword) pair that was actually
            # KNOWN to be correct. This addition is to prevent that from happening, and actually use our prior knowledge.
            (x, y) = random.choice([elem for elem in nextState if elem not in K])

            # Select a random keyword which is not equal to y to change the configuration
            y2 = random.choice([r for r in D if r != y])

            # Remove old configuration, and add the new.
            nextState.remove((x, y))
            nextState.append((x, y2))

            # Loop through elements in currentState and check if there exists (z, y2).
            # If so, it means we need to remove (z, y2) because we just added (x, y2) before.
            # We cannot have two queries mapped to the same keyword.
            # So what we can do to fix the problem, is to assign query z to keyword y instead.
            # Notice that y is unused, because it was removed previously.
            output = [elem for elem in currentState if elem[1] == y2 and elem[0] != x]
            if len(output) > 0:
                (z, y2) = output[0]
                nextState.remove((z, y2))
                nextState.append((z, y))

            # We have a currentState and nextState which contains query to keyword mapping.
            # Calculate currentCost and nextCost using the sum of squared euclidean distance.
            # And also do the same calculation but for nextState, so we can compare the configurations.
            for i in range(len(Mc)):
                for j in range(len(Mc)):
                    # Each pair is (Query, Keyword) mapping.
                    (i_current, k) = currentState[i]
                    (i_next, k2) = nextState[i]
                    (j_current, l) = currentState[j]
                    (j_next, l2) = nextState[j]

                    # We use Mc which is the co-occurrence of queries, a query to query mapping to a probability.
                    # And Mp, which is the co-occurrence of keywords, a keyword to keyword mapping to a probability.
                    # We want to determine how close they are to each other.
                    currentCost += (Mc[q_to_index.get(i_current)][q_to_index.get(j_current)] - Mp[k_to_index.get(k)][k_to_index.get(l)]) ** 2
                    nextCost += (Mc[q_to_index.get(i_next)][q_to_index.get(j_next)] - Mp[k_to_index.get(k2)][k_to_index.get(l2)]) ** 2

            # Also known as the delta E, to check if we have found a better configuration
            E = nextCost - currentCost

            accept_new_state = False
            if E < 0:
                accept_new_state = True
                accepted_count += 1

            # We still want to accept even if it is a worse configuration with some probability
            # to avoid being stuck in local minimum. We are looking for the global minimum.
            else:
                accept_new_state = random.random() < math.exp(-E / currT)

        # Accept the new state, reset succReject counter and set nextState as currentState
            if accept_new_state:
                succReject = 0
                currentState = nextState
                print(calc_acc(nextState))
            else:
                succReject += 1

            # Decrease temperature
            currT = coolingRate * currT

            # Progress bar information
            pbar.n = currT
            pbar.refresh()

    print("loop count: ", loop_count)
    print("accepted count: ", accepted_count)
    return currentState

