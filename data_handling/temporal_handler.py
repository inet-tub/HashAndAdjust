import random
import json
import os
from math import log2

from params import param

#n_hosts = 1000
#size_dataset = 100000
single = True


def fetch_temp_seq():
    filename = "../loben/data/temp_file_processed_mod_n"+str(param.n_items)+"_m"+str(param.size_dataset)+"_more_ps"
    with open(filename+".json", 'r') as handle:
        traces_dict = json.load(handle)
    return traces_dict


def adapt_temporal_locality(sequence, temporal_p):
    """(Avin et al. : On the Complexity of Traffic Traces and Implications)
    At each step, we add a new request to the trace: with probability p, we repeat the last request.
    With probability (1 ? p), we sample a new request from the sequence."""
    p = temporal_p             # repeating probability p: the probability of sampling the last request
    if p == 0:
        return sequence
    trace = []
    original = sequence.copy()
    trace.append(original.pop(0))      # start by sampling the first pair from M.
    curr = trace[0]
    while original:         # if original size = 500, do 500 times
        if repeat(p):       # do with probability p
            trace.append(curr)
        else:       # pick next item
            trace.append(original[0])
            curr = original[0]      # update curr to new item
        original.pop(0)     # goto next item
    return trace


def create_nodes(size):
    """
    :param size: number of hosts to generate
    :return: randomly distributed hosts
    """
    nodes = []
    for i in range(1, size+1):
        nodes.append(i)
    res = random.sample(nodes, len(nodes))
    if single:
        return res
    else:
        return get_pair_reqs(res, size)


def get_pair_reqs(trace, n_hosts):
    """
    Transform simple trace (single requests) into pair-wise trace
    """
    pair_trace = []
    for r in trace:
        rand_dest = -99
        while rand_dest < 0 or rand_dest == r:
            random_n = random.randint(0, n_hosts-1)
            rand_dest = trace[random_n]
        pair_trace.append((r, rand_dest))      # add original item as source + one random item as destination
    return pair_trace


# return True if random number smaller than p
def repeat(probability):
    return random.random() < probability


def test_metrics(sequences_array):    
    entropy_set = calc_entropy(sequences_array)
    temporal_p_set = calc_temporal_locality(sequences_array)
    entropy = 0
    for e in entropy_set:
        entropy += e
    entropy /= len(entropy_set)  # take mean of entropies
    temp_p = 0
    for p in temporal_p_set:
        temp_p += p
    temp_p /= len(temporal_p_set)
    n_nodes = count_nodes(sequences_array[0])
    return [entropy, temp_p, n_nodes]


def get_metrics(sequence):
    n_nodes = count_nodes(sequence)
    n_req = len(sequence)
    return [n_nodes, n_req]


def calc_entropy(seq_array):
    entropy = 0
    entropy_set =[]
    for s in seq_array:
        frequency_set = sorted_nodes_occur(s)
        for i in range(len(frequency_set)):
            fx = frequency_set[i][1]/len(s)
            entropy += fx*log2(1/fx)
        entropy_set.append(entropy)
        entropy = 0
    return entropy_set


def calc_temporal_locality(seq_array):
    counter = 0
    p_set = []
    for s in seq_array:
        if single:
            curr = s[0]
            for i in range(1, len(s)):
                if s[i] == curr:
                    counter += 1
                curr = s[i]
        else:
            flat_s = [item for sublist in s for item in sublist]    # flatten list of lists
            curr = flat_s[0]
            for i in range(1, len(flat_s)):
                if flat_s[i] == curr:
                    counter += 1
                curr = flat_s[i]
        p_set.append(counter/len(s))
        counter = 0
    return p_set


def sorted_nodes_occur(x):
    counter = dict()
    for item in x:
        if item in counter:
            counter[item] += 1
        else:
            counter[item] = 1
    listofTuples = sorted(counter.items(), key=lambda x: x[1])
    return listofTuples    


def count_nodes(sequence):
    """returns counter: the set of distinct nodes (hosts) in the sequence"""
    counter = dict()
    #print(set)
    #print(sequence)
    if type(sequence) is dict:
        sequence = list(sequence.values())
    for item in sequence:
        if single:
            if item in counter:
                counter[item] += 1
            else:
                counter[item] = 0
        else:
            if item[0] in counter:
                counter[item[0]] += 1
            else:
                counter[item[0]] = 0
            if item[1] in counter:
                counter[item[1]] += 1
            else:
                counter[item[1]] = 0
    return len(counter)


def get_temp_original():
    filename = "data/temp_p_original.json"
    with open(filename) as f_in:
        file = json.load(f_in)
    return file


def pre_process_temp_seq():
    dataset = get_temp_original()
    new_dataset = {}
    logger = open("temp_process_log" + ".txt", "a")
    for temp_p in dataset:
        new_dataset[temp_p] = {}
        new_dataset[temp_p]["sequences"] = {}
        print("Temp p: " + str(temp_p))
        logger.write("Temp p: " + str(temp_p) + "\n")
        for sample in dataset[temp_p]["sequences"]:
            #if sample == "entropy" or sample == "temp_p": continue
            distinct_items = []
            #print(dataset[temp_p]["sequences"][sample][0])
            for r in range(0, len(dataset[temp_p]["sequences"][sample])):
            #for r in dataset[temp_p]["sequences"][sample]:
                if dataset[temp_p]["sequences"][sample][r] not in distinct_items:
                    distinct_items.append(dataset[temp_p]["sequences"][sample][r])
            new_dataset[temp_p]["sequences"][sample] = {}
            if len(distinct_items) == param.n_items:  # nothing to do
                print(len(distinct_items))
                logger.write("Sample " + str(sample) + " has all needed hosts! \n")
                for i in range(0, len(dataset[temp_p]["sequences"][sample])):
                    new_dataset[temp_p]["sequences"][sample][i] = dataset[temp_p]["sequences"][sample][i]
                    new_dataset[temp_p]["n_hosts"] = param.n_items
            else:
                print("analyzing sample: " + str(sample))
                logger.write("analyzing sample: " + str(sample) + "\n")
                appearing_items = []
                double_appearing_items = []
                curr_item = dataset[temp_p]["sequences"][sample][0]
                for i in range(0, len(dataset[temp_p]["sequences"][sample]) - 1):  # stop at second-last request
                    if curr_item in appearing_items:  # this means that it appears in 2 subsequences
                        double_appearing_items.append(curr_item)
                    if curr_item != dataset[temp_p]["sequences"][sample][i + 1] and curr_item not in appearing_items:  # add to candidates once we switch to another item
                        appearing_items.append(curr_item)     
                    curr_item = dataset[temp_p]["sequences"][sample][i + 1]
                print("Found " + str(len(double_appearing_items)) + " items that appear in 2 subsequences")
                logger.write(
                    "Found " + str(len(double_appearing_items)) + " items that appear in 2 subsequences" + "\n")

                # now create new sequence based on the missing items
                missing_items = param.n_items - len(distinct_items)
                logger.write("Replacing " + str(missing_items) + " items" + "\n")
                #print(("Replacing " + str(missing_items) + " items" + "\n"))
                random_sample = []
                while missing_items:
                    tentative = random.randint(0, param.n_items)
                    tentative2 = random.randint(0, param.n_items)
                    if single:
                        if tentative not in distinct_items and tentative not in random_sample:
                            random_sample.append(tentative)
                            missing_items -= 1
                    else:
                        if (tentative, tentative2) not in distinct_items and \
                                (tentative, tentative2) not in random_sample:  # get a random sample of items not in sequence
                            random_sample.append((tentative, tentative2))
                            missing_items -= 1
                logger.write("Picked " + str(len(random_sample)) + " random items" + "\n")
                already_replaced_items = []  # keep track of what items have been replaced already
                i = 0
                counter = 0
                while i < len(dataset[temp_p]["sequences"][sample]):
                    if random_sample and dataset[temp_p]["sequences"][sample][i] in double_appearing_items \
                            and dataset[temp_p]["sequences"][sample][i] not in already_replaced_items:
                        curr_req = dataset[temp_p]["sequences"][sample][i]
                        if curr_req not in already_replaced_items:
                            already_replaced_items.append(curr_req)
                        replacement = random_sample.pop()
                        new_dataset[temp_p]["sequences"][sample][i] = replacement
                        while dataset[temp_p]["sequences"][sample][i + 1] == curr_req:
                            i += 1
                            new_dataset[temp_p]["sequences"][sample][i] = replacement
                        logger.write("Replaced " + str(curr_req) + " with " + str(replacement) + "\n")
                        #print(("Replaced " + str(curr_req) + " with " + str(replacement) + "\n"))
                        counter += 1
                    else:
                        new_dataset[temp_p]["sequences"][sample][i] = dataset[temp_p]["sequences"][sample][i]
                        #print(new_dataset[temp_p]["sequences"][sample][i])
                    i += 1
                test = []
                
                for r in new_dataset[temp_p]["sequences"][sample]:
                    #print(new_dataset[temp_p]["sequences"][sample][r])
                    if new_dataset[temp_p]["sequences"][sample][r] not in test:     # check how many items there are now
                        test.append(new_dataset[temp_p]["sequences"][sample][r])
                new_dataset[temp_p]["n_hosts"] = len(test)
                logger.write("Test, sample" + str(sample) + " now has " + str(len(test)) + " items and " +
                             str(len(new_dataset[temp_p]["sequences"][sample])) + " requests, replaced " + str(counter) + " items  \n")
     
    with open("data/temp_file_processed.json", "w") as handle:
        json.dump(new_dataset, handle)
    logger.close()


def create_original():
    incr_p_set = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                  0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # params
    num_simulations = 1
    res_dict = {}
    tree_nodes = create_nodes(int(param.n_items))
    for p in incr_p_set:
        sequences_array = []
        res_dict[p] = {}
        for i in range(0, num_simulations):
            new_t_seq = []
            while len(new_t_seq) < param.size_dataset:
                r = random.randint(0, param.n_items-1)      # pick hosts randomly and add them to sequence
                request = tree_nodes[r]
                new_t_seq.append(request)
            sequences_array.append(adapt_temporal_locality(new_t_seq, p))      # postprocess sequence
        metrics = test_metrics(sequences_array)
        res_dict[p]["entropy"] = metrics[0]
        res_dict[p]["temp_p"] = metrics[1]
        res_dict[p]["n_hosts"] = metrics[2]
        res_dict[p]["sequences"] = {}
        for i in range (0, len(sequences_array)):
           res_dict[p]["sequences"][i] = sequences_array[i]
        #res_dict[p]["sequences"] = sequences_array
        
    with open("data/temp_p_original.json", 'w+') as f:
        json.dump(res_dict, f)


def mod_temp_seq():
    filename = "data/temp_file_processed"
    with open(filename + ".json", 'r') as handle:
        traces_dict = json.load(handle)
    new_dict = {}
    for p in traces_dict:
        metrics = get_metrics(traces_dict[p]["sequences"]["0"])
        new_dict[str(p)+"_n"+str(metrics[0])+"_m"+str(metrics[1])] = traces_dict[p]["sequences"]["0"]
    with open(filename+"_mod_n"+str(param.n_items)+"_m"+str(param.size_dataset)+"_more_ps.json", "x") as f:
        json.dump(new_dict, f, ensure_ascii=False)


def create_temp_sequence():
    if os.path.exists("../loben/data/temp_file_processed"+"_mod_n"+str(param.n_items)+"_m"+str(param.size_dataset)+"_more_ps.json"):
       return
    #if os.path.exists("../loben/data/temp_file_processed"+"_mod_n"+str(param.n_items)+"_m"+str(param.size_dataset)+".json"):
     #   return
    print(f"Creating temporal sequence for n_hosts: {param.n_items} and size_dataset: {param.size_dataset}")
    create_original()
    pre_process_temp_seq()
    mod_temp_seq()
    os.remove("data/temp_p_original.json")
    os.remove("data/temp_file_processed.json")
