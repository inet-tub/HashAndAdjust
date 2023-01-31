import datetime

import pandas as pd
import json

def fetch_aol_seq(test=False, time=False):
    traces_dict = None
    filename = "ordered_aol"
    if test: filename = "test_"+filename
    if time: filename += "_time"
    with open(filename+".json", 'r') as handle:
        traces_dict = json.load(handle)
    aol_trace = None
    for trace in traces_dict:
        aol_trace = traces_dict[trace]["trace_variant"]["original_trace"]
        print("Fetching seq " + str(trace))
    return aol_trace

def fetch_ctr_seq():
    filename = "data/ctr_1mio.csv"
    with open(filename+".json", 'r') as handle:
        traces_dict = json.load(handle)
    if len(traces_dict.keys()) > 1:
        raise Exception("There are more than one seq in dict!")
    for trace in traces_dict:
        ctr_trace = traces_dict[trace]["trace_variant"]["original_trace"]
        return [trace, ctr_trace]

def ctr_csv_to_json_():
    filename = "ctr_1mio.csv"
    df = pd.read_csv('data/'+filename)
    req_list = []
    for index, row in df.iterrows():
        req_list.append(row['site_id'])
    metrics = get_metrics(req_list, single=True)
    print("Req counts " + str(metrics[0]) + " nodes and " + str(metrics[1]) + " req")
    trace_dict = {}
    trace_dict[filename + "_n" + str(metrics[0]) + "_m" + str(metrics[1])] = {}
    trace_dict[filename + "_n" + str(metrics[0]) + "_m" + str(metrics[1])]["trace_variant"] = {}
    trace_dict[filename + "_n" + str(metrics[0]) + "_m" + str(metrics[1])]["trace_variant"]["original_trace"] = req_list
    with open("data/"+filename + ".json", "x") as f:
        json.dump(trace_dict, f, ensure_ascii=False)

def fetch_ctr_seq_time():
    """
    :return: list, where first item is a string with name of trace and second item is the trace itself
    """
    filename = "data/ctr_seq_time"
    with open(filename+".json", 'r') as handle:
        traces_dict = json.load(handle)
    if len(traces_dict.keys()) > 1:
        raise Exception("There are more than one seq in dict!")
    for trace in traces_dict:
        ctr_trace = traces_dict[trace]["trace_variant"]["original_trace"]
        return [trace, ctr_trace]


def save_ctr_seq_time_100k():
    """
    :return: list, where first item is a string with name of trace and second item is the trace itself
    """
    filename = "data/ctr_seq_time"
    with open(filename+".json", 'r') as handle:
        traces_dict = json.load(handle)
    if len(traces_dict.keys()) > 1:
        raise Exception("There are more than one seq in dict!")
    new_trace=[]
    req_list=[]
    for trace in traces_dict:
        ctr_trace = traces_dict[trace]["trace_variant"]["original_trace"]
    for i in range(0, 100000):
        new_trace.append(ctr_trace[i])
        req_list.append(ctr_trace[i][0])
    metrics = get_metrics(req_list, single=True)
    trace_dict = {}
    filename+="_100k"
    trace_dict[filename + "_n" + str(metrics[0]) + "_m" + str(metrics[1])] = {}
    trace_dict[filename + "_n" + str(metrics[0]) + "_m" + str(metrics[1])]["trace_variant"] = {}
    trace_dict[filename + "_n" + str(metrics[0]) + "_m" + str(metrics[1])]["trace_variant"]["original_trace"] = new_trace
    with open(filename + ".json", "x") as f:
        json.dump(trace_dict, f, ensure_ascii=False)

def fetch_ctr_seq_time_100k():
    filename = "data/ctr_seq_time_100k_mapped"
    with open(filename + ".json", 'r') as handle:
        traces_dict = json.load(handle)
    for trace in traces_dict:
        ctr_trace = traces_dict[trace]["trace_variant"]["original_trace"]
        return [trace, ctr_trace]

def transform_ctr_seq_time_100k():
    """
    :return: list, where first item is a string with name of trace and second item is the trace itself
    """
    filename = "data/ctr_seq_time_100k"
    dest_name = "data/ctr_seq_time_100k_mapped"
    dest_dict = {}
    with open(filename+".json", 'r') as handle:
        traces_dict = json.load(handle)
    if len(traces_dict.keys()) > 1:
        raise Exception("There are more than one seq in dict!")
    for trace in traces_dict:
        ctr_trace = traces_dict[trace]["trace_variant"]["original_trace"]

        # map strings to integers
        unique_vals = set()
        for x in ctr_trace:
            unique_vals.add(x[0])
        unique_list = list(unique_vals)
        keys_to_str = dict(zip(unique_list, range(len(unique_list))))
        #print(keys_to_str)
        new_ctr_trace = []
        for x in ctr_trace:
            new_ctr_trace.append([keys_to_str[x[0]], x[1]])
        dest_dict[trace] = {}
        dest_dict[trace]["trace_variant"] = {}
        dest_dict[trace]["trace_variant"]["original_trace"] = {}
        dest_dict[trace]["trace_variant"]["original_trace"] = new_ctr_trace
        # mapping test
        #print(ctr_trace[:4])
        #print(new_ctr_trace[:4])
        with open(dest_name + ".json", "x") as f:
            json.dump(dest_dict, f, ensure_ascii=False)

def get_n_items(dict_name):
    spl_char = '_n'
    first_part = dict_name.split(spl_char, 1)[1]
    spl_char = '_m'
    return first_part.split(spl_char, 1)[0]

def order_save_data():
    df = pd.read_csv('aol4ps_data.csv', sep="\t", encoding='utf-8')

    # print(list(df.columns))
    sel_cols = df[["AnonID", "DocIndex", "QueryTime"]].copy()

    # order dataset by timestamp
    sel_cols["QueryTime"] = pd.to_datetime(df['QueryTime'])
    sel_cols = sel_cols.sort_values(by="QueryTime")
    print(sel_cols)
    sel_cols.to_csv("ordered_aol.csv")

def save_json_seq(time=False):
    df = pd.read_csv('data/ordered_aol.csv')
    #print(df)
    req_list = []
    for index, row in df.iterrows():
        if time:
            req_list.append((row['AnonID'], row['DocIndex'],  row['QueryTime']))
        else:
            req_list.append((row['AnonID'], row['DocIndex']))
    metrics = get_metrics(req_list)
    print("Req counts " + str(metrics[0]) + " nodes and " + str(metrics[1]) + " req")
    trace_dict = {}
    trace_dict["ordered_aol_n" + str(metrics[0]) + "_m" + str(metrics[1])] = {}
    trace_dict["ordered_aol_n" + str(metrics[0]) + "_m" + str(metrics[1])]["trace_variant"] = {}
    trace_dict["ordered_aol_n" + str(metrics[0]) + "_m" + str(metrics[1])]["trace_variant"]["original_trace"] = req_list
    dest_name = "ordered_aol"
    if time: dest_name += "_time"
    with open(dest_name+".json", "x") as f:
        json.dump(trace_dict, f, ensure_ascii=False)

def save_json_test_seq(time=False):
    df = pd.read_csv('data/ordered_aol.csv')
    req_list = []
    counter = 10000
    for index, row in df.iterrows():
        if time:
            req_list.append((row['AnonID'], row['DocIndex'],  row['QueryTime']))
        else:
            req_list.append((row['AnonID'], row['DocIndex']))
        counter -= 1
        if not counter: break
    metrics = get_metrics(req_list)
    print("Req counts " + str(metrics[0]) + " nodes and " + str(metrics[1]) + " req")
    trace_dict = {}
    trace_dict["ordered_aol_n" + str(metrics[0]) + "_m" + str(metrics[1])] = {}
    trace_dict["ordered_aol_n" + str(metrics[0]) + "_m" + str(metrics[1])]["trace_variant"] = {}
    trace_dict["ordered_aol_n" + str(metrics[0]) + "_m" + str(metrics[1])]["trace_variant"]["original_trace"] = req_list
    dest_name = "test_ordered_aol"
    if time: dest_name += "_time"
    with open(dest_name+".json", "x") as f:
        json.dump(trace_dict, f, ensure_ascii=False)

def count_nodes(set, single):
        """returns number of distinct nodes (hosts) in the sequence"""

        counter = dict()
        for item in set:
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

def get_metrics(sequence, single=False):
    n_nodes = count_nodes(sequence, single=single)
    n_req = len(sequence)
    return [n_nodes, n_req]