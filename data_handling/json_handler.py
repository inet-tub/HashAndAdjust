import datetime
import json
import os.path
import random
from params import param
from scipy.stats import poisson


import data_handling.temporal_handler as th

def check_dataset():
    if param.exp_type == "temp":
        th.create_temp_sequence()
    elif param.exp_type == "ctr":
        if not os.path.exists("data/ctr_seq_time.json"):
            raise Exception("Could not find ctr-dataset")
    else:
        raise Exception("Valid exp_type inputs: \"ctr\" or \"temp\"")

def add_temp_res_to_json(path, data):
    if os.path.exists(path):
        with open(path, "r") as fp:
            curr_data = json.load(fp)       # curr_data = data we had up to now
        for trace in data:                  # data = data we add to curr_data
            if trace not in curr_data:
                curr_data[trace] = data[trace]
            else:
                #print(f"Adding to current data: {curr_data}")
                for add_c in data[trace]:
                    if add_c not in curr_data[trace]:   # if addit_capac not registered, add it with new values
                        curr_data[trace][add_c] = data[trace][add_c]
                        print(f"Adding {add_c} to {curr_data}")
                    else:       # if we have this add_c, add new params/overwrite results for old params
                        for n_serv in data[trace][add_c]:
                            curr_data[trace][add_c][n_serv] = data[trace][add_c][n_serv]
        with open(path, "w") as fp:
            json.dump(curr_data, fp)
    else:
        with open(path, "w") as fp:
            json.dump(data, fp)

def generate_serv_ins_del_ctr():
    server_insert_timestamps = poisson.rvs(mu=param.serv_ins_freq,
                                                size=int(
                                                    131490 / param.serv_ins_freq + 131490 / param.serv_ins_freq * 0.1)).tolist()
    server_deletion_timestamps = poisson.rvs(mu=param.serv_del_freq,
                                                  size=int(
                                                      131490 / param.serv_del_freq + 131490 / param.serv_del_freq * 0.1)).tolist()
    saving_dict = {}
    saving_dict["ins_times"] = server_insert_timestamps.copy()
    saving_dict["del_times"] = server_deletion_timestamps.copy()
    present_servers = set()
    removed_servers = set()
    for i in range(0, param.init_servers):
        present_servers.add(i)
    deleting_server_ids = []
    tot_intervals = int(len(server_deletion_timestamps))
    for i in range(0, tot_intervals):
        rand_server_id = random.sample(sorted(present_servers), 1)[0]  # choose random server to delete
        print(i)
        if (i > 1 and deleting_server_ids[i - 1] == rand_server_id) or (
                i > 1 and deleting_server_ids[i - 2] == rand_server_id) \
                or (i > 2 and deleting_server_ids[i - 3] == rand_server_id):
            deleting_server_ids.append(-99)
            continue
        if i > 1:
            print(f"inserting {rand_server_id}, ancestor = {deleting_server_ids[i - 1]}")
        deleting_server_ids.append(rand_server_id)
        present_servers.remove(rand_server_id)
        removed_servers.add(rand_server_id)
        if len(present_servers) < param.init_servers * 0.6:
            reinserting_server = random.sample(sorted(removed_servers), 1)[0]
            present_servers.add(reinserting_server)
            removed_servers.remove(reinserting_server)

    # test that there are no 2 consecutive ids in list
    for d in deleting_server_ids:
        if d == -99:
            deleting_server_ids.remove(d)
    for i in range(0, len(deleting_server_ids)):
        if i < len(deleting_server_ids) - 4 and (
                deleting_server_ids[i] == deleting_server_ids[i + 1] or deleting_server_ids[i] == deleting_server_ids[
            i + 2]):
            raise Exception(f"Two same server IDs to remove too close to each other, should not happen."
                            f"Error shows up with id={deleting_server_ids[i]}")
    saving_dict["del_items"] = deleting_server_ids
    diff = len(server_deletion_timestamps) - len(deleting_server_ids)
    if diff > 0:
        for i in range(0, diff):
            deleting_server_ids.append(deleting_server_ids[i])
    print(f"Len of intervals: {len(server_deletion_timestamps)}")
    print(f"Len of deleting items: {len(deleting_server_ids)}")
    print(deleting_server_ids)
    with open(f"data/serv_ins_del_interv_ctr_m{param.size_dataset}_n{param.init_servers}_f{param.serv_ins_freq}.json", "w") as fp:
        json.dump(saving_dict, fp)
def generate_serv_ins_del_temp():
    tot_minutes = param.size_dataset / 60   # a request per second, so /60 to get the minutes
    server_insert_timestamps = poisson.rvs(mu=param.serv_ins_freq,
                                                size=int(
                                                    tot_minutes / param.serv_ins_freq + tot_minutes / param.serv_ins_freq * 0.1) + 100).tolist()
    server_deletion_timestamps = poisson.rvs(mu=param.serv_del_freq,
                                                  size=int(
                                                      tot_minutes / param.serv_del_freq + tot_minutes / param.serv_del_freq * 0.1) + 100).tolist()
    saving_dict = {}
    saving_dict["ins_times"] = server_insert_timestamps.copy()
    saving_dict["del_times"] = server_deletion_timestamps.copy()
    present_servers = set()
    removed_servers = set()
    for i in range(0, param.init_servers):
        present_servers.add(i)
    deleting_server_ids = []
    tot_intervals = int(len(server_deletion_timestamps))
    for i in range(0, int(tot_intervals+tot_intervals*0.1)):        # add 10% to avoid problems at end of generation
        rand_server_id = random.sample(sorted(present_servers), 1)[0]  # choose random server to delete
        print(i)
        if (i>1 and deleting_server_ids[i-1] == rand_server_id) or (i>1 and deleting_server_ids[i-2] == rand_server_id)\
                or (i>2 and deleting_server_ids[i-3] == rand_server_id):
            deleting_server_ids.append(-99)
            continue
        if i > 1:
            print(f"inserting {rand_server_id}, ancestor = {deleting_server_ids[i-1]}" )
        deleting_server_ids.append(rand_server_id)
        present_servers.remove(rand_server_id)
        removed_servers.add(rand_server_id)
        if len(present_servers) < param.init_servers*0.6:
            reinserting_server = random.sample(sorted(removed_servers), 1)[0]
            present_servers.add(reinserting_server)
            removed_servers.remove(reinserting_server)

    # test that there are no 2 consecutive ids in list
    for d in deleting_server_ids:
        if d == -99:
            deleting_server_ids.remove(d)
    for i in range(0, len(deleting_server_ids)):
        if i < len(deleting_server_ids)-4 and (deleting_server_ids[i] == deleting_server_ids[i+1] or deleting_server_ids[i] == deleting_server_ids[i+2]):
            raise Exception(f"Two same server IDs to remove too close to each other, should not happen."
                            f"Error shows up with id={deleting_server_ids[i]}")
    saving_dict["del_items"] = deleting_server_ids
    diff = len(server_deletion_timestamps) - len(deleting_server_ids)
    if diff > 0:
        for i in range(0, diff):
            deleting_server_ids.append(deleting_server_ids[i])
    print(f"Len of intervals: {len(server_deletion_timestamps)}")
    print(f"Len of deleting items: {len(deleting_server_ids)}")
    print(deleting_server_ids)
    with open(f"data/serv_ins_del_interv_temp_m{param.size_dataset}.json", "w") as fp:
        json.dump(saving_dict, fp)

def get_serv_ins_del_temp():
    with open(f"../loben/data/serv_ins_del_interv_temp_m{param.size_dataset}.json", "r") as fp:
        curr_data = json.load(fp)
    #print(curr_data)
    return curr_data

def get_serv_ins_del_ctr():
    with open(f"data/serv_ins_del_interv_ctr_m{param.size_dataset}"
              f"_n{param.init_servers}_f{param.serv_ins_freq}.json", "r") as fp:
        curr_data = json.load(fp)
    #print(curr_data)
    return curr_data


