import csv
import datetime
import json
import os.path
import random

from params import param
from implementation.circle import Circle
from data_handling.ctr_handler import get_n_items, fetch_ctr_seq_time_100k
from data_handling.temporal_handler import fetch_temp_seq
from implementation.push_down_algo import Push_down_algo
from implementation.static_algo import Static_algo
import data_handling.json_handler as jh
import implementation.plotter as pl
import implementation.algo_utils as au

def run_paper_experiment_1():
    capacities = ["4", "7", "10"]
    temp_ps = ["0.15", "0.3", "0.45", "0.6", "0.75", "0.9"]
    param.exp_type = "temp"
    param.algorithm = "Both"
    param.set_n_items(1000)
    param.set_size_dataset(1000000)
    jh.check_dataset()
    for c in capacities:
        param.addit_augm = c
        for p in temp_ps:
            param.temp_p = p
            run_experiment_temp()
    pl.sum_results_all_ps()
    pl.plot_3d_all_ps_fixed_server_n()

def run_paper_experiment_1_unbound():
    temp_ps = ["0.15", "0.9"]
    param.exp_type = "temp"
    param.algorithm = "Both"
    param.set_n_items(10000)
    param.set_size_dataset(100000)
    param.unbounded_capacity = True
    param.show_circle_occup = True
    jh.check_dataset()
    for p in temp_ps:
        param.temp_p = p
        run_experiment_temp()
    pl.plot_infin_cap_mtf()
    pl.plot_occ()


def run_paper_experiment_2():
    n_servers_params = [10, 25, 50, 75, 100, 150, 200]
    param.addit_augm = "4"
    param.temp_p = "0.75"
    param.set_n_items(1000)
    param.set_size_dataset(100000)
    jh.check_dataset()
    if param.exp_type == "temp":
        for n in n_servers_params:
            param.set_n_servers(n)
            run_experiment_temp()
    pl.line_plot_n_servers()

def run_paper_experiment_3():
    stale_time_params = [50, 100, 150, 200, 300]
    param.addit_augm = "4"
    param.temp_p = "0.75"
    param.set_n_servers(100)
    param.set_n_items(1000)
    param.set_size_dataset(100000)
    jh.check_dataset()
    for s in stale_time_params:
        param.stale_time=s
        if param.exp_type == "temp":
            run_experiment_temp()
    pl.line_plot_stale()

def run_paper_experiment_4():
    freq_params = [8, 10]#[50, 100, 150, 200, 300]
    param.addit_augm = "4"
    param.temp_p = "0.75"
    param.set_n_servers(100)
    param.set_n_items(1000)
    param.set_size_dataset(100000)
    jh.check_dataset()
    for f in freq_params:
        param.serv_ins_freq = param.serv_del_freq = f
        if param.exp_type == "temp":
            run_experiment_temp()
    pl.line_plot_freq()


def run_single_experiment():
    if param.exp_type == "temp":
        run_experiment_temp()
    elif param.exp_type == "ctr":
        run_experiment_ctr()


def run_server_occ(mult=False, infin=False):
    """
    Test server occupancy with various capacity on temporal data
    """
    temp_ps = ["0.15", "0.9"]
    param.addit_augm = "2"
    param.set_n_servers(20)
    param.set_n_items(10000)
    param.set_size_dataset(100000)
    param.exp_type = "ctr"      # used in paper
    param.initial_occup_factor = 1
    param.hash_f = "5k"
    jh.check_dataset()
    if param.exp_type == "temp":
        sequences = fetch_temp_seq()
        for p in temp_ps:
            param.temp_p = p
            for seq in sequences:
                sequence = list(sequences[seq].values())
                if not seq.startswith(p):
                    continue
                m_items = int(get_n_items(seq))
                print(f"Running on seq {seq}, # items: {m_items}, # req: {len(sequence)}")
                if mult:
                    server_capacity = m_items / param.init_servers * param.initial_occup_factor * 1.25
                elif infin:
                    server_capacity = m_items * 2
                else:
                    server_capacity = m_items / param.init_servers * param.initial_occup_factor + int(param.addit_augm)
                preloading_items = au.get_preloaded_items(sequence.copy())
                static_c = Circle(id=1, n_servers=param.init_servers, m_items=m_items)
                static_c.server_capacity = int(server_capacity)
                static_c.init_servers(preloaded_items=preloading_items.copy())
                record_circle_occupation(static_c, mult=mult, infin=infin)
    elif param.exp_type == "ctr":
        seq_data = fetch_ctr_seq_time_100k()
        seq = seq_data[0]
        sequence = seq_data[1]
        m_items = int(get_n_items(seq))
        print(f"Running on seq {seq}, # items: {m_items}, # req: {len(sequence)}")
        if mult:
            server_capacity = m_items / param.init_servers * param.initial_occup_factor * 1.25
        elif infin:
            server_capacity = m_items * 2
        else:
            server_capacity = m_items / param.init_servers * param.initial_occup_factor + int(param.addit_augm)
        print(f"Server capacity: {int(server_capacity)}")
        preloading_items = au.get_preloaded_items(sequence.copy())
        static_c = Circle(id=1, n_servers=param.init_servers, m_items=m_items)
        static_c.server_capacity = int(server_capacity)
        static_c.init_servers(preloaded_items=preloading_items.copy())
        record_circle_occupation(static_c, mult=mult, infin=infin)
    else:
        raise Exception("Invalid exp-type")

   # if infin:
    #    pl.plot_occ(cap=infin)
    #else:
     #   pl.plot_occ(cap=param.addit_augm)



def run_show_server_access():
    if param.exp_type == "temp":
        run_show_server_access_temp()
    else:
        run_show_server_access_ctr()


def run_show_server_access_ctr():
    param.show_server_occup = True
    param.unbounded_capacity = True
    param.set_n_servers(100)
    param.set_size_dataset(100000)
    jh.check_dataset()
    seq_data = fetch_ctr_seq_time_100k()
    seq = seq_data[0]
    sequence=seq_data[1]
    m_items = int(get_n_items(seq))
    print(f"Running on seq {seq}, # items: {m_items}, # req: {len(sequence)}")
    static_c = Circle(id=1, n_servers=param.init_servers, m_items=m_items)
    rand_server = random.randint(0, param.init_servers)
    addit_time_as_int = 1
    date_format = "%Y-%m-%d %H:%M:%S"
    curr_time = datetime.datetime.strptime(sequence[0][1], date_format)
    limit = 1000
    limit_copy = limit
    res = {}
    for r in sequence:
        if limit_copy == 0:
            break
        if static_c.get_item_position(r[0]) == rand_server:
            new_time = datetime.datetime.strptime(r[1], date_format)
            addit_time_as_int += (new_time - curr_time).total_seconds()
            res[addit_time_as_int] = r[0]
            curr_time = new_time
            limit_copy -= 1

    with open(f"results3/{param.hash_f}_lim{limit}_server_ctr_acc_overtime.json", "w") as fp:
        json.dump(res, fp)
    print(res)
    print("Plotting")
    pl.plot_server_acc(limit)


def run_show_server_access_temp():
    param.show_server_occup = True
    temp_ps=[param.temp_p]
    param.unbounded_capacity = True
    param.set_n_servers(100)
    param.set_n_items(10000)
    param.set_size_dataset(100000)
    jh.check_dataset()
    sequences = fetch_temp_seq()
    for p in temp_ps:
        param.temp_p = p
        for seq in sequences:
            res = {}
            sequence = list(sequences[seq].values())
            if not seq.startswith(p):
                continue
            init_time = "2006-03-01 00:01:13"
            date_format = "%Y-%m-%d %H:%M:%S"
            curr_time = datetime.datetime.strptime(init_time, date_format)
            m_items = int(get_n_items(seq))
            print(f"Running on seq {seq}, # items: {m_items}, # req: {len(sequence)}")
            static_c = Circle(id=1, n_servers=param.init_servers, m_items=m_items)
            rand_server = random.randint(0, param.init_servers)
            curr_time_as_int = 1
            limit = 100
            limit_copy = limit
            for r in sequence:
                if limit_copy == 0:
                    break
                if static_c.get_item_position(r) == rand_server:
                    res[curr_time_as_int] = r
                    limit_copy -= 1
                curr_time += datetime.timedelta(seconds=1)
                curr_time_as_int += 1
            with open(f"results/server_p{p}_lim{limit}_acc_overtime.json", "w") as fp:
                json.dump(res, fp)
            print(res)
            print("Plotting")
            pl.plot_server_acc(limit=limit)

def run_show_server_occ():
    param.show_server_occup = True
    temp_ps = ["0.15", "0.9"]
    param.unbounded_capacity = True
    param.set_n_servers(100)
    param.set_n_items(10000)
    param.set_size_dataset(100000)
    param.hash_f = "sha"
    jh.check_dataset()
    sequences = fetch_temp_seq()
    for p in temp_ps:
        param.temp_p = p
        for seq in sequences:
            res = {}
            sequence = list(sequences[seq].values())
            if not seq.startswith(p):
                continue
            init_time = "2006-03-01 00:01:13"
            date_format = "%Y-%m-%d %H:%M:%S"
            curr_time = datetime.datetime.strptime(init_time, date_format)
            m_items = int(get_n_items(seq))
            print(f"Running on seq {seq}, # items: {m_items}, # req: {len(sequence)}")
            static_c = Circle(id=1, n_servers=param.init_servers, m_items=m_items)
            present_items = set()
            rand_server = random.randint(0, param.init_servers)
            time_mark = curr_time+datetime.timedelta(minutes=60)
            mark_counter = 1
            for r in sequence:
                curr_time += datetime.timedelta(seconds=1)
                if static_c.get_item_position(r) == rand_server and r not in present_items:
                    present_items.add(r)
                if curr_time > time_mark:        # save # of items in server every ten seconds
                    res[mark_counter] = len(present_items)
                    time_mark += datetime.timedelta(minutes=60)
                    mark_counter += 1
            with open(f"results/server_p{p}_occ_overtime.json", "w") as fp:
                json.dump(res, fp)
            print(res)
    print("Plotting")
    pl.plot_server_occ()


def run_experiment_temp():
    """
    Run a temporal experiment for one given temporal parameter and one n_servers parameter
    """
    capacities = [param.addit_augm]
    server_n = [param.init_servers]
    tmp_p = param.temp_p
    res_dict = {}
    sequences = fetch_temp_seq()
    for seq in sequences:
        sequence = list(sequences[seq].values())
        if not seq.startswith(tmp_p+"_"):
            continue
        m_items = int(get_n_items(seq))
        print(f"Running on seq {seq}, # items: {m_items}, # req: {len(sequence)}")
        res_dict[seq] = {}
        for c in capacities:
            res_dict[seq][c] = {}
            for n in server_n:
                res_dict[seq][c][n] = {}
                n_servers = n
                if param.unbounded_capacity:
                    server_capacity = m_items * 2
                elif c == "*1.25":
                    server_capacity = m_items / n_servers * 1.25 * param.initial_occup_factor
                else:
                    server_capacity = int((m_items/n_servers + int(c))*param.initial_occup_factor)
                if param.algorithm == "AdjustHash" or param.algorithm == "All":
                    algo_c = Circle(id=0, n_servers=n_servers, m_items=m_items)
                    algo_c.server_capacity = int(server_capacity)
                #if param.algorithm == "Static" or param.algorithm == "All":
                 #   static_c = Circle(id=1, n_servers=n_servers, m_items=m_items)
                  #  static_c.server_capacity = int(server_capacity)       # "normal" static
                if param.algorithm == "WBL" or param.algorithm == "All":
                    wbl_c = Circle(id=2, n_servers=n_servers, m_items=m_items)
                    wbl_c.server_capacity = int(m_items / n_servers * 1.25 * param.initial_occup_factor)   # wbl
                print("Runnning for c=" + c + ", server_n="+ str(n)+", server capacity = " + str(server_capacity))
                logger = open("results/n"+str(param.init_servers)+"_h"+str(m_items)+"_m"+str(len(sequence))+
                              "_p"+tmp_p+"_s"+str(param.stale_time)+"_f"+str(param.serv_ins_freq)+"_res.txt", 'a')
                logger.write(f"Running on seq {seq}, # items: {m_items}, # req: {len(sequence)}, stale: {param.stale_time}"
                             f", serv_ins_freq: {param.serv_ins_freq}, serv_del_freq: {param.serv_del_freq}"
                             f", \n")
                logger.write("Starting at " + str(datetime.datetime.now())+"\n")
                logger.write(f"Additive capacity= {c} , server capacity= {server_capacity}, n_servers= {n_servers} \n")
                preloading_items = au.get_preloaded_items(sequence.copy())
                if param.algorithm == "AdjustHash" or param.algorithm == "All":
                    algo_c.init_servers(preloaded_items=preloading_items)
                    algo = Push_down_algo(algo_c)
                    if param.unbounded_capacity:
                        algo.serve_sequence_unbounded_cap(sequence=sequence.copy(), type="temp")
                        record_circle_occupation(algo_c)
                    else:
                        algo.serve_sequence(sequence=sequence.copy(), type="temp")
                    res_dict[seq][c][n]["algo_access"] = algo.access_cost
                    res_dict[seq][c][n]["algo_reconfig"] = algo_c.reconfig_cost
                    res_dict[seq][c][n]["algo_max_acc"] = algo.max_iteration
                    res_dict[seq][c][n]["algo_mean_serv_c"] = sum(algo_c.server_c_record) / len(algo_c.server_c_record)
                    logger.write(f"Algo access-cost: {algo.access_cost}\n"
                    f"Algo reconfig-cost: {algo_c.reconfig_cost}\n"
                    f"Items-stats: \n"
                    f"algo-deleted: {algo.sum_del_items}, "
                    f"algo-reinserted: {algo.sum_reinserted_items}\n")
                """if param.algorithm == "Static" or param.algorithm == "All":
                    static_c.init_servers(preloaded_items=preloading_items)
                    static_algo = Static_algo(static_c)
                    if param.unbounded_capacity:
                        static_algo.serve_sequence_unbounded_cap(sequence=sequence.copy(), type="temp")
                        record_circle_occupation(static_c)
                    else:
                        static_algo.serve_sequence(sequence=sequence.copy(), type="temp")
                    res_dict[seq][c][n]["static_access"] = static_algo.access_cost
                    res_dict[seq][c][n]["static_reconfig"] = static_c.reconfig_cost
                    logger.write(f"Static access-cost: {static_algo.access_cost} \n"
                     f"Static reconfig-cost: {static_c.reconfig_cost}\n"
                     f"Items-stats: \n"
                     f"static-deleted: {static_algo.sum_del_items}, "
                     f"static-reinserted: {static_algo.sum_reinserted_items}\n")
                    if param.unbounded_capacity and param.show_circle_occup:
                        record_circle_occupation(static_c)"""
                if param.algorithm == "WBL" or param.algorithm == "All":
                    wbl_c.init_servers(preloaded_items=preloading_items)
                    wbl_algo = Static_algo(wbl_c)
                    if param.unbounded_capacity:
                        wbl_algo.serve_sequence_unbounded_cap(sequence=sequence.copy(), type="temp")
                        record_circle_occupation(wbl_c)
                    else:
                        wbl_algo.serve_sequence(sequence=sequence.copy(), type="temp")
                    res_dict[seq][c][n]["wbl_access"] = wbl_algo.access_cost
                    res_dict[seq][c][n]["wbl_reconfig"] = wbl_c.reconfig_cost
                    res_dict[seq][c][n]["wbl_max_acc"] = wbl_algo.max_iteration
                    res_dict[seq][c][n]["wbl_mean_serv_c"] = sum(wbl_c.server_c_record) / len(wbl_c.server_c_record)
                    logger.write(f"wbl access-cost: {wbl_algo.access_cost} \n"
                     f"wbl reconfig-cost: {wbl_c.reconfig_cost}\n"
                     f"Items-stats: \n"
                     f"wbl-deleted: {wbl_algo.sum_del_items}, "
                     f"wbl-reinserted: {wbl_algo.sum_reinserted_items}\n")
                    if param.unbounded_capacity and param.show_circle_occup:
                        record_circle_occupation(wbl_c)
                logger.write("Finished at " + str(datetime.datetime.now())+"\n\n")
                logger.close()
        print(res_dict)
        dir = "../loben100k_results"
        path = (dir+"/n"+str(param.init_servers)+"_h"+str(m_items)+"_m"+str(len(sequence))+
                              "_p"+tmp_p+"_s"+str(param.stale_time)+"_f"+str(param.serv_ins_freq)
                                     +"_alg"+param.algorithm+"_results.json")
        if os.path.exists(path):
                dir = "../"
        jh.add_temp_res_to_json(path=dir+"/n"+str(param.init_servers)+"_h"+str(m_items)+"_m"+str(len(sequence))+
                              "_p"+tmp_p+"_s"+str(param.stale_time)+"_f"+str(param.serv_ins_freq)
                                     +"_alg"+param.algorithm+"_results.json", data=res_dict)

def record_circle_occupation(circle, mult=False, infin=False):
    occ = {}
    s = circle.root
    sum = 0
    for n in range(0, circle.n_servers):
        #print(f"Looking at server {s.id}")
        occ[s.id] = s.get_current_occupation()
        sum += s.get_current_occupation()
        s = s.child_pointer
    if param.exp_type == "temp":
        if param.unbounded_capacity:
            with open(f"results/circle_inf_h{param.hash_f}_p{param.temp_p}_occupation.json", "w") as fp:
                json.dump(occ, fp)
                print(f"Circle counts {sum} items")
        else:
            if mult:
                with open(f"results/circle_1.25_h{param.hash_f}_p{param.temp_p}_occupation.json", "w") as fp:
                    json.dump(occ, fp)
                    print(f"Circle counts {sum} items")
            elif infin:
                with open(f"results/circle_infin_h{param.hash_f}_p{param.temp_p}_occupation.json", "w") as fp:
                    json.dump(occ, fp)
                    print(f"Circle counts {sum} items")
            else:
                with open(f"results/circle_c{param.addit_augm}_h{param.hash_f}_p{param.temp_p}_occupation.json", "w") as fp:
                    json.dump(occ, fp)
                    print(f"Circle counts {sum} items")
    elif param.exp_type == "ctr":
        if param.unbounded_capacity:
            with open(f"results/circle_inf_h{param.hash_f}_ctr_occupation.json", "w") as fp:
                json.dump(occ, fp)
                print(f"Circle counts {sum} items")
        else:
            if mult:
                with open(f"results/circle_1.25_h{param.hash_f}_ctr_occupation.json", "w") as fp:
                    json.dump(occ, fp)
                    print(f"Circle counts {sum} items")
            elif infin:
                with open(f"results/circle_infin_h{param.hash_f}_ctr_occupation.json", "w") as fp:
                    json.dump(occ, fp)
                    print(f"Circle counts {sum} items")
            else:
                with open(f"results/circle_c{param.addit_augm}_h{param.hash_f}_ctr_occupation.json", "w") as fp:
                    json.dump(occ, fp)
                    print(f"Circle counts {sum} items")

def run_experiment_ctr():
    seq_data = fetch_ctr_seq_time_100k()
    seq_name = seq_data[0]
    sequence = seq_data[1]
    capacities = param.addit_augm
    server_n = [param.init_servers]
    m_items = int(get_n_items(seq_name))
    print("m_items = " + str(m_items))
    res_dict = {}
    logger = open("results/ctr_results.txt", 'a')
    for c in capacities:
        res_dict[c] = {}
        for n in server_n:
            res_dict[c][n] = {}
            n_servers = n
            algo_c = Circle(id=0, n_servers=n_servers, m_items=m_items)  # algo_c = 10,000 req
            static_c = Circle(id=1, n_servers=n_servers, m_items=m_items)
            wbl_c = Circle(id=2, n_servers=n_servers, m_items=m_items)
            if param.unbounded_capacity:
                server_capacity = m_items * 2
            elif c == "*1.25":
                server_capacity = m_items / n_servers * 1.25 * param.initial_occup_factor
            else:
                server_capacity = (m_items / n_servers + int(c)) * param.initial_occup_factor
            algo_c.server_capacity = int(server_capacity)
            static_c.server_capacity = int(server_capacity)
            wbl_c.server_capacity = int(m_items / n_servers * 1.25 * param.initial_occup_factor)
            print("Runnning for c=" + c + ", server_n="+ str(n)+", server capacity = " + str(algo_c.server_capacity))
            logger.write(f"\nRunning for c = {c} , server capacity = {algo_c.server_capacity}, n_servers = {n_servers} \n")
            logger.write("Starting at " + str(datetime.datetime.now()) + "\n")
            preloading_items = au.get_preloaded_items(sequence.copy())
            algo_c.init_servers(preloaded_items=preloading_items)
            algo = Push_down_algo(algo_c)
            algo.serve_sequence(sequence.copy(), type="ctr")
            res_dict[c][n]["algo_access"] = algo.access_cost
            res_dict[c][n]["algo_max_acc"] = algo.max_iteration
            res_dict[c][n]["algo_mean_serv_c"] = sum(algo_c.server_c_record) / len(algo_c.server_c_record)

            #static_c.init_servers(preloaded_items=preloading_items)
            #static_algo = Static_algo(static_c)
            #static_algo.serve_sequence(sequence.copy(), type="ctr")
            #res_dict[c][n]["static_access"] = static_algo.access_cost

            wbl_c.init_servers(preloaded_items=preloading_items)
            wbl_algo = Static_algo(wbl_c)
            wbl_algo.serve_sequence(sequence.copy(), type="ctr")
            res_dict[c][n]["wbl_access"] = wbl_algo.access_cost
            res_dict[c][n]["wbl_max_acc"] = wbl_algo.max_iteration
            res_dict[c][n]["wbl_mean_serv_c"] = sum(wbl_c.server_c_record) / len(wbl_c.server_c_record)
            logger.write(f"Algo: {algo.access_cost}, "
                         #f", Static: {static_algo.access_cost}, "
                         f"WBL: {wbl_algo.access_cost}\n")
            logger.write("Finished at " + str(datetime.datetime.now()) + "\n")
    print(res_dict)
    logger.close()
    with open(f"results/ctr_results_n{param.init_servers}_f{param.serv_ins_freq}_s{param.stale_time}.json", "w") as fp:
        json.dump(res_dict, fp)