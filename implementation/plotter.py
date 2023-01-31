import collections
import os

import matplotlib as mpl
from matplotlib import rc
import numpy as np
from matplotlib import cm
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FuncFormatter

from params import param

def sum_results_all_ps():
    """
    Take the result-files of different temporal parameters and put them in one file
    Pick one server-n parameter which got to be present in all single files
    This one will serve as basis for the 3d plotting all p's.
    """
    files = os.listdir("results")
    sel_files = []
    for f in files:
        if f.startswith(f"n{int(param.init_servers)}") and f.endswith(".json")\
                and not f.endswith("all_tmp_results.json"):
            sel_files.append(f)
    curr_data = {}
    if os.path.exists(f"results/n{param.init_servers}_h{param.n_items}_m"
                  f"{param.size_dataset}_all_tmp_results.json"):
        with open(f"results/n{param.init_servers}_h{param.n_items}_m"
                  f"{param.size_dataset}_all_tmp_results.json", "r") as fp:
            curr_data = json.load(fp)
    new_dict = curr_data
    for f in sel_files:
        with open("results/"+f, "r") as fr:
            data = json.load(fr)
            print(f"Loading data from {data}")
            for p in data:      # take all temp-param, just # servers is fixed to given param
                if p[3] == '_':     # case of 0.9, 0.3, ...
                    if p[:3] not in new_dict:
                        new_dict[p[:3]] = {}
                    for ad in data[p]:
                        new_dict[p[:3]][str(ad)] = data[p][str(ad)][str(param.init_servers)]
                else:               # 0.75, 0.15, ...
                    if p[:4] not in new_dict:
                        new_dict[p[:4]] = {}
                    for ad in data[p]:
                        new_dict[p[:4]][str(ad)] = data[p][str(ad)][str(param.init_servers)]
    print(new_dict)
    sorted_dict_keys = list(reversed(sorted(new_dict)))
    saving_dict = {}
    for key in sorted_dict_keys:
        saving_dict[key] = new_dict[key]
    print(saving_dict)
    with open("results/n" + str(param.init_servers) + "_h" + str(param.n_items)
              + "_m" + str(param.size_dataset) + "_all_tmp_results.json", "w") as fw:
        json.dump(new_dict, fw)

def plot_2d_all_ps_fixed_sever_n():
    sum_results_all_ps()
    with open(f"results/n{param.init_servers}_h{param.n_items}_m"
              f"{param.size_dataset}_all_tmp_results.json", "r") as fp:
        data = json.load(fp)
    plot_data = []
    for p in data:
        #if len(data[p]) > 1:
         #   raise Exception("For this plot we need only one augmentation parameter")
        for c in data[p]:
            if str(c) == param.addit_augm:
                for algo in data[p][c]:
                    if algo == "algo_access":
                        plot_data.append(["H&A", float(p), data[p][c][algo]/param.n_items])
                    elif algo == "wbl_access":
                        plot_data.append(["WBL", float(p), data[p][c][algo]/param.n_items])
                    #elif algo == "static_access":
                     #   plot_data.append(["Static_augm", float(p), data[p][c][algo]/param.n_items])
    # print(data)
    df = pd.DataFrame(plot_data, columns=['name', 'p', 'cost'])
    print(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xticks(df.p.unique())
    fs = 20
    sns.lineplot(x='p', y='cost', data=df, hue='name')
    #plt.legend(bbox_to_anchor=(1, 0.3), loc='lower right', fontsize=fs)
    plt.ylabel('Average Cost per Item', fontsize=fs)
    plt.xlabel('Temporal Locality', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    #leg = ax.legend(bbox_to_anchor=(1, 0.3), loc='lower right', fontsize=fs, ncol=1)
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0.5, 0.), loc='lower right', fontsize=fs, ncol=2)
    leg_lines = leg.get_lines()
    # print(leg_lines)
    for line in range(0, len(ax.lines)):

        if line == 1:
            ax.lines[line].set_color('#FFB570')
            leg_lines[line].set_color('#FFB570')
        if line == 0:
            ax.lines[line].set_color('#67AB9F')
            leg_lines[line].set_color('#67AB9F')
    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(reversed(handles), reversed(labels), bbox_to_anchor=(1, 0.4), loc='lower right', fontsize=fs)
    plt.savefig(f"plots/2d_all_ps_n{param.n_items}_m{param.size_dataset}_access.pdf", bbox_inches='tight')
    plt.show()


def plot_2d_all_ps_fixed_sever_n_more_sim():
    """
    Take the results of more simulations on temporal data and plot mean
    """
    files = os.listdir("results8")
    wbl_tot = {}
    algo_tot = {}

    for f in files:
        with open("results8/" + f, 'r') as fr:
            data = json.load(fr)
        for p in data:
            algo_tot[p] = 0
            wbl_tot[p] = 0
            for c in data[p]:
                if str(c) == param.addit_augm:
                    for algo in data[p][c]:
                        if algo == "algo_access":
                            algo_tot[p] += data[p][c][algo] / param.n_items
                            #plot_data.append(["H&A", float(p), data[p][c][algo]/param.n_items])
                        elif algo == "wbl_access":
                            wbl_tot[p] += data[p][c][algo] / param.n_items
                            #plot_data.append(["WBL", float(p), data[p][c][algo]/param.n_items])
    plot_data = []
    for p in algo_tot:
        plot_data.append(["H&A", float(p), algo_tot[p]/ len(files)]) #
        plot_data.append(["WBL", float(p), wbl_tot[p] / len(files)]) #
    df = pd.DataFrame(plot_data, columns=['name', 'p', 'cost'])
    print(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xticks(df.p.unique())
    fs = 20
    sns.lineplot(x='p', y='cost', data=df, hue='name')
    #plt.legend(bbox_to_anchor=(1, 0.3), loc='lower right', fontsize=fs)
    plt.ylabel('Average Cost Per Item', fontsize=fs)
    plt.xlabel('Temporal Locality', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    leg = ax.legend(bbox_to_anchor=(0.5, 0), loc='lower right', fontsize=fs, ncol=2)
    leg_lines = leg.get_lines()
    # print(leg_lines)
    for line in range(0, len(ax.lines)):
        if line == 1:  # 1.25
            ax.lines[line].set_linestyle(':')
            ax.lines[line].set_linewidth(5)
            ax.lines[line].set_color('#FFB570')  # orange
            leg_lines[line].set_color('#FFB570')  # green
            leg_lines[line].set_linestyle(':')
            leg_lines[line].set_linewidth(5)

        if line == 0:  # c2
            ax.lines[line].set_linestyle('-')
            ax.lines[line].set_linewidth(5)
            ax.lines[line].set_color('#67AB9F')  # green
            ax.lines[line].set_alpha(0.5)
            leg_lines[line].set_color('#67AB9F')  # green
            leg_lines[line].set_alpha(0.5)
            leg_lines[line].set_linestyle('-')
            leg_lines[line].set_linewidth(5)
    plt.savefig(f"plots/2d_nsim{len(files)}_all_ps_n{param.n_items}_m{param.size_dataset}_access.pdf", bbox_inches='tight')
    plt.show()

def plot_3d_all_ps_fixed_server_n():
    print("Plotting for all temp-ps")
    sum_results_all_ps()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    with open(f"results/n{param.init_servers}_h{param.n_items}_m"
              f"{param.size_dataset}_all_tmp_results.json", "r") as fp:
        data = json.load(fp)
    p_desc = []
    ps = []
    cs = []
    # following x-coord (capacity)
    static_acc = {}
    algo_acc = {}
    static_rec = {}
    algo_rec = {}
    for p in data:
        if str(p) not in p_desc:
            ps.append(float(p))
            p_desc.append(str(p))
        sorting_cs = {int(k) : v for k, v in data[p].items()}
        sorted_dict = collections.OrderedDict(sorted(sorting_cs.items()))
        for c in sorted_dict:
            if c not in static_acc:
                static_acc[c] = []
                algo_acc[c] = []
                static_rec[c] = []
                algo_rec[c] = []
            if int(c) not in cs:
                cs.append(int(c))
            #algo[c].append(data[p][c]["algo_access"])
            #static[c].append(data[p][c]["static_access"])
            algo_acc[c].append(sorted_dict[c]["algo_access"])
            static_acc[c].append(sorted_dict[c]["static_access"])
            algo_rec[c].append(sorted_dict[c]["algo_reconfig"])
            static_rec[c].append(sorted_dict[c]["static_reconfig"])
    #cs = sorted(cs)
    x = ps
    y = cs
    print(x)
    print(y)
    X, Y = np.meshgrid(x, y)
    ax.set_xlabel('temp_p')
    ax.set_ylabel('augm_cap')
    ax.set_zlabel('cost', rotation='vertical')
    #ax.set_xticks(x)
    ax.set_xticks(x)
    ax.set_yticks(y)

    x_descr = p_desc
    #x_descr = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    y_descr = cs
    ax.set_yticklabels(y_descr, fontsize = 10)
    ax.set_xticklabels(x_descr, fontsize = 10)
    Z = np.array(Y)

    j = 0
    for n in algo_acc:
        print("Filling algo-Z with " + str(n) )
        for i in range(0, len(algo_acc[n])):
            Z[j][i] = algo_acc[n][i]
        j+=1

    print(X)
    print(Y)
    print(Z)

    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.Blues(norm(Z))
    rcount, ccount, _ = colors.shape
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                           facecolors=colors, shade=False, label="algo")
    surf.set_facecolor((0.0, 0, 0.8, 0.6))
    ax.view_init(15, 40)

    # second plane
    j = 0
    for n in static_acc:
        print("Filling static-Z with " + str(n))
        for i in range(0, len(static_acc[n])):
            Z[j][i] = static_acc[n][i]
        j += 1
    print(Z)
    colors = cm.Reds(norm(Z))
    rcount, ccount, _ = colors.shape
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                           facecolors=colors, shade=False, label="static")
    surf.set_facecolor((0.5, 0, 0.0, 0.6))
    ax.view_init(15, 40)
    fake2Dline = mpl.lines.Line2D([0], [0], linestyle="none", c='r', marker='o')
    fake2Dline2 = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    ax.legend([fake2Dline, fake2Dline2], ['static', 'algo'], numpoints=1, bbox_to_anchor=(1, 0.75))
    #ax.legend()
    if param.exp_type == "temp":
        plt.savefig("plots/3d_all_ps_access_augm.pdf", bbox_inches='tight')
    plt.show()


    # reconfig plot
    print("Plotting reconfig cost")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x = ps
    y = cs
    print(x)
    print(y)
    X, Y = np.meshgrid(x, y)
    ax.set_xlabel('temp_p')
    ax.set_ylabel('augm_cap')
    ax.set_zlabel('cost', rotation='vertical')
    # ax.set_xticks(x)
    ax.set_xticks(x)
    ax.set_yticks(y)

    x_descr = p_desc
    # x_descr = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    y_descr = cs
    ax.set_yticklabels(y_descr, fontsize=10)
    ax.set_xticklabels(x_descr, fontsize=10)
    Z = np.array(Y)
    j = 0
    for n in algo_acc:
        print("Filling algo-Z with " + str(n))
        for i in range(0, len(algo_rec[n])):
            Z[j][i] = algo_rec[n][i]
        j += 1

    print(X)
    print(Y)
    print(Z)

    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.Blues(norm(Z))
    rcount, ccount, _ = colors.shape
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                           facecolors=colors, shade=False, label="algo")
    surf.set_facecolor((0.0, 0, 0.8, 0.6))
    ax.view_init(15, 40)

    # second plane
    j = 0
    for n in static_acc:
        print("Filling static-Z with " + str(n))
        for i in range(0, len(static_rec[n])):
            Z[j][i] = static_rec[n][i]
        j += 1
    print(Z)
    colors = cm.Reds(norm(Z))
    rcount, ccount, _ = colors.shape
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                           facecolors=colors, shade=False, label="static")
    surf.set_facecolor((0.5, 0, 0.0, 0.6))
    ax.view_init(15, 40)
    fake2Dline = mpl.lines.Line2D([0], [0], linestyle="none", c='r', marker='o')
    fake2Dline2 = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    ax.legend([fake2Dline, fake2Dline2], ['static', 'algo'], numpoints=1, bbox_to_anchor=(1, 0.75))
    # ax.legend()
    if param.exp_type == "temp":
        plt.savefig("plots/3d_all_ps_reconfig_augm.pdf", bbox_inches='tight')
    plt.show()

def plot_3d_single_p():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #df = pd.read_csv("data/data.csv")
    with open(f"results/n{param.init_servers}_h{param.n_items}_m{param.size_dataset}_p{param.temp_p}_results.json", "r") as fp:
        data = json.load(fp)
    capacities_desc = []
    capacities = []
    n_servers = []
    # following x-coord (capacity)
    static = {}
    algo = {}
    for trace in data:
        for c in data[trace]:
            if str(c) not in capacities_desc:
                if c == '*2':
                     capacities.append(20)
                else:
                    capacities.append(int(c))
                capacities_desc.append(str(c))

            for n in data[trace][c]:
                if n not in static:
                    static[n] = []
                    algo[n] = []
                if int(n) not in n_servers:
                    n_servers.append(int(n))
                algo[n].append(data[trace][c][n]["algo_access"])
                static[n].append(data[trace][c][n]["static_access"])
    x = capacities
    y = n_servers
    print(x)
    print(y)
    X, Y = np.meshgrid(x, y)
    ax.set_xlabel('capacity')
    ax.set_ylabel('n_servers')
    ax.set_zlabel('cost', rotation='vertical')
    ax.set_xticks(x)
    ax.set_yticks(y)

    x_descr = capacities_desc
    #x_descr = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    # y_descr = [16, 14, 12, 10, 8, 6, 4, 2]
    y_descr = n_servers
    ax.set_yticklabels(y_descr, fontsize = 10)
    ax.set_xticklabels(x_descr, fontsize = 10)
    Z = np.array(Y)

    j = 0
    for n in algo:
        print("Filling algo-Z with " + n )
        for i in range(0, len(algo[n])):
            Z[j][i] = algo[n][i]
        j+=1

    print(X)
    print(Y)
    print(Z)

    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.Blues(norm(Z))
    rcount, ccount, _ = colors.shape
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                           facecolors=colors, shade=False, label="algo")
    surf.set_facecolor((0.0, 0, 0.8, 0.6))
    ax.view_init(15, 40)

    # second plane
    j = 0
    for n in static:
        print("Filling static-Z with " + n)
        for i in range(0, len(static[n])):
            Z[j][i] = static[n][i] + 10000
        j += 1
    print(Z)
    colors = cm.Reds(norm(Z))
    rcount, ccount, _ = colors.shape
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount,
                           facecolors=colors, shade=False, label="static")
    surf.set_facecolor((0.5, 0, 0.0, 0.6))
    ax.view_init(15, 40)
    fake2Dline = mpl.lines.Line2D([0], [0], linestyle="none", c='r', marker='o')
    fake2Dline2 = mpl.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    ax.legend([fake2Dline, fake2Dline2], ['static', 'algo'], numpoints=1)
    if param.exp_type == "temp":
        plt.savefig("plots/3d_tmp_plot_augm.pdf", bbox_inches='tight')
    else:
        plt.savefig("plots/3d_plot_augm.pdf", bbox_inches='tight')
    plt.show()

def line_plot_freq():
    """
    Plots data according to server insertion/deletion parameters
    Plot will search for data for init_servers- and stale_time- parameters given on param.py

    data_format = [ ['Algo', '0', 121], ['Static', '0', 117],
            ['Algo', '50', 103], ['Static', '50', 117] ]
    """
    data = []
    files = os.listdir("results")
    capacity = param.addit_augm
    n_servers = param.init_servers
    for f in files:
        if not f.endswith("all_tmp_results.json"):
            if f.startswith(f"n{int(param.init_servers)}") and f.endswith("_results.json") \
                    and get_stale(f) == str(param.stale_time):
                with open("results/"+f, 'r') as fr:
                    single_res = json.load(fr)
                for trace in single_res:
                    if capacity in single_res[trace]:
                        data.append(['Algo', int(get_freq(f)), single_res[trace][capacity][str(n_servers)]['algo_access']])
                        #data.append(['Static', int(get_freq(f)), single_res[trace][capacity][str(n_servers)]['static_access']])
                        data.append(
                            ['Wbl', int(get_freq(f)), single_res[trace][capacity][str(n_servers)]['wbl_access']])
    df = pd.DataFrame(data, columns=['name', 'freq', 'cost'])
    print(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xticks(df.freq.unique())
    fs = 20
    sns.lineplot(x='freq', y='cost', data=df, hue='name')
    plt.legend(bbox_to_anchor=(1, 0.3), loc='lower right', fontsize=fs)
    plt.ylabel('access cost', fontsize=fs)
    plt.xlabel('server ins/del freq.', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    if param.exp_type == "temp":
        plt.savefig(f"plots/freq_line_n{n_servers}_c{capacity}_p{param.temp_p}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"plots/freq_line_n{n_servers}_c{capacity}_ctr.pdf", bbox_inches='tight')
    plt.show()


def line_plot_stale():
    """
    Plots data according to server stale_time parameters
    Plot will search for data for init_servers- and server ins/del-frequency- parameters given on param.py

    data_format = [ ['Algo', '0', 121], ['Static', '0', 117],
            ['Algo', '200', 103], ['Static', '200', 117] ]
    """
    data = []
    files = os.listdir("results")
    capacity = param.addit_augm
    n_servers = param.init_servers
    for f in files:
        if param.exp_type == "temp" and not f.endswith("all_tmp_results.json"):
            if f.startswith(f"n{int(param.init_servers)}") and f.endswith("_results.json") \
                    and get_freq(f) == str(param.serv_ins_freq):
                with open("results/" + f, 'r') as fr:
                    single_res = json.load(fr)
                for trace in single_res:
                    print(single_res[trace][capacity])
                    data.append(['Algo', int(get_stale(f)), single_res[trace][capacity][str(n_servers)]['algo_access']])
                    data.append(
                        ['Wbl', int(get_stale(f)), single_res[trace][capacity][str(n_servers)]['wbl_access']])
        elif param.exp_type == "ctr" and f.__contains__("ctr"):
            if f.__contains__(f"n{int(param.init_servers)}"):
                with open("results/" + f, 'r') as fr:
                    single_res = json.load(fr)
                    print(single_res[capacity])
                    data.append(['H&A', int(get_stale(f)), single_res[capacity][str(n_servers)]['algo_access']/1029,
                                     single_res[capacity][str(n_servers)]['algo_mean_serv_c'], 2])
                    data.append(
                        ['WBL', int(get_stale(f)), single_res[capacity][str(n_servers)]['wbl_access']/1029,
                                    single_res[capacity][str(n_servers)]['wbl_mean_serv_c'], 0.5])
    df = pd.DataFrame(data, columns=['name', 'stale', 'cost', 'mean_server_c', 'size_p'])
    print(df)
    """rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xticks(df.stale.unique())
    fs = 25
    sns.lineplot(x='stale', y='cost', hue='name', data=df)
    ax2 = plt.twinx()
    sns.scatterplot(x='stale', y='mean_server_c', s=200 * df['size_p'], hue='name', data=df, ax=ax2, legend=False,
                    palette=['#67AB9F', '#FFB570'])
    ax.set_ylabel('Average Cost per Item', fontsize=fs)
    ax.set_xlabel('Stale Time', fontsize=fs)
    ax2.set_ylabel('Mean Server Capacity', fontsize=fs)
    # plt.ylabel('Average Cost per Item', fontsize=fs)
    # plt.xlabel('Number of Servers', fontsize=fs)
    # plt.tick_params(axis='both', which='major', labelsize=fs)
    x_labels = ax.get_xticks()
    ax.set_xticklabels(labels=x_labels, fontsize=fs)
    # y1_labels = ax.get_yticks()
    # y2_labels = ax2.get_yticks()
    # ax.set_yticklabels(labels=y1_labels, fontsize=fs)
    # ax2.set_yticklabels(labels=y2_labels, fontsize=fs)

    # change labels to be integer
    yint = []
    labels = ax.get_yticks()
    for each in labels:
        yint.append(int(each))
    ax.set_yticklabels(labels=yint, fontsize=fs)
    yint = []
    labels = ax2.get_yticks()
    for each in labels:
        yint.append(int(each))
    ax2.set_yticklabels(labels=yint, fontsize=fs)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(bbox_to_anchor=(0.62, 0.8), loc='lower right', fontsize=fs, ncol=2)
    leg_lines = leg.get_lines()
    for line in range(0, len(ax.lines)):
        if line == 1:  # 1.25
            ax.lines[line].set_linestyle(':')
            ax.lines[line].set_linewidth(5)
            ax.lines[line].set_color('#FFB570')  # orange
            leg_lines[line].set_color('#FFB570')
            leg_lines[line].set_linestyle(':')
            leg_lines[line].set_linewidth(5)

        if line == 0:  # c2
            ax.lines[line].set_linestyle('-')
            ax.lines[line].set_linewidth(5)
            ax.lines[line].set_color('#67AB9F')  # green
            ax.lines[line].set_alpha(0.5)
            leg_lines[line].set_color('#67AB9F')  # green
            leg_lines[line].set_alpha(0.5)
            leg_lines[line].set_linestyle('-')
            leg_lines[line].set_linewidth(5)
    #plt.ylabel('Average Cost per Item', fontsize=fs)
    #plt.xlabel('Stale Time', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs)

    if param.exp_type == "temp":
        plt.savefig(f"plots/stale_line_n{n_servers}_c{capacity}_p{param.temp_p}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"plots/stale_line_n{n_servers}_c{capacity}_ctr.pdf", bbox_inches='tight')
    plt.show()


def line_plot_n_servers():
    """
    Plots data according to init_server-parameters
    Plot will search for data for stale time and server ins/del-frequency- parameters given on param.py

    data_format = [ ['Algo', '0', 121], ['Static', '0', 117],
            ['Algo', '100', 103], ['Static', '100', 117] ]
    """
    data = []
    files = os.listdir("results")
    capacity = param.addit_augm
    for f in files:
        if param.exp_type=="temp" and f.endswith("_results.json") and not f.endswith("all_tmp_results.json"):
            if get_freq(f) == str(param.serv_ins_freq) and get_stale(f) == str(param.stale_time):
                with open("results/" + f, 'r') as fr:
                    single_res = json.load(fr)
            for trace in single_res:
                for n_servers_str in single_res[trace][capacity]:
                    n_servers = int(n_servers_str)
                    print(single_res[trace][capacity])
                    if param.exp_type == "AdjustHash" or param.algorithm == "All":
                        data.append(['H&A', n_servers, single_res[trace][capacity][str(n_servers)]['algo_access']])
                    if param.exp_type == "WBL" or param.algorithm == "All":
                        data.append(['WBL', n_servers, single_res[trace][capacity][str(n_servers)]['wbl_access']])
        elif param.exp_type == "ctr" and f.__contains__("ctr"):
            if get_stale(f) == str(param.stale_time):
                with open("results/" + f, 'r') as fr:
                    single_res = json.load(fr)
                for n_servers_str in single_res[capacity]:
                    n_servers = int(n_servers_str)
                    print(single_res[capacity])
                    if param.exp_type == "AdjustHash" or param.algorithm == "All":
                        data.append(['H&A', n_servers, single_res[capacity][str(n_servers)]['algo_access']/1029,
                                     single_res[capacity][str(n_servers)]['algo_max_acc'], 2])
                    if param.exp_type == "WBL" or param.algorithm == "All":
                        data.append(['WBL', n_servers, single_res[capacity][str(n_servers)]['wbl_access']/1029,
                                    single_res[capacity][str(n_servers)]['wbl_max_acc'], 0.5])
    df = pd.DataFrame(data, columns=['name', 'n_servers', 'cost', 'max_cost', 'size_p'])
    print(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xticks(df.n_servers.unique())
    fs = 25
    """mpl.rcParams.update(mpl.rcParamsDefault)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })"""
    sns.lineplot(x='n_servers', y='cost', hue='name', data=df)
    #ax2 = plt.twinx()
    #sns.scatterplot(x='n_servers', y='max_cost', s=100*df['size_p'], hue='name', data=df, ax=ax2, legend=False,
                    #palette=['#FFB570', '#67AB9F'])
    ax.set_ylabel('Average Cost per Item', fontsize=fs)
    ax.set_xlabel('Number of Servers', fontsize=fs)
    #ax2.set_ylabel('Max. Access Cost', fontsize=fs)

    # plt.legend(bbox_to_anchor=(1, 0.75), loc='lower right', fontsize=fs)
    #plt.ylabel('Average Cost per Item', fontsize=fs)
    #plt.xlabel('Number of Servers', fontsize=fs)
    #plt.tick_params(axis='both', which='major', labelsize=fs)

    x_labels = ax.get_xticks()
    ax.set_xticklabels(labels=x_labels, fontsize=fs)

    #y1_labels = ax.get_yticks()
    #y2_labels = ax2.get_yticks()
    #ax.set_yticklabels(labels=y1_labels, fontsize=fs)
    #ax2.set_yticklabels(labels=y2_labels, fontsize=fs)

    # change labels to be integer
    yint = []
    labels = ax.get_yticks()
    for each in labels:
        yint.append(int(each))
    ax.set_yticklabels(labels=yint, fontsize=fs)
    yint = []
    #labels = ax2.get_yticks()
    #for each in labels:
     #   yint.append(int(each))
    #ax2.set_yticklabels(labels=yint, fontsize=fs)

    # add legend
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(bbox_to_anchor=(1, 0.8), loc='lower right', fontsize=fs, ncol=2)
    #leg = ax.legend(bbox_to_anchor=(1, 0.8), loc='lower right', fontsize=fs, ncol=2)
    leg_lines = leg.get_lines()
    for line in range(0, len(ax.lines)):
        if line == 1:  # 1.25
            ax.lines[line].set_linestyle(':')
            ax.lines[line].set_linewidth(5)
            ax.lines[line].set_color('#FFB570')  # orange
            leg_lines[line].set_color('#FFB570')  # green
            leg_lines[line].set_linestyle(':')
            leg_lines[line].set_linewidth(5)

        if line == 0:  # c2
            ax.lines[line].set_linestyle('-')
            ax.lines[line].set_linewidth(5)
            ax.lines[line].set_color('#67AB9F')  # green
            ax.lines[line].set_alpha(0.5)
            leg_lines[line].set_color('#67AB9F')  # green
            leg_lines[line].set_alpha(0.5)
            leg_lines[line].set_linestyle('-')
            leg_lines[line].set_linewidth(5)
    if param.exp_type == "temp":
        plt.savefig(f"plots/n_servers_p{param.temp_p}_c{capacity}_f{param.serv_ins_freq}_s"
                    f"{param.stale_time}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"plots/n_servers_ctr_c{capacity}_f{param.serv_ins_freq}_s"
                    f"{param.stale_time}.pdf", bbox_inches='tight')
    plt.show()


def plot_infin_cap_mtf():
    data = []
    files = os.listdir("results")
    capacity = param.addit_augm
    for f in files:
        if f.endswith("_results.json") and not f.endswith("all_tmp_results.json"):
            if get_freq(f) == str(param.serv_ins_freq) and get_stale(f) == str(param.stale_time):
                with open("results/" + f, 'r') as fr:
                    single_res = json.load(fr)
                for trace in single_res:
                    for n_servers_str in single_res[trace][capacity]:
                        n_servers = int(n_servers_str)
                        print(single_res[trace][capacity])
                        if param.exp_type == "AdjustHash" or param.algorithm == "All":
                            data.append(['Algo', float(trace[:3]), single_res[trace][capacity][str(n_servers)]['algo_access']])
                        if param.exp_type == "Static" or param.algorithm == "All":
                            data.append(
                                ['Static', float(trace[:3]), single_res[trace][capacity][str(n_servers)]['static_access']])
                        if param.exp_type == "WBD" or param.algorithm == "All":
                            data.append(
                                ['Wbd', float(trace[:3]), single_res[trace][capacity][str(n_servers)]['wbd_access']])
    df = pd.DataFrame(data, columns=['name', 'p', 'cost'])
    print(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xticks(df.p.unique())
    fs = 20
    sns.lineplot(x='p', y='cost', hue='name', data=df)
    plt.legend(bbox_to_anchor=(1, 0.75), loc='lower right', fontsize=fs)
    plt.ylabel('access cost', fontsize=fs)
    plt.xlabel('p', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    if param.exp_type == "temp":
        plt.savefig(f"plots/inf_cap_s"
                    f"{param.stale_time}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"plots/n_servers_ctr_c{capacity}_f{param.serv_ins_freq}_s"
                    f"{param.stale_time}.pdf", bbox_inches='tight')
    plt.show()


def plot_server_occ():
    files = os.listdir("results")
    for f in files:
        if f.endswith("_occ_overtime.json"):
            with open("results/" + f, 'r') as fr:
                single_res = json.load(fr)
            plt.bar(single_res.keys(), single_res.values(), width=1, color='#009988')
            plt.savefig(f"plots/n{param.n_items}_server_occ_p{get_p_occ(f)}.pdf", bbox_inches='tight')


def plot_server_acc(limit):
    fig, ax = plt.subplots(figsize=(10, 6))
    files = os.listdir("results3")
    if len(files) > 1:
        raise Exception("For this particular plot, please make sure "
                        "there is only one interested result-file in results-directory")
    for f in files:
        if f.endswith("_acc_overtime.json"):
            with open("results3/" + f, 'r') as fr:
                single_res = json.load(fr)
            if param.exp_type == "temp":
                plt.scatter(x=single_res.keys(), y=single_res.values(), c=list(single_res.values()), cmap='brg')
            else:   # to plot ctr, transform string values to integers
                seq_as_list = list(single_res.values())
                int_dict = dict([(y,x+1) for x,y in enumerate(sorted(set(seq_as_list)))])
                print(int_dict)
                for x in single_res:
                    val = single_res[x]
                    single_res[x] = int(int_dict[val])
                print(single_res)
                plotting_dict = {int(float(k)): int(v) for k, v in single_res.items()}
                plt.scatter(x=plotting_dict.keys(), y=plotting_dict.values(), c=list(single_res.values()), cmap='brg')
            #plotting_dict = {int(float(k)):v for k,v in single_res.items()}
            key_list = []
            x_ax_lim = 0.0
            fs = 20
            for key in single_res.keys():
                if int(float(key)) > x_ax_lim:
                    key_list.append(x_ax_lim)
                    x_ax_lim += 86400
            #ax = plt.gca()
            plt.xticks(key_list)
            ax.set_xticklabels([int(x/60/24) for x in key_list])
            #for y in y_ticks:
             #   y_ax_desc.append(int(y))
            #ax.set_yticklabels(y_ax_desc)
            #ax.set_yticklabels([int(y) for y in plotting_dict.values()])
            #ax.set_xticks(key_list)
            #ax.set_xticks(ax.get_xticks()[::50])
            #for key in plotting_dict.keys():
             #   key_list.append(key)
            #print(plotting_dict)
            #plt.xticks(np.arange(0, 250000, 50000))
            #plt.xticks(list(range(1,max(key_list)+10000)),[str(i) for i in range(1,max(key_list)+10000)])
            #plt.xticks(key_list)#range(min(plotting_dict.keys()),
                                         #max(plotting_dict.keys())) if x % 10 == 0])
            ax.set_xlabel("Passed Hours", fontsize=fs)
            ax.set_ylabel("Accessed Item", fontsize=fs)
            plt.tick_params(axis='both', which='major', labelsize=fs)
            if param.exp_type == "temp":
                plt.savefig(f"plots/{param.hash_f}_lim{limit}_n{param.n_items}_server_acc_p{get_p_acc(f)}.pdf", bbox_inches='tight')
            elif param.exp_type == "ctr":
                plt.savefig(f"plots/{param.hash_f}_lim{limit}_n{param.n_items}_server_acc_ctr.pdf", bbox_inches='tight')
            plt.show()

def plot_occ(cap="infinite"):
    files = [f for f in os.listdir('results/') if os.path.isfile(os.path.join('results/', f))]
    for f in files:
        #if f.endswith("0.15_occupation.json") or f.endswith("0.9_occupation.json"):
        with open("results/" + f, 'r') as fr:
            single_res = json.load(fr)
        sorted_dict = {k: int(v) for k, v in sorted(single_res.items(), key=lambda item: item[1])}
        plt.bar(sorted_dict.keys(), sorted_dict.values(), width=1, color='#009988')
        plt.yticks(np.arange(0, 140, 10))
        plt.xticks([])
        if cap == "infinite":
            if f.startswith("circle_inf"):
                print(get_p_occ(f) + " " + str(sorted_dict))
                if param.exp_type == "temp":
                    plt.savefig(f"plots/n{param.n_items}_inf_cap_occ_h{param.hash_f}_p{get_p_occ(f)}_s"
                                f"{param.stale_time}.pdf", bbox_inches='tight')
                else:
                    plt.savefig(f"plots/inf_cap_occ_h{param.hash_f}_ctr_f{param.serv_ins_freq}_s"
                                f"{param.stale_time}.pdf", bbox_inches='tight')
        else:
            if f.startswith("circle_c"):
                if param.exp_type == "temp":
                    plt.savefig(f"plots/n{param.n_items}_cap_{get_c_occ(f)}_occ_h{param.hash_f}_p{get_p_occ(f)}_s"
                                f"{param.stale_time}.pdf", bbox_inches='tight')
                else:
                    plt.savefig(f"plots/cap_{cap}_occ_h{param.hash_f}_ctr_f{param.serv_ins_freq}_s"
                                f"{param.stale_time}.pdf", bbox_inches='tight')
        #plt.show()


def plotting_occ_all(temp_p=param.temp_p):
    files = [f for f in os.listdir('results2_7/') if os.path.isfile(os.path.join('results2_7/', f))]
    data = []
    extremes = {}
    map_input = []
    access_costs = {}
    tot_items = -99
    max_unbound = -99
    print(files)
    with open("results2_7/ctr_results.json", 'r') as fr:
        res = json.load(fr)
    for c in res:
        for s in res[c]:
            for algo in res[c][s]:
                print(algo)
                if algo == "algo_access":
                    access_costs["H&A"] = res[c][s][algo]
                elif algo == "wbl_access":
                    access_costs["WBL"] = res[c][s][algo]
                elif algo == "static_access":
                    access_costs["Traditional"] = 0
    for f in files:
        if f.__contains__(temp_p) or f.__contains__("ctr"):
            if f == "ctr_results.json":
                continue
            with open("results2_7/" + f, 'r') as fr:
                single_res = json.load(fr)
            sorted_dict = {k: int(v) for k, v in sorted(single_res.items(), key=lambda item: item[1])}
            print(f)
            print(sorted_dict)
            if tot_items == -99:
                tot_items = sum(sorted_dict.values())
            count = 0
            #test_sum = 0
            for req in sorted_dict:
                #if f.__contains__("inf") and f.__contains__("pow2"):
                 #   data.append(['Unbound_pow2', count, sorted_dict[req]])
                if f.__contains__("inf"):
                    struct_type = 'Traditional'
                    plot_desc = 'Traditional Consistent Hashing [28]'
                elif f.__contains__("1.25"):
                    struct_type = 'WBL'
                    plot_desc = 'Consistent Hashing with \nBounded Loads [35]'
                elif f.__contains__("c2"):
                    struct_type = 'H&A'
                    plot_desc = 'Hash & Adjust'
                else:
                    raise Exception("File type in results not valid")
                data.append([struct_type, count, sorted_dict[req]])
                if count == 0:
                    extremes[struct_type] = {}
                    extremes[struct_type]['first'] = sorted_dict[req]
                elif count == len(sorted_dict) - 1:
                    extremes[struct_type]['last'] = sorted_dict[req]
                    #print(f"Dividing {extremes[struct_type]['first']} by {extremes[struct_type]['last']}")
                    #map_input.append((extremes[struct_type]['first']/extremes[struct_type]['last'], struct_type))
                    map_input.append((tot_items/(extremes[struct_type]['last']*param.init_servers),
                                      access_costs[struct_type]/param.n_items, plot_desc))
                    if struct_type == 'Traditional':
                        max_unbound = extremes[struct_type]['last']
                count += 1
                #test_sum += sorted_dict[req]
            #print(test_sum)
    count = 0
    for i in range(0, param.init_servers):
        #data.append(['m/n', count, int(tot_items / param.init_servers)])
        data.append(['M.L. H&A', count, int(tot_items/param.init_servers)+2])
        data.append(['M.L. WBL', count, int(tot_items/param.init_servers*1.25)])
        data.append(['M.L. Traditional', count, max_unbound])
        count+=1

    df = pd.DataFrame(data, columns=['name', 'server', 'n_items'])
    print(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    #ax.set_xticks(df.server.unique())
    #ax.set_xticks([])
    fs = 20
    sns.set_palette("hls", 6)
    sns.lineplot(x='server', y='n_items', data=df, hue='name')
    ax.set_ylabel('Number of Items', fontsize=fs)
    ax.set_xlabel('Server IDs', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    plt.xticks([y for y in range(0, param.init_servers) if y%2])
    leg = ax.legend(bbox_to_anchor=(1, 0), loc='lower right', fontsize=fs-2, ncol=2)
    leg_lines = leg.get_lines()
    for line in range(0,len(ax.lines)):
        if line == 0:   # 1.25
            ax.lines[line].set_linestyle(':')
            ax.lines[line].set_linewidth(5)
            leg_lines[line].set_linestyle(':')
            leg_lines[line].set_linewidth(5)
            ax.lines[line].set_color('#FFB570') # orange
            leg_lines[line].set_color('#FFB570')

        if line == 1:       # c2
            ax.lines[line].set_linestyle('-')
            ax.lines[line].set_linewidth(5)

            leg_lines[line].set_alpha(1)
            leg_lines[line].set_linestyle('-')
            leg_lines[line].set_linewidth(5)
            ax.lines[line].set_color('#67AB9F') #green
            leg_lines[line].set_color('#67AB9F') #green
            leg_lines[line].set_alpha(0.5)
            ax.lines[line].set_alpha(0.5)
        if line == 2:       # unbound
            ax.lines[line].set_alpha(0.8)
            ax.lines[line].set_linestyle('-')
            ax.lines[line].set_linewidth(1)
            ax.lines[line].set_color('#A680B8') # purple
            leg_lines[line].set_alpha(0.8)
            leg_lines[line].set_linestyle('-')
            leg_lines[line].set_linewidth(1)
            leg_lines[line].set_color('#A680B8')
        if line == 3 or line == 4 or line == 5:
            ax.lines[line].set_linestyle('--')
            ax.lines[line].set_alpha(0.8)
            #ax.lines[line].set_dashes(10)
            ax.lines[line].set_dashes([5, 10, 5, 10])    # 2pt line, 2pt break, 3pt line, 2pt break.
            ax.lines[line].set_linewidth(3)
            if line == 3:
                ax.lines[line].set_color('green')  # purple
                leg_lines[line].set_color('green')
            #ax.lines[line].set_dash_capstyle('round')
            leg_lines[line].set_linestyle("--")
            leg_lines[line].set_dashes([5, 10, 5, 10])
            leg_lines[line].set_linewidth(2)
    if param.exp_type == "temp":
        plt.savefig(f"plots/occ_line_p{param.temp_p}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"plots/occ_line_ctr.pdf", bbox_inches='tight')
    plt.show()

    # plot ratios
    fs=25
    for i in map_input:
        print(i)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.grid()
    util, cost, size = [], [], []

    for i in range(len(map_input)):
        util.append(map_input[i][0])
        cost.append(map_input[i][1])
        size.append(100)

    plt.xlabel('Adjusted Average Access Cost', fontsize=fs)
    plt.ylabel('Memory Utilization', fontsize=fs)

    font = {'family': 'normal',
            # 'weight': 'bold',
            'size': fs}
    plt.rc('font', **font)
    colors = ['#FFB570', '#67AB9F', '#A680B8']
    ax.scatter(cost, util, s=size, c=colors, clip_on=False)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    plt.yticks(np.arange(0, 1.1, 0.1))
    vals = ax.get_yticks()
    print(vals)
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.xlim(0, 5)
    plt.ylim(0.7, 1)
    plt.gca().invert_xaxis()
    """
    # plot arrows
    # from https://3diagramsperpage.wordpress.com/2014/05/25/arrowheads-for-axis-in-matplotlib/
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_visible(False)
    # removing the axis ticks
    plt.xticks([])  # labels
    plt.yticks([])
    ax.xaxis.set_ticks_position('none')  # tick markers
    ax.yaxis.set_ticks_position('none')

    # wider figure for demonstration
    fig.set_size_inches(4, 2.2)
    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1. / 20. * (ymax - ymin)
    hl = 1. / 20. * (xmax - xmin)
    lw = 1.  # axis line width
    ohg = 0.3  # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
    yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax - xmin, 0., fc='k', ec='k', lw=lw,
    head_width = hw, head_length = hl, overhang = ohg,
    length_includes_head = True, clip_on = False)

    ax.arrow(0, ymin, 0., ymax - ymin, fc='k', ec='k', lw=lw,
    head_width = yhw, head_length = yhl, overhang = ohg,
    length_includes_head = True, clip_on = False)"""

    for i, e in enumerate(map_input):
        if i == 0:      # WBL
            plt.annotate(e[2], (cost[i]+3.2, util[i]-0.02), weight='bold', color=colors[i], fontsize=fs)
        if i == 1:      # H&A
            plt.annotate(e[2], (cost[i]+1.9, util[i] + 0.0), weight='bold', color=colors[i], fontsize=fs)
        if i == 2:      # Traditional
            plt.annotate(e[2], (cost[i] + 4.6, util[i] + 0.0), weight='bold', color=colors[i], fontsize=fs)
    if param.exp_type == "temp":
        plt.savefig(f"plots/util&cost_p{param.temp_p}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"plots/util&cost_ctr.pdf", bbox_inches='tight')
    plt.show()


def get_p_acc(filename):
    spl_char = '_p'
    first_part = filename.split(spl_char, 1)[1]
    spl_char = '_lim'
    return first_part.split(spl_char, 1)[0]


def get_p_occ(filename):
    spl_char = '_p'
    first_part = filename.split(spl_char, 1)[1]
    spl_char = '_occ'
    return first_part.split(spl_char, 1)[0]


def get_c_occ(filename):
    spl_char = '_c'
    first_part = filename.split(spl_char, 1)[1]
    spl_char = '_p'
    return first_part.split(spl_char, 1)[0]


def get_freq(filename):
    spl_char = '_f'
    first_part = filename.split(spl_char, 1)[1]
    spl_char = '_alg'
    return first_part.split(spl_char, 1)[0]


def get_stale(filename):
    spl_char = '_s'
    first_part = filename.split(spl_char, 1)[1]
    if param.exp_type == "ctr":
        spl_char = '.json'
        return first_part.split(spl_char, 1)[0]
    else:
        spl_char = '_f'
        return first_part.split(spl_char, 1)[0]