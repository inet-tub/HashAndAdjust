import math
class Params:
    def __init__(self):
        self.exp_type = "ctr"               # ctr, temp
        self.temp_p = "0.4"                # if exp_type="temp": 0.15, 0.3, 0.45, 0.6, 0.75, 0.9
        self.algorithm = "All"              # AdjustHash, Static, WBD; or All -> executes all 3
        self.hash_f = "sha"                 # sha, 5k, pow2
        self.addit_augm = "4"               # 4, 7, 10 . Set to "*1.25" algorithm="WBD"
        self.stale_time = 200               # in minutes
        self.init_servers = 20              # number of servers in datastructure
        self.size_dataset = 100000          # number of requests (artificial datasets)
        self.n_items = 10000                # number of items (artificial datasets)
        self.random_serv_ins_del = False    # insert/delete servers randomly if True. If False: fetch from file
        self.serv_ins_freq = 200            # in minutes
        self.serv_del_freq = 200
        self.initial_occup_factor = 0.5     # server occupation at initialization (before serving requests)
        self.unbounded_capacity = False
        self.show_circle_occup = False
        self.show_server_occup = False

    def set_n_servers(self, n):
        self.init_servers = n

    def set_n_items(self, n):
        self.n_items = n

    def set_size_dataset(self, n):
        self.size_dataset = n


param = Params()
