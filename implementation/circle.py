import hashlib
import random
from bitstring import BitArray
import datetime
import bisect
from scipy.stats import poisson

from params import param
import data_handling.json_handler as jh

class Server:
    def __init__(self, id, capacity=4):
        self.id = id
        self.capacity = capacity
        self.slots = set()
        self.left = self.right = None
        self.child_pointer = None
        self.parent_pointer = None
        self.lru_list = []      # list to keep track of LRU. Least recent items will be at end of list
        self.item_timestamps = dict()
        #self.hashed_items = set()

        # keep track of items that have been removed because last access surpassed time constraint
        # items could have been removed from other servers in circle, but this is the original one (=result of hash)
        self.timestamps_orig_hash = dict()

    def insert(self, item_id, init=False, buffer=False):
        """
        :param init: true if in Circle-initialization phase
        :param item_id: item to insert
        :return: True if successful
        """
        if item_id in self.slots:
            raise Exception(f"Item {item_id} already in server!")
        if not self.is_full():
            if init:
                self.slots.add(item_id)
                self.lru_list.insert(0, item_id)
                init_time = "2006-03-01 00:01:13"
                date_format = "%Y-%m-%d %H:%M:%S"
                self.item_timestamps[item_id] = datetime.datetime.strptime(init_time, date_format)
            else:
                self.slots.add(item_id)
            return True     # returns True if spot was found
        elif buffer:
            if len(self.slots) < self.capacity + 1:
                self.slots.add(item_id)
                return True  # returns True if spot was found
        else:
            raise Exception("Trying to insert item " + str(item_id) + " in full server: " + str(self))

    def get_current_occupation(self):
        return len(self.slots)

    def is_full(self):
        if self.get_current_occupation() > self.capacity:
            raise Exception(f"Server {self.id}, capacity {self.capacity}, {self.slots} overflown, should not happen")
        return self.get_current_occupation() == self.capacity

    def check_for_item(self, item_id):
        return item_id in self.slots

    def remove_item(self, item_id):
        if item_id in self.slots:
            self.slots.remove(item_id)
            return True
        return False

    def is_empty(self):
        return len(self.slots) == 0

    def delete_oldest_item(self):
        init_time = "2006-03-01 00:01:13"
        date_format = "%Y-%m-%d %H:%M:%S"
        oldest_time = datetime.datetime.strptime(init_time, date_format)
        deleting_item = None
        for i in self.slots:
            if self.item_timestamps[i] > oldest_time:
                deleting_item = i
                oldest_time = self.item_timestamps[i]
        if not deleting_item:
            deleting_item = random.choice(sorted(self.slots))
        del self.item_timestamps[deleting_item]
        self.lru_list.remove(deleting_item)
        self.slots.remove(deleting_item)
        return deleting_item

    def get_oldest_item_id(self):
        init_time = "2006-03-01 00:01:13"
        date_format = "%Y-%m-%d %H:%M:%S"
        oldest_time = datetime.datetime.strptime(init_time, date_format)
        deleting_item = None
        for i in self.slots:
            if self.item_timestamps[i] > oldest_time:
                deleting_item = i
                oldest_time = self.item_timestamps[i]
        if not deleting_item:
            deleting_item = random.choice(sorted(self.slots))
        return deleting_item


    def __str__(self):
        return str(self.id) + ", " + str(self.slots)

class Circle:
    def __init__(self, id, n_servers, m_items):
        self.id = id
        self.n_servers = n_servers
        self.n_items = m_items
        self.hashed_items = set()
        self.deleted_items = set()
        self.reconfig_cost = 0

        # server capacity is set according to experiment parameters and initialization
        self.server_capacity = -99 #if not m_items%n_servers else int(m_items/n_servers)+1
        # print("Server capacity = " + str(self.server_capacity))

        self.root = None
        self.debug = False

        # hashing-arguments
        self.random_seed = random.randint(0, 1000)      # determine random seed for this Circle
        self.random_seed2 = random.randint(0, 1000)
        self.large_prime = 5123448959389961462809
        self.b = random.randint(0, self.large_prime)
        self.c = random.randint(1, self.large_prime)
        self.d = random.randint(1, self.large_prime)
        self.e = random.randint(1, self.large_prime)

        self.access_cost = 0
        self.reconfig_cost = 0
        #self.filled_servers = 0
        self.server_c_record = []

        if param.exp_type == "ctr":
            # ctr (aol-timestamps):
            if param.random_serv_ins_del:
                # if mean of server del/ins=200 minutes, first item 2006-03-01 00:01:13,last 2006-05-31 23:59:48
                # 3 months = 131490 minutes. 131490/200 = ~660 times. Generating 10% more intervals just to be sure
                self.server_insert_timestamps = poisson.rvs(mu=param.serv_ins_freq,
                                                            size=int(131490/param.serv_ins_freq+131490/param.serv_ins_freq*0.1)).tolist()
                self.server_deletion_timestamps = poisson.rvs(mu=param.serv_del_freq,
                                                            size=int(131490/param.serv_del_freq+131490/param.serv_del_freq*0.1)).tolist()
            else:
                data_dict = jh.get_serv_ins_del_ctr()
                self.server_insert_timestamps = data_dict["ins_times"]
                self.server_deletion_timestamps = data_dict["del_times"]
                self.deleting_server_list = data_dict["del_items"]
        else:                   # temporal p
            if param.random_serv_ins_del:
                tot_minutes = param.size_dataset/60     # one second for each request, hence tot-minutes = tot-req/60
                self.server_insert_timestamps = poisson.rvs(mu=param.serv_ins_freq,
                                                            size=int(tot_minutes/param.serv_ins_freq+tot_minutes/param.serv_ins_freq*0.1)+100).tolist()
                self.server_deletion_timestamps = poisson.rvs(mu=param.serv_del_freq,
                                                              size=int(tot_minutes/param.serv_del_freq+tot_minutes/param.serv_del_freq*0.1)+100).tolist()
                if param.serv_ins_freq > tot_minutes:
                    raise Exception(f"Sum of time intervals for server ins/del does not cover total length of sequence. "
                                    f"Please set lower frequency (<{tot_minutes})")
                else:
                    if self.debug:
                        print(f"Created {len(self.server_insert_timestamps)} intervals for server insertion")
                        print(f"Created {len(self.server_deletion_timestamps)} intervals for server deletion")
            else:
                data_dict = jh.get_serv_ins_del_temp()
                self.server_insert_timestamps = data_dict["ins_times"]
                self.server_deletion_timestamps = data_dict["del_times"]
                self.deleting_server_list = data_dict["del_items"]

        self.init_time = "2006-03-01 00:01:13"
        self.date_format = "%Y-%m-%d %H:%M:%S"
        self.insert_time_limit = self.delete_time_limit = datetime.datetime.strptime(self.init_time, self.date_format)
        self.present_server_ids = []
        self.present_item_ids = set()
        self.root_server_id = -99

    def get_item_position(self, id):
        """
        Return the position of the item in the circle.
        """
        if param.hash_f == "sha":
            return self.get_hash_simple(id)
        elif param.hash_f == "5k":
            return self.get_hash_5k(id)
        elif param.hash_f == "pow2":
            return self.get_hash_pow2(id)
        else:
            raise Exception("Hash-f parameter got to be \"sha\" or \"5k\"")

    def get_hash_pow2(self, item_id):
        """Calculates hash using sha and comparing the
        occupancy of the two servers that resulted from the different random seed.
        Return the server that is least occupied"""
        server1 = (int(BitArray(hex=hashlib.sha512(str(str(item_id)).encode()).hexdigest()).bin, 2) \
                  + int(BitArray(hex=hashlib.sha512(str(int(self.random_seed)).encode()).hexdigest()).bin, 2)) % self.n_servers
        server2 = (int(BitArray(hex=hashlib.sha512(str(str(item_id)).encode()).hexdigest()).bin, 2) \
                  + int(BitArray(hex=hashlib.sha512(str(int(self.random_seed2)).encode()).hexdigest()).bin, 2)) % self.n_servers
        curr_s = self.root
        for i in range(0, len(self.present_server_ids)):
            if curr_s.id == server1:
                s1_occ = curr_s.get_current_occupation()
            elif curr_s.id == server2:
                s2_occ = curr_s.get_current_occupation()
            curr_s = curr_s.child_pointer
        if s1_occ == s2_occ:
            return server1
        elif s1_occ < s2_occ:
            #print("Selected s1")
            return server1
        else:
            #print("Selected s2")
            return server2
            # return int(BitArray(hex=hashlib.sha512(str(str(item_id)).encode()).hexdigest()).bin, 2) % self.n_servers

    def get_hash_5k(self, item_id):
        """ Calculate a 5-k independent hash using the 5 random numbers defined in __init__
        formula = (ex^4 + dx^3 + cx^2 + b) % p % m
        where
        m = n_servers,
        p = fixed random prime,
        b = fixed random number:  0 <= b < p
        c , d, e = fixed random number: 0 < c,d,e < p
        """
        # simply hash the item_id first, as we might have strings as input
        b_hash = abs(hash(item_id)) % (10 ** 8)     # take last 8 digits of has

        # apply formula
        return (self.e*pow(b_hash, 4) + self.d*pow(b_hash, 3) + self.c*pow(b_hash, 2) + self.b) \
               % self.large_prime % self.n_servers

    def get_hash_simple(self, item_id):
        """Calculates hash on the fly"""
        if type(item_id) == int:
            return int(BitArray(hex=hashlib.sha512(str(int(item_id)).encode()).hexdigest()).bin,
                       2) % self.n_servers
            #return int(BitArray(hex=hashlib.sha512(str(int(item_id) + self.random_seed).encode()).hexdigest()).bin, 2) % self.n_servers
        else:
            return int(BitArray(hex=hashlib.sha512(str(str(item_id)).encode()).hexdigest()).bin, 2) % self.n_servers

    def init_servers(self, preloaded_items):
        """
        Create the basic structure of the circle with empty servers
        """
        parent = self.root
        index_counter = 0
        if self.debug:
            print("Creating initial server structure")
        for n in range(0, self.n_servers):
            if parent is None:      # means we are at root
                self.root = Server(id=index_counter, capacity=self.server_capacity)
                self.root_server_id = self.root.id
                parent = self.root
            else:
                new_child = Server(id=index_counter, capacity=self.server_capacity)
                parent.child_pointer = new_child
                new_child.parent_pointer = parent
                parent = new_child
            bisect.insort(self.present_server_ids, index_counter)
            index_counter += 1
        parent.child_pointer = self.root
        self.root.parent_pointer = parent

        # snippet to use if we want to preload all/(half of the) destinations first
        for x in preloaded_items:
            self.insert(x, init=True)


    def insert(self, item_id, init=False):
        """Insert a new value in the circle. Takes one argument (the Item) """
        parent = self.root
        for i in range (0, len(self.present_server_ids)):

            # >= because if hash(x)=4 and there is no server.id=4, then server(x)=5/6/7,...
            if parent.id >= self.get_item_position(item_id):
                if init:
                    if self.debug:
                        print(f"INIT Inserting {item_id}, h={self.get_item_position(item_id)} from server {parent.id}")
                    self.hashed_items.add(item_id)  # record first time insertion
                else:
                    if self.debug:
                        print(f"Inserting {item_id}, h={self.get_item_position(item_id)} from server {parent.id}")
                if not parent.is_full():
                    parent.insert(item_id=item_id, init=init)
                else:
                    parent = self.forwarded_insert(server=parent, item_id=item_id, init=init)
                #if parent.is_full():
                 #   self.filled_servers += 1
                self.present_item_ids.add(item_id)
                return parent
            # special case where we are trying to insert in a deleted server at end of circle
            elif i == len(self.present_server_ids)-1 and self.get_item_position(item_id) != self.present_server_ids[i]:
                if init:
                    if self.debug:
                        print(f"Last server: INIT Inserting {item_id}, h={self.get_item_position(item_id)} from server {parent.id}")
                    self.hashed_items.add(item_id)  # record first time insertion
                else:
                    if self.debug:
                        print(f"Last server: Inserting {item_id}, h={self.get_item_position(item_id)} from server {parent.id}")
                if not self.root.is_full():
                    if self.debug:
                        print("Inserting at root")
                    self.root.insert(item_id=item_id, init=init)
                    self.present_item_ids.add(item_id)
                    return self.root
                else:
                    parent = self.forwarded_insert(server=self.root, item_id=item_id, init=init)
                    self.present_item_ids.add(item_id)
                    return parent
                #if parent.is_full():
                 #   self.filled_servers += 1
            else:
                parent = parent.child_pointer   # iterate through circle

    def forwarded_insert(self, server, item_id, init=False):
        """ iterate through circle until we find a free slot to insert item """
        for i in range(0, len(self.present_server_ids)):      # go through maximum all servers
            if self.debug:
                print("Forwarding " + str(item_id) + " from server " + str(server.id) + ", whose occupancy=" + str(
                    server.get_current_occupation()))
            server = server.child_pointer
            self.reconfig_cost += 1
            if not server.is_full():
                server.insert(item_id=item_id, init=init)
                if self.debug:
                    print("Forward-inserted " + str(item_id) + " into server " + str(server.id))
                return server
        raise Exception("Inserting " + str(item_id) + ": no free slots found in circle " + str(self.id))

    def delete_item(self, item_id):
        """ Delete item from circle """
        parent = self.root
        deleting_server = self.search_and_delete(parent, item_id)
        if deleting_server:  # if a server has been returned
            if item_id in self.deleted_items:
                raise Exception("Item id already in server's deleted items, should not happen")
            else:
                # if self.debug:
                # print("Deleted " + str(item_id) + " from " + str(deleting_server.id))
                self.deleted_items.add(item_id)
                self.present_item_ids.remove(item_id)
                return deleting_server
        else:
            raise Exception("Did not succeed in deleting item "
                            + str(item_id) + ", h=" + str(self.get_item_position(item_id)))
        """for i in range(0, len(self.present_server_ids)):         # get to hashed server. In practice we do not need to iterate

            # case where item has been forwarded from deleted server to beginning of circle
            if i == len(self.present_server_ids)-1 and self.get_item_position(item_id) != self.present_server_ids[i]:
                deleting_server = self.search_and_delete(self.root, item_id)
                print(f"Looking at last server, deleting from {deleting_server.id}")
                if deleting_server:  # if a server has been returned
                    if item_id in self.deleted_items:
                        raise Exception("Item id already in server's deleted items, should not happen")
                    else:
                        # if self.debug:
                        # print("Deleted " + str(item_id) + " from " + str(deleting_server.id))
                        self.deleted_items.add(item_id)
                        self.present_item_ids.remove(item_id)
                        return deleting_server
                else:
                    raise Exception("Did not succeed in deleting item " + str(item_id))
            elif self.present_server_ids[i] >= self.get_item_position(item_id):         # found spot where item has been hashed
                deleting_server = self.search_and_delete(parent, item_id)     # now look for the item starting from this server
                if deleting_server:     # if a server has been returned
                    if item_id in self.deleted_items:
                        raise Exception("Item id already in server's deleted items, should not happen")
                    else:
                        #if self.debug:
                        #print("Deleted " + str(item_id) + " from " + str(deleting_server.id))
                        self.deleted_items.add(item_id)
                        self.present_item_ids.remove(item_id)
                        return deleting_server
                else:
                    raise Exception("Did not succeed in deleting item " + str(item_id))
            else:
                parent = parent.child_pointer  # iterate through circle
        raise Exception("Delete item unsuccessful: iterated through circle without finding server containing itemss")"""

    def search_and_delete(self, server, item_id):
        curr_server = server
        for i in range(0, len(self.present_server_ids)):
            if curr_server.remove_item(item_id=item_id):
                curr_server.lru_list.remove(item_id)  # in any case, remove item from lru-list
                del curr_server.item_timestamps[item_id]  # and from timestamp-dict
                if self.debug:
                    print("Deleted " + str(item_id) + " from " + str(curr_server.id))
                return curr_server
            else:
                curr_server = curr_server.child_pointer
                #if self.debug:
                 #   print("Item deletion: checking server " + str(curr_server.id))
        return None    # if no item deleted

    def adjust_circle(self, server_to_fill):
        """
        Make sure there are no free spots between the original server and servers with forwarded items.
        This can happen when
        1. we delete an item from a server
        2. we increase the server's size by a slot
        3. we insert a new server (here we have to check *capacity* times)
        -> Procedure pulls up any furtherly hashed items
        :param server_to_fill: server on which there is a free slot to fill
        """
        curr_server = server_to_fill
        if self.debug:
            print(f"Filling spot of {server_to_fill}, capacity: {server_to_fill.capacity}, starting adjustment")
        for i in range(0, len(self.present_server_ids)):  # potentially do for whole circle
            if not curr_server:
                raise Exception(f"Could not find server {curr_server}")
            if curr_server.child_pointer.is_full():

                # start pull-up procedure, assign curr_server to server we pulled from
                curr_server = self.pull_up_check(curr_server)
                if curr_server:
                    if self.debug:
                        print(f"Restarting adj-procedure for {curr_server}")
                    return self.adjust_circle(curr_server)     # restart procedure for the server we just deleted from
                else:   # if no server has been returned, it means we reached non-full server
                    return
            else:       # if there is a free spot in the child server, it must be the last one we check
                if not curr_server.child_pointer.is_empty():  # no need to pull up from empty child
                    self.check_item_to_pull_up(curr_server, curr_server.child_pointer)
                return      # stop in any case once we get to non-full server
        raise Exception("Adjust circle after deletion reached end of circle without encountering free spot")

    def pull_up_check(self, original_server):
        """
        Iterate to next servers to see if there are any items to pull up
        If there is a full server without any pull-up candidates, skip the server
        If we get to full server with candidates, pull up the candidate, return the full server
        If we get to non-full server, pull up the eventual candidate and exit
        """
        if original_server.is_full():
            raise Exception(f"Server passed to pull_up_check got to be non-full. Passed {original_server}")
        curr_server = original_server
        for i in range(0, len(self.present_server_ids)):
            if curr_server.child_pointer.is_full():
                if self.check_item_to_pull_up(original_server, curr_server.child_pointer):
                    return curr_server.child_pointer
                else:
                    pass
                if self.debug:
                    print(f"Adj: pulling up to {curr_server.id}, skipping full server (with no candidate)")
                curr_server = curr_server.child_pointer     # get to next server only if it is full
            else:
                if not curr_server.child_pointer.is_empty():  # if child is empty, no need to pull up
                    if curr_server.is_full():       # if parent is full, pull up to original server
                        self.check_item_to_pull_up(original_server, curr_server.child_pointer)
                    else:
                        self.check_item_to_pull_up(curr_server, curr_server.child_pointer)
                    if self.debug:
                        # print(self.present_server_ids)
                        print(f"Finished pull up after {i + 1} iterations")
                return None     # stop in any case if we get to non-full server

    def check_item_to_pull_up(self, parent, child):
        """
        Check whether there is an item that can be pulled up to the parent-server
        If there are more candidates, choose the item with the youngest timestamp
        :return: True if item has been pulled up, False if there was no candidate to pull up
        """
        if parent.is_full():
            raise Exception(f"Parent server passed to this method got to be non-full!, passed {parent}")
        if child.id < parent.id:  # if child_id is smaller than parent it means we are around root of circle
            return self.check_around_root(parent, child)
        youngest_candidate = None
        curr_timestamp = datetime.datetime.strptime(self.init_time, self.date_format)
        for i in child.slots:

            # pull up only if hash(item) <= server_id (ex.: item with h=3 cannot go to server = 2)
            if self.get_item_position(i) <= parent.id:
                if child.item_timestamps[i] > curr_timestamp:
                    youngest_candidate = i
                    curr_timestamp = child.item_timestamps[i]
        if not youngest_candidate:
            for i in child.slots:
                # pull up only if hash(item)>server id
                if self.get_item_position(i) <= parent.id:
                    youngest_candidate = i
        if not youngest_candidate:  # if there is no candidate
            if self.debug:
                print(f"Adj: No item found to pull up from {child}-capacity:{child.capacity}, to {parent}")
            return False

        # remove from child
        child.remove_item(youngest_candidate)
        del child.item_timestamps[youngest_candidate]
        child.lru_list.remove(youngest_candidate)

        # insert in parent
        parent.insert(youngest_candidate)
        parent.item_timestamps[youngest_candidate] = curr_timestamp
        if youngest_candidate in parent.lru_list:
            raise Exception(f" Inserting {youngest_candidate}. Should not happen")
        parent.lru_list.insert(0, youngest_candidate)
        if self.debug:
            print(f"Adj: Pulled up {youngest_candidate}, h={self.get_item_position(youngest_candidate)} "
                  f"from {child}-capacity: {child.capacity}, to {parent}")
        self.reconfig_cost += 1
        return True

    def check_around_root(self, parent, child):
        """do not pull up any item that belongs on the right side of root
        by checking if h(item) is in second half of circle, but smaller than server_id """
        youngest_candidate = None
        curr_timestamp = datetime.datetime.strptime(self.init_time, self.date_format)
        #print(f"Len of slots: {len(child.slots)}, len of item_timestamps: {len(child.item_timestamps)}")
        for i in child.slots:
            if len(self.present_server_ids)/2 < self.get_item_position(i) <= parent.id:
                if child.item_timestamps[i] > curr_timestamp:
                    youngest_candidate = i
                    curr_timestamp = child.item_timestamps[i]
        if not youngest_candidate:      # in case there is no difference in timestamps, pick random item
            for i in child.slots:
                if len(self.present_server_ids)/2 < self.get_item_position(i) <= parent.id:
                    youngest_candidate = i
        if not youngest_candidate:  # if there is no candidate
            if self.debug:
                print(f"Adj: No item found to pull up from {child}-capacity:{child.capacity} to {parent}")
            return False

        # remove from child
        child.remove_item(youngest_candidate)
        del child.item_timestamps[youngest_candidate]
        child.lru_list.remove(youngest_candidate)

        # insert in parent
        parent.insert(youngest_candidate)
        parent.item_timestamps[youngest_candidate] = curr_timestamp
        if youngest_candidate in parent.lru_list:
            raise Exception(f" Inserting {youngest_candidate}. Should not happen")
        parent.lru_list.insert(0, youngest_candidate)
        if self.debug:
            print(f"Adj: Pulled up {youngest_candidate}, h={self.get_item_position(youngest_candidate)} "
                  f"from {child}-capacity: {child.capacity}, to {parent}")
        self.reconfig_cost += 1
        return True

    def adjust_random_server_capacity(self, increase, curr_server=None):
        """
        Pick random server and increase/decrease its capacity by one
        :param increase: if true, we increase the capacity. Otherwise, we decrease
        :return:
        """
        n_max_attempts = len(self.present_server_ids)
        while n_max_attempts:       # set limit of attempts since we might try to diminish capacity of full server
            index = random.randint(0, len(self.present_server_ids))
            for i in range(0, len(self.present_server_ids)):
                if i == index and index in self.present_server_ids:
                    if increase:    # just increase the capacity of the random server
                        self.get_server(index).capacity += 1
                        return
                    else:       # if not increase: make sure that there is at least one free slot
                        if self.get_server(index).get_current_occupation() < self.get_server(index).capacity and \
                                self.get_server(index).capacity > 1:

                                # also assert we are not diminishing capacity of server we just deleted from
                                # because we need to pull up another item
                                if not index == curr_server.id:
                                    self.get_server(index).capacity -= 1
                                    return
            n_max_attempts -= 1
        raise Exception(f"Could not increase a random server's capacity after {n_max_attempts} attempts")

    def check_server_capacity_adj_for_item_ins_del(self, sum_insertions, sum_deletions):
        """
        Check whether it is necessary to increase/decrease server's capacities (by one) according to
        the item insertions/deletions performed until now
        If adjustment is necessary, increase/decrease all server's capacity
        Once adjustment is done, also adjust circle by pulling up any forwarded items
        return: True if adjustment has been done, False otherwise
        """
        if sum_insertions-sum_deletions >= len(self.present_server_ids):
            curr_server = self.root
            #if self.debug:
            #print(f"Positive diff of {sum_insertions-sum_deletions} items: "
             #     f"Increasing capacity of all servers to {curr_server.capacity+1}!")
            for i in range(len(self.present_server_ids)):
                curr_server.capacity += 1
                if self.debug:
                    print(f"Adjusting circle after increasing capacity of {curr_server}")
                self.adjust_circle(curr_server)     # pull up any forwarded items to newly freed slot
                curr_server = curr_server.child_pointer
            self.server_c_record.append(curr_server.capacity)
            return True
        elif sum_insertions-sum_deletions <= -(len(self.present_server_ids)):
            curr_server = self.root
            #print(f"Negative diff of {sum_insertions-sum_deletions} items: "
             #     f"Decreasing capacity of all servers to {curr_server.capacity-1}!")
            for i in range(len(self.present_server_ids)):
                if not curr_server.is_full():
                    curr_server.capacity -= 1
                else:
                    timestamps_copy = curr_server.item_timestamps.copy()
                    deleted_item_id = curr_server.delete_oldest_item()
                    if self.debug:
                        print(f"Decreasing capacity of full server {curr_server}")
                    curr_server.capacity -= 1
                    inserting_server = self.insert(item_id=deleted_item_id, init=False)
                    inserting_server.item_timestamps[deleted_item_id] = timestamps_copy[deleted_item_id]
                    if deleted_item_id in inserting_server.lru_list:
                        raise Exception("Should not happen")
                    inserting_server.lru_list.insert(0, deleted_item_id)
                    self.reconfig_cost += 1
                    #print(f"Cap-Adj: deleted {deleted_item_id}, h={self.get_item_position(deleted_item_id)}"
                     #     f" from {curr_server.id}, inserted in {inserting_server.id}")

                curr_server = curr_server.child_pointer
            self.server_c_record.append(curr_server.capacity)
            return True
        return False


    def check_server_adjustments(self, curr_timestamp, random_adj):
        """
        Check whether it is necessary to insert/delete servers according to the timestamps generated at beginning
        :param curr_timestamp: current time
        """
        insert_time_constr = datetime.timedelta(minutes=int(self.server_insert_timestamps[0]))
        delete_time_constr = datetime.timedelta(minutes=int(self.server_deletion_timestamps[0]))

        # execute only if time_limit surpassed
        if curr_timestamp > self.insert_time_limit + insert_time_constr:
            print("Inserting a new server")
            self.server_insert_timestamps.pop(0)  # delete this timestamp
            self.insert_new_server()
            self.cap_adj_after_server_ins_del()
            self.insert_time_limit += insert_time_constr
            #print(self.present_server_ids)
            #self.print_circle()
        if curr_timestamp > self.delete_time_limit + delete_time_constr:
            print("Deleting a server")
            self.server_deletion_timestamps.pop(0)  # delete this timestamp
            deleted_server = self.delete_server(random_adj=random_adj)
            self.cap_adj_after_server_ins_del()
            self.forward_items_after_del(curr_server=deleted_server, random_adj=random_adj)
            self.delete_time_limit += delete_time_constr
            #print(self.present_server_ids)
            #self.print_circle()

    def delete_server(self, random_adj):
        """
        delete all items of a random server, cut this server out of the circle
        @param random_adj: if True, increase circle's capacity (random servers) by # of deleted items
        reinsert the items in the circle
        """
        if param.random_serv_ins_del:
            rand_server_id = random.choice(self.present_server_ids)    # choose random server out of the present ones
        else:
            rand_server_id = self.deleting_server_list.pop(0)
            while rand_server_id not in self.present_server_ids:
                rand_server_id = self.deleting_server_list.pop(0)
                #raise Exception("Picked a non existing random server")
        curr_server = self.root
        for i in range(0, len(self.present_server_ids)):
            if curr_server.id == rand_server_id:     # found deleting server
                #print(f"Deleting server {curr_server}, parent {curr_server.parent_pointer}, child {curr_server.child_pointer}")
                deleting_s_child = curr_server.child_pointer
                deleting_s_parent = curr_server.parent_pointer
                deleting_s_parent.child_pointer = deleting_s_child
                deleting_s_child.parent_pointer = deleting_s_parent
                if curr_server == self.root:
                    self.root = curr_server.child_pointer
                    self.root_server_id = self.root.id
                self.present_server_ids.remove(rand_server_id)
                print(
                    f"Deleted server {curr_server}, connected {curr_server.parent_pointer.id} with {curr_server.child_pointer.id}")
                if random_adj:
                    self.forward_items_after_del(curr_server=curr_server,random_adj=True)
                else:
                    return curr_server      # return server so that capacity can be increased before moving items
            else:
                curr_server = curr_server.child_pointer

    def forward_items_after_del(self, curr_server, random_adj):
        for item_id in curr_server.slots:
            if random_adj:
                self.adjust_random_server_capacity(increase=True)  # for each item, add a spot in circle
            if not curr_server.child_pointer.is_full():
                curr_server.child_pointer.insert(item_id=item_id, init=False)
                if item_id in curr_server.child_pointer.lru_list:
                    raise Exception(f"Inserting {item_id}. Should not happen")
                curr_server.child_pointer.lru_list.insert(0, item_id)
                curr_server.child_pointer.item_timestamps[item_id] = curr_server.item_timestamps[item_id]
                #print(f"Inserted item {item_id} in {curr_server.child_pointer.id}")
            else:       # if child is full, forward from it
                inserting_server = self.forwarded_insert(server=curr_server.child_pointer, item_id=item_id)
                if item_id in inserting_server.lru_list:
                    raise Exception(f"Inserting {item_id}. Should not happen")
                inserting_server.lru_list.insert(0, item_id)
                inserting_server.item_timestamps[item_id] = curr_server.item_timestamps[item_id]
                #print(f"Inserted item {item_id} in {inserting_server.id}")
            self.reconfig_cost += 1

    def insert_new_server(self):
        """ Insert new server in the circle
        New Server_id:
        The new server will insert itself in the first "free spot" according to IDs
        That is, if we have IDs 0,1,2,3,5,6, ... , the new server will have ID=4
        If we have 60 servers having IDs 0-59, the newly inserted server will have id=60
        After insertion, eventually pull up any items that could be hosted by this server
        -> Check item pull up *capacity* times
        After pulling up, set all server's capacities to #items/#servers + augmentation (not in this method)"""
        curr_server = self.root
        for i in range(0, len(self.present_server_ids)+1):      # iterate from 0 to current n_servers+1
            if i not in self.present_server_ids:
                new_s = Server(id=i, capacity=int(len(self.present_item_ids)/len(self.present_server_ids))
                                              +int(param.addit_augm))   # create the new Server
                curr_s_parent = curr_server.parent_pointer  # new server comes before curr_server

                # example: curr_server = 42. new_s = 41
                new_s.parent_pointer = curr_server.parent_pointer
                new_s.child_pointer = curr_server
                curr_s_parent.child_pointer = new_s
                curr_server.parent_pointer = new_s
                bisect.insort(self.present_server_ids, i)
                if self.debug:
                    print(f"Inserted server {i} as child of {new_s.parent_pointer} and parent of {new_s.child_pointer}")
                    print(
                        f"Server {new_s.child_pointer} has parent "
                        f"{new_s.child_pointer.parent_pointer} and child {new_s.child_pointer.child_pointer}")
                    print(f"Starting to evtl. pull up items to {new_s}")
                for i in range(new_s.capacity):   # evtl. pull up *capacity* times
                   self.adjust_circle(new_s)
                # if we are inserting a server before current root.id, we need to make it the new root
                if new_s.child_pointer == self.root and new_s.id < len(self.present_server_ids)/2:
                    self.root = new_s
                    self.root_server_id = self.root.id
                return new_s
            else:
                curr_server = curr_server.child_pointer


    def cap_adj_after_server_ins_del(self):
        """
        All servers should have capacity = # items / # servers + augmentation
        Once capacity adjustment is done, also adjust circle by pulling up any forwarded items
        return: True if adjustment has been done, False otherwise
        """
        curr_server = self.root
        if self.debug:
            print(f"Adjusting capacity of all servers, new cap should be: "
                  f"{(int(len(self.present_item_ids)/len(self.present_server_ids)) + int(param.addit_augm))} ")
            print(f"{len(self.present_item_ids)} items for {len(self.present_server_ids)} servers")
        for i in range(len(self.present_server_ids)):       # do for all servers
            diff = (int(len(self.present_item_ids)/len(self.present_server_ids)) +
                int(param.addit_augm)) - curr_server.capacity
            # diff>0: a server has previously been deleted -> #items/#servers -ratio will be bigger (less servers)
            if diff > 0:
                #print(
                    #f"Setting new capacity to {int(len(self.present_item_ids) / len(self.present_server_ids)) + int(param.addit_augm)}")
                if curr_server.is_full():       # pull up evtl. items only if server was full before cap-adj
                    curr_server.capacity += diff
                    for new_spot in range(diff):       # pull up *new_spot* times
                        self.adjust_circle(curr_server)
                else:
                    curr_server.capacity += diff
                if self.debug:
                    print(f"Added {diff} to capacity, curr= {curr_server.capacity}")
            # diff<0: a server has previously been added -> #items/#servers -ratio will be smaller
            elif diff < 0:
                free_deleting_spots = (curr_server.capacity-abs(diff)) - curr_server.get_current_occupation()

                #  are we trying to delete occupied slots?
                if free_deleting_spots < 0:
                    occupied_deleting_spots = abs(free_deleting_spots)
                    if self.debug:
                        print(f"Removing {occupied_deleting_spots} occupied "
                              f"slots from {curr_server.id}, capacity={curr_server.capacity}")
                    deleted_ids = set()
                    while occupied_deleting_spots:
                        item_id = curr_server.get_oldest_item_id()
                        if not curr_server.child_pointer.is_full():
                            curr_server.child_pointer.insert(item_id=item_id, init=False)
                            if item_id in curr_server.child_pointer.lru_list:
                                raise Exception(f"Inserting {item_id}. Should not happen")
                            curr_server.child_pointer.lru_list.insert(0, item_id)
                            curr_server.child_pointer.item_timestamps[item_id] = curr_server.item_timestamps[item_id]
                            occupied_deleting_spots -= 1
                        else:
                            inserting_server = self.forwarded_insert(server=curr_server.child_pointer, item_id=item_id)
                            if item_id in inserting_server.lru_list:
                                raise Exception(f"Inserting {item_id}. Should not happen")
                            inserting_server.lru_list.insert(0, item_id)
                            inserting_server.item_timestamps[item_id] = curr_server.item_timestamps[item_id]
                            occupied_deleting_spots -= 1
                        curr_server.lru_list.remove(item_id)  # in any case, remove item from lru-list
                        del curr_server.item_timestamps[item_id]  # and from timestamp-dict
                        deleted_ids.add(item_id)
                        if not curr_server.remove_item(item_id):        # removes the item from slots
                            raise Exception(f"Decreasing server's cap: could not remove item {item_id} from {curr_server}")
                    curr_server.capacity += diff        # does minus diff
                    self.reconfig_cost += len(deleted_ids)
                    #print(f"Removed {deleted_ids} from server {curr_server.id}, whose capacity is now: {curr_server.capacity}")
                else:
                    curr_server.capacity += diff
            curr_server = curr_server.child_pointer
        self.server_c_record.append(curr_server.capacity)

    def get_server(self, id):
        """ Get server-object for an item that has been hashed to passed id """
        parent = self.root
        for i in range(0, len(self.present_server_ids)):
            if parent.id >= id:
                return parent
            else:
                parent = parent.child_pointer
        if id >= self.n_servers*0.8 and id not in self.present_server_ids:
            return self.root

    def get_next_server_id(self, curr_server_id):
        next_server_id = -99
        for iter in range(0, len(self.present_server_ids)):
            if self.present_server_ids[iter] > curr_server_id:
                return self.present_server_ids[iter]
            elif iter == len(self.present_server_ids) - 1:  # if reached last spot (n-1), set next_id=first id
                return self.present_server_ids[0]
        # print(f"{curr_event.item_id} not found on {curr_event.inserting_server}. Setting next server_id to be {next_server_id}")
        if next_server_id == -99:
            raise Exception("Unable to find next_server_id")

    def print_circle(self):
        parent = self.root
        output = "Circle " + str(self.id) + ", server capacity = <"+str(self.server_capacity)+"> :\n"
        for n in range(0, len(self.present_server_ids)):
            output += (str(parent) + " | lru: " + str(parent.lru_list) + " | item_stamps-keys : "
                       + str(parent.item_timestamps.keys()) + "\n")
            parent = parent.child_pointer
        print(output)
