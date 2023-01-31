import copy
import random

#from implementation.circle import *
import implementation.event_handling as eh
import implementation.algo_utils as au
from params import param


class Static_algo:
    def __init__(self, circle):
        self.circle = circle
        self.debug = False

        self.max_iteration = 0
        self.access_cost = 0
        self.random_adjustment = False

        self.deleted_items = 0
        self.inserts_after_init = 0
        self.reinserted_items = 0

        self.sum_del_items = 0
        self.sum_reinserted_items = 0

    def serve_sequence(self, sequence, type):
        queue = eh.create_queue(sequence=sequence, circle=self.circle, type=type)
        event_seq = copy.deepcopy(queue)  # deep copy = events are also duplicated (not just ref)
        print("Starting to work through event queue")
        while not event_seq.is_empty():
            curr_event = event_seq.delete()  # pop queue, event after event
            if curr_event.get_type() == "access":
                self.check_insertion_need(curr_event)  # insert item if needed
                if not au.lookup(circle=self.circle, debug=self.debug,
                                 item_id=curr_event.item_id, server_id=curr_event.inserting_server):
                    au.search_update_queue(circle=self.circle, debug=self.debug, event_seq=event_seq, curr_event=curr_event)
                else:  # found item_id on server associated with curr_event
                    if self.debug:
                        print("Event " + str(curr_event.id) + " succeeds in finding " + str(curr_event.item_id) +
                              " in server " + str(curr_event.inserting_server))
                    self.access(item_id=curr_event.item_id, timestamp=curr_event.timestamp,
                            server_id=curr_event.inserting_server)
                    event_seq.update_del_timestamp(curr_event)  # update deleting timestamps
            elif curr_event.get_type() == "deleting":
                if self.debug:
                    print("Deleting item " + str(curr_event.item_id))
                deleting_server = self.circle.delete_item(item_id=curr_event.item_id)
                self.deleted_items += 1
                if self.random_adjustment:
                    self.circle.adjust_random_server_capacity(increase=False, curr_server=deleting_server)
                self.circle.adjust_circle(server_to_fill=deleting_server)

            # after each event, determine whether to delete/insert a server or not
            self.circle.check_server_adjustments(curr_timestamp=curr_event.timestamp, random_adj=self.random_adjustment)

            # if capacity adjustment in phases, check whether it is time to adjust
            if not self.random_adjustment:
                if self.circle.check_server_capacity_adj_for_item_ins_del(
                        sum_insertions=self.inserts_after_init + self.reinserted_items,
                        sum_deletions=self.deleted_items):
                    self.sum_reinserted_items += self.reinserted_items
                    self.sum_del_items += self.deleted_items

                    self.deleted_items = 0
                    self.inserts_after_init = 0
                    self.reinserted_items = 0
        print("Max iteration for static: " + str(self.max_iteration))

    def serve_sequence_unbounded_cap(self, sequence, type):
        """
        Serve the sequence of requests passed as parameter.
        The parameter is used to create an event queue, on which the algo will work
        """
        queue = eh.create_queue(sequence=sequence, circle=self.circle, type=type)
        event_seq = copy.deepcopy(queue)  # deep copy = events are also duplicated (not just ref)
        print("Starting to work through event queue")
        while not event_seq.is_empty():
            curr_event = event_seq.delete()  # pop queue, event after event
            if curr_event.get_type() == "access":  # access event
                self.access_mtf(item_id=curr_event.item_id)

    def access_mtf(self, item_id):
        original_server = self.circle.get_server(self.circle.get_item_position(item_id))
        steps_needed = 0
        for item_ids in original_server.slots:
            if original_server.item_timestamps[item_ids] > original_server.item_timestamps[item_id]:
                steps_needed += 1
        if steps_needed == 0:
            steps_needed = random.randint(1, original_server.get_current_occupation())
        self.access_cost += steps_needed

    def access(self, item_id, timestamp, server_id):
        if self.debug:
            print("Accessing " + str(item_id) + ", starting from server " + str(self.circle.get_item_position(item_id)))
        original_server = self.circle.get_server(self.circle.get_item_position(item_id))
        self.check_item_status(parent=original_server, item_id=item_id, timestamp=timestamp)
        n_iter = self.search_item(curr_server=original_server, item_id=item_id)
        self.access_cost += n_iter      # steps from original server to find the item on actual server
        if n_iter > self.max_iteration:
            self.max_iteration = n_iter
        if self.debug:
            print("Found item " + str(item_id) + " after " + str(n_iter) + " iterations")

    def search_item(self, curr_server, item_id):
        iteration = 0
        for i in range(0, len(self.circle.present_server_ids)):
            if curr_server.check_for_item(item_id):
                return iteration
            curr_server = curr_server.child_pointer
            iteration += 1
        raise Exception(str(item_id) + " not found in circle!")

    def search_item_recurs(self, curr_server, item_id, n_iteration):
        if curr_server.check_for_item(item_id):  # accessing item found
            return n_iteration
        else:
            return self.search_item_recurs(curr_server.child_pointer, item_id, n_iteration+1)

    def check_insertion_need(self, event):
        """ Iterate to server where the item has been hashed and check whether it is inserted """
        if self.debug:
            print("Checking insertion of " + str(event.item_id))
        parent = self.circle.root
        for i in range(0, len(self.circle.present_server_ids)):
            if self.check_item_status(parent, event.item_id, event.timestamp) \
                    or i >= self.circle.get_item_position(event.item_id):  # make sure item is present
                return
            else:
                parent = parent.child_pointer  # iterate through circle

    def check_item_status(self, parent, item_id, timestamp):
        """
        Check the item status:
        If it is the first time it has been hashed to this server, record it and insert item
        If it has already been hashed, but not present, check whether it has been deleted (and evtl. re-insert)
        :param parent: server where item should be (original position)
        :param item_id: accessed item
        :param timestamp: access-timestamp
        """
        if item_id not in self.circle.hashed_items:
            self.circle.hashed_items.add(item_id)
            inserting_server = self.circle.insert(item_id)
            if item_id in inserting_server.lru_list:
                raise Exception("Should not happen")
            inserting_server.lru_list.insert(0, item_id)
            inserting_server.item_timestamps[item_id] = timestamp
            if self.random_adjustment:
                self.circle.adjust_random_server_capacity(increase=True)
            if self.debug:
                print("First time-insertion of " + str(item_id) + " in " + str(parent.id))
            self.inserts_after_init += 1
            return True
        elif item_id in self.circle.deleted_items:  # item has been hashed to the circle before
            self.reinsert_removed_item(parent, item_id, timestamp)
            return True
        if self.debug:
            print("No insertion needed")

    def reinsert_removed_item(self, parent, item_id, timestamp):
        """
        Check if item has been removed in the past because its access-timestamp surpassed the timeconstraint.
        If so, re-insert previously removed item.
        :param item_id: accessed item to be re-inserted
        :param parent: server where the item has been hashed originally
        :param timestamp: new timestamp to be recorded for this item
        """
        inserting_server = self.circle.insert(item_id=item_id)
        inserting_server.item_timestamps[item_id] = timestamp
        if item_id in inserting_server.lru_list:
            raise Exception(f"Inserting {item_id}. Should not happen")
        inserting_server.lru_list.insert(0, item_id)
        self.circle.deleted_items.remove(item_id)
        self.reinserted_items += 1
        if self.random_adjustment:
            self.circle.adjust_random_server_capacity(increase=True)
        if self.debug:
            print("Reinserted " + str(item_id) + ", hashed at server " + str(parent.id))
