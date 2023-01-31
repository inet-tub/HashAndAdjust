from params import param
from implementation.event import *
import datetime

class PriorityQueue(object):

    def __init__(self):
        self.queue = []
        self.init_del_items = set()

    def __str__(self):
        output = ""
        sorted_q = sorted(self.queue, key=lambda event: event.timestamp)
        for e in sorted_q:
            output += str(e) + "\n"
        return output
        #return ' '.join([str(i.id) for i in self.queue])

    def __len__(self):
        return len(self.queue)

    # for checking if the queue is empty
    def is_empty(self):
        return len(self.queue) == 0

    # for inserting an element in the queue
    def insert(self, event, init):
        """
        Parameter init distinguishes whether we are at the first phase of the experiment, in which we are
        loading the items into the servers for the first time.
        """
        self.queue.append(event)
        if event.get_type() == "access" and init and event.item_id not in self.init_del_items:
            new_del_event = DeletingEvent(id=len(self.queue), item_id=event.item_id,
                                          timestamp=event.timestamp + datetime.timedelta(minutes=param.stale_time))
            self.queue.append(new_del_event)
            self.init_del_items.add(event.item_id)
        elif event.get_type() == "access" and not init:
            self.update_del_timestamp(event)

    # for popping an element based on Priority
    def delete(self):
        try:
            min_time = 0
            for i in range(len(self.queue)):
                if self.queue[i].timestamp < self.queue[min_time].timestamp:
                    min_time = i
            item = self.queue[min_time]
            del self.queue[min_time]
            return item
        except IndexError:
            print()
            exit()

    def update_del_timestamp(self, access_event):
        """
        Upon access of an item, create an event to delete the item after *stale_time* minutes
        If such an event is already present in queue, also delete the event with the old timestamp
        :param item_id: the id of the item which has been accessed
        """
        for i in range(0, len(self.queue)):
            if self.queue[i].get_type() == "deleting" and self.queue[i].item_id == access_event.item_id:
                del self.queue[i]
                #print("Item " + str(access_event.item_id) + ": Removing old deleting event from queue")
                break       # once found, no need to check other events
        new_del_event = DeletingEvent(id=len(self.queue), item_id=access_event.item_id,
                                      timestamp=access_event.timestamp+datetime.timedelta(minutes=param.stale_time))
        self.queue.append(new_del_event)