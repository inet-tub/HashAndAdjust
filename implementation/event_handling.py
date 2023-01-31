import random

import datetime
import implementation.priority_queue as pq
import implementation.event as ev
import copy

def create_queue(sequence=None, circle=None, type=None):
    """
    Create event queue out of trace
    :param sequence: the access trace
    :param circle: required structure
    :return: through the call of add_first_access_events, the final queue
    """
    if not sequence:
        raise Exception("A sequence needs to be passed in order to create an event queue")
    if not circle:
        raise Exception("Need to be passed circle instance as arg to compute hash")
    date_format = "%Y-%m-%d %H:%M:%S"
    queue = pq.PriorityQueue()
    init_time = "2006-03-01 00:01:13"
    incr_time = datetime.datetime.strptime(init_time, date_format)
    print("Creating event queue")
    for i in range(0, len(sequence)):

        # setting of params might differ ( e.g. aol-tuple is (src,dst,time) )
        if type == "ctr":   # format= request: [item_id, timestamp]
            timestamp = datetime.datetime.strptime(sequence[i][1], date_format)
            queue.insert(ev.AccessEvent(id=i, timestamp=timestamp, item_id=sequence[i][0],
                                        inserting_server=circle.get_item_position(sequence[i][0])), init=True)
        elif type == "temp":    # format= request: item_id
            #print(sequence)
            incr_time += datetime.timedelta(seconds=1)
            queue.insert(ev.AccessEvent(id=i,
                                            timestamp=incr_time,#+datetime.timedelta(seconds=1), #random.randint(0, 10)),
                                            item_id=sequence[i],
                                            inserting_server=circle.get_item_position(sequence[i])), init=True)
        else:
            raise Exception("Please enter which kind of sequence we are serving (ctr/temp)")
    return queue
    #return replace_access_events(queue, circle)

    #curr_time = datetime.datetime.strptime(init_time, date_format)


def add_first_access_events(queue, circle):
    """
    add the events that represent the actual access (with random delay)
    """
    adding_events = []
    copy_q = copy.deepcopy(queue)       # deep copy = events are also duplicated (not just ref)
    n_events = len(copy_q)
    i = n_events
    while not copy_q.is_empty():
        event = copy_q.delete()
        adding_events.append(ev.AccessEvent(id=i,
                                            timestamp=event.timestamp+datetime.timedelta(seconds=random.randint(0, 10)),
                                            item_id=event.item_id,
                                            inserting_server=circle.get_item_position(event.item_id)))
        i += 1
    for e in adding_events:
        queue.insert(e)
    return queue

"""
def replace_access_events(queue, circle):
    
    #replace the Outside-access events with actual access (with random delay)
    adding_events = []
    copy_q = copy.deepcopy(queue)       # deep copy = events are also duplicated (not just ref)
    n_events = len(copy_q)
    for i in range(0, n_events):
        event = copy_q.delete()
        adding_events.append(ev.AccessEvent(id=i,
                                            timestamp=event.timestamp+datetime.timedelta(seconds=random.randint(0, 10)),
                                            item_id=event.item_id,
                                            inserting_server=circle.get_item_position(event.item_id)))
        i += 1
    for e in adding_events:
        queue.insert(e)
    return queue """