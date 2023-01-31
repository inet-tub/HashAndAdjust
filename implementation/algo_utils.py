import datetime
import random
import implementation.event as e
from params import param

def lookup(circle, debug, item_id, server_id):
    """
    Look if item can be found on mentioned server
    """
    parent = circle.root
    for i in range(0, len(circle.present_server_ids)):
        if circle.present_server_ids[i] == server_id:
            if debug:
                print(f"Looking for {item_id}, h={circle.get_item_position(item_id)}, "
                      f"on server {parent}, iter={i}, looked up server_id={server_id}")
            return parent.check_for_item(item_id)
        else:  # iterate through circle (theoretically, in practice we assume we are already at server "server_id")
            parent = parent.child_pointer

def search_update_queue(circle, debug, event_seq, curr_event):
    """
    Search for the server which holds the item that has been accessed.
    Once found, add an access event to the item on this server to the queue.
    """
    # determine the id of next server (either forward in circle or start from 0 again)
    next_server_id = circle.get_next_server_id(curr_server_id=curr_event.inserting_server)

    # try for each server on circle until we find the item
    for i in range(0, len(circle.present_server_ids)):
        if lookup(circle=circle, debug=debug, item_id=curr_event.item_id, server_id=next_server_id):
            if debug:
                print(
                    "Adding access-event of item " + str(curr_event.item_id) + " in server " + str(
                        next_server_id))
            event_seq.insert(e.AccessEvent(id=len(event_seq),
                                           timestamp=curr_event.timestamp + datetime.timedelta(
                                               seconds=random.randint(1, 10)),
                                           item_id=curr_event.item_id,
                                           inserting_server=next_server_id), init=False)
            break

        # if next_server does not contain the item, jump to next server
        next_server_id = circle.get_next_server_id(curr_server_id=next_server_id)
        if i == len(circle.present_server_ids) - 1:
            raise Exception(f"Item {curr_event.item_id}, "
                            f"h={circle.get_item_position(curr_event.item_id)} "
                            f"not found in whole circle, should not happen")

def get_preloaded_items(sequence):
    destinations = []
    for r in sequence:
        if param.exp_type == "ctr":
            if r[0] not in destinations:  # ctr
                destinations.append(r[0])
        elif param.exp_type == "temp":
            if r not in destinations:  # temp
                destinations.append(r)
    if param.unbounded_capacity:
        selected_items = []
        #selected_items = random.sample(destinations, int(len(destinations)))
        for i in range(int(len(destinations))):
            selected_items.append(destinations[i])
    else:
        selected_items = []
        for i in range(int(len(destinations) * param.initial_occup_factor)):
            selected_items.append(destinations[i])
        #selected_items = random.sample(destinations, int(len(destinations) * param.initial_occup_factor))
    #print("Selected " + str(len(selected_items)) + " items to insert beforehand")
    #print(selected_items[:50])
    return selected_items