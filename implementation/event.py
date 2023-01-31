

class Event:
    def __init__(self, id, timestamp):
        self.id = id
        self.timestamp = timestamp
        self.type = "no_type"
        self.item_id = -99

    def __str__(self):
        return "event-id:" + str(self.id)+", " +self.get_type() + " item " + str(self.item_id) + " time:" + str(
            self.timestamp)

    def print(self):
        print(self.get_type()+" event, id:"+str(self.id)+" item "+str(self.item_id)+" time:"+str(self.timestamp))

    def get_type(self):
        return self.type


class DeletingEvent(Event):
    def __init__(self, id, timestamp, item_id):
        super().__init__(id, timestamp)
        self.type = "deleting"
        self.item_id = item_id

    def get_type(self):
        return self.type


class AddingEvent(Event):
    def __init__(self, id, timestamp, item_id):
        super().__init__(id, timestamp)
        self.type = "adding"
        self.item_id = item_id

    def get_type(self):
        return self.type


class OutsideAccessEvent(Event):
    def __init__(self, id, timestamp, item_id):
        super().__init__(id, timestamp)
        self.type = "outside-access"
        self.item_id = item_id

    def get_type(self):
        return self.type


class AccessEvent(Event):
    def __init__(self, id, timestamp, item_id, inserting_server):
        super().__init__(id, timestamp)
        self.type = "access"
        self.item_id = item_id
        self.inserting_server = inserting_server

    def get_type(self):
        return self.type
