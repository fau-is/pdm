
"""The module implements a data structure to represent an event log"""
import enum
import opyenxes.data_in.XesXmlParser as XESParser
import opyenxes.model.XAttributable as xA


class Violations(enum.Enum):
    """
    This class reflects the three classes of violations
    Type 0 - Everything is fine
    Type 1 - Condition
    Type 2 - Milestone
    Type 3 - Excluded
    Type 4 - Role
    Type 5 - Condition + Milestone
    Type 6 - Condition + Excluded
    Type 7 - Condition + Role
    Type 8 - Milestone + Excluded
    Type 9 - Milestone + Role
    Type 10 - Excluded + Role
    Type 11 - Condition + Milestone + Excluded
    Type 12 - Condition + Milestone + Role
    Type 13 - Condition + Excluded + Role
    Type 14 - Milestone + Excluded + Role
    Type 15 - Condition + Milestone + Excluded + Role
    Type 16 - Missing event
    """
    Type0 = 0
    Type1 = 1
    Type2 = 2
    Type3 = 3
    Type4 = 4
    Type5 = 5
    Type6 = 6
    Type7 = 7
    Type8 = 8
    Type9 = 9
    Type10 = 10
    Type11 = 11
    Type12 = 12
    Type13 = 13
    Type14 = 14
    Type15 = 15
    Type16 = 16


class EventLog(object):
    """ Represents a whole event log """

    event_log = None
    detailed = True
    def __init__(self):
        """
        Default constructor for a whole event log
        """
        self.Traces = []
        self.AttributeNames = set()

    def append_trace(self, trace):
        """
        Appends a trace to the event log from the XES file
        :param trace: the trace that is added
        """
        self.Traces.append(trace)

    def print_event_log(self):
        """
        Is used to print the log for test purposes
        :return: None
        """
        for trace in self.Traces:
            print("In Trace {} there are {} Events: \n ".format(self.Traces.index(trace), len(trace.Events)))
            for event in trace.Events:
                print("     Event Info|| Event Name: {}      Event Role: {}     Time_stamp: {}".format(
                    event.EventName, event.Role, event.Timestamp))

    @staticmethod
    def create_event_log_xes(handler: XESParser, custom_role, detailed=True):
        """
        static method to create the whole event log
        :param handler: opyenxes file handler
        :return: event log
        """
        event_log = EventLog()
        trace_list = handler[0]
        EventLog.detailed = detailed

        trace_id = 0
        for trace in trace_list:
            trace_tmp = Trace(trace_id)
            trace_id += 1
            for event in trace:
                # get the attributes for an event
                attributes: xA = event.get_attributes()
                if attributes.get('lifecycle:transition').get_value().lower() != 'complete':
                    continue
                event_timestamp = None
                event_name = str()
                event_role = str()
                if 'concept:name' in attributes:
                    event_name = attributes['concept:name']
                    attributes.pop('concept:name')

                # Todo Temporary solution
                # If roles are given in org:roles use those otherwise it is assumed that the organizational resource
                # is used
                if custom_role is not None and custom_role in attributes:
                    event_role = attributes.get(custom_role).get_value()
                    attributes.pop(custom_role)
                elif 'org:role' in attributes:
                    event_role = attributes.get('org:role').get_value()
                    attributes.pop('org:role')
                elif 'org:resource' in attributes:
                    event_role = attributes.get('org:resource').get_value()
                    attributes.pop('org:resource')
                if 'time:timestamp' in attributes:
                    event_timestamp = attributes.get('time:timestamp').get_value()
                    attributes.pop('time:timestamp')
                elif 'end' in attributes:
                    event_timestamp = attributes.get('end').get_value()
                    attributes.pop('end')
                for key in attributes.keys():
                    event_log.AttributeNames.add(key)
                event_tmp = Event(event_name, event_role, event_timestamp, attributes)
                trace_tmp.append_event(event_tmp)
            # append the event to the trace
            if len(trace_tmp.Events) > 0:
                event_log.append_trace(trace_tmp)
        return event_log


class Event(object):
    """
    The event class that represents an atomic event
    """

    def __init__(self, event_name, event_role, time_stamp, attributes=None):
        """
        Default constructor of the Event class
        """
        self.EventName = event_name
        self.Role = event_role
        self.Timestamp = time_stamp
        self.Attributes = attributes
        self.__violation = Violations.Type0

    def change_conf(self, violation):
        if self.__violation is Violations.Type0:
            self.__violation = violation
        elif EventLog.detailed:
            if violation is Violations.Type1:
                if self.__violation is Violations.Type2:
                    self.__violation = Violations.Type5
                elif self.__violation is Violations.Type3:
                    self.__violation = Violations.Type6
                elif self.__violation is Violations.Type4:
                    self.__violation = Violations.Type7
                elif self.__violation is Violations.Type8:
                    self.__violation = Violations.Type11
                elif self.__violation is Violations.Type9:
                    self.__violation = Violations.Type12
                elif self.__violation is Violations.Type10:
                    self.__violation = Violations.Type13
                elif self.__violation is Violations.Type14:
                    self.__violation = Violations.Type15

            elif violation is Violations.Type2:
                if self.__violation is Violations.Type1:
                    self.__violation = Violations.Type5
                elif self.__violation is Violations.Type3:
                    self.__violation = Violations.Type8
                elif self.__violation is Violations.Type4:
                    self.__violation = Violations.Type9
                elif self.__violation is Violations.Type6:
                    self.__violation = Violations.Type11
                elif self.__violation is Violations.Type7:
                    self.__violation = Violations.Type12
                elif self.__violation is Violations.Type10:
                    self.__violation = Violations.Type14
                elif self.__violation is Violations.Type13:
                    self.__violation = Violations.Type15

            elif violation is Violations.Type3:
                if self.__violation is Violations.Type1:
                    self.__violation = Violations.Type6
                elif self.__violation is Violations.Type2:
                    self.__violation = Violations.Type8
                elif self.__violation is Violations.Type4:
                    self.__violation = Violations.Type10
                elif self.__violation is Violations.Type5:
                    self.__violation = Violations.Type11
                elif self.__violation is Violations.Type7:
                    self.__violation = Violations.Type13
                elif self.__violation is Violations.Type9:
                    self.__violation = Violations.Type14
                elif self.__violation is Violations.Type12:
                    self.__violation = Violations.Type15

            elif violation is Violations.Type4:
                if self.__violation is Violations.Type1:
                    self.__violation = Violations.Type7
                elif self.__violation is Violations.Type2:
                    self.__violation = Violations.Type9
                elif self.__violation is Violations.Type3:
                    self.__violation = Violations.Type10
                elif self.__violation is Violations.Type5:
                    self.__violation = Violations.Type12
                elif self.__violation is Violations.Type6:
                    self.__violation = Violations.Type13
                elif self.__violation is Violations.Type8:
                    self.__violation = Violations.Type14
                elif self.__violation is Violations.Type11:
                    self.__violation = Violations.Type15

    def get_violation(self):
        return self.__violation

    @staticmethod
    def create_end_event(time_stamp, pviolation):
        """
        Creates an artificial End event
        :param time_stamp: 1 microsecond later than the actual recorded last event
        :param pviolation: pending activity violation
        :return:
        """
        event = Event("End", "", time_stamp)
        if pviolation:
            event.__violation = Violations.Type16
        return event


class Trace(object):
    """
    Represents a trace of a log
    """

    def __init__(self, trace_id):
        """
        Default constructor of a Trace, a trace is a set of related events often also called a case.
        A case is the instance of one process at a certain time.
        """
        self.Events = []
        self.TraceId = trace_id

    def append_event(self, event):
        """
        Append an event to the log trace
        :param event: The event that is added to the log trace
        :return:
        """
        self.Events.append(event)
