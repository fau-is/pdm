# coding=utf-8
"""
The module contains the conformance data informations
"""
import threading


class ConformanceAnalysisData(object):
    """
    The overall conformance analysis data contains information about all failed traces in an event log in an aggregated
    form
    """
    conformance_analysis_data = None

    def __init__(self):
        """
        Default initializer
        """
        self.ViolatingTraces = []
        self.ViolatedActivities = {}
        self.ViolatedConnections = {}
        self.ViolatedExcluded = {}
        self.ViolatedPending = {}
        self.ViolatedRoles = {}
        self.ViolatedTraces = {}
        self.Lock = threading.RLock()  # Used to make the class thread safe
        ConformanceAnalysisData.conformance_analysis_data = self

    def create_violated_traces_dict(self):
        """
        Creates a dict from the Violating traces
        :return: a dict with the process paths
        """
        process_paths = {}
        for trace in self.ViolatingTraces:
            process_path = str()
            for event in trace.Events:
                process_path += "  -{}".format(event.EventName)
            process_path = process_path.lstrip()[1:]
            if process_path in process_paths:
                process_paths[process_path] += 1
            else:
                process_paths[process_path] = 1
        return process_paths

    def append_violating_trace(self, trace):
        """
        Appends a violating trace to the ViolatingTraces array
        :param trace: the trace to be added
        :return:
        """
        self.ViolatingTraces.append(trace)

    def append_conformance_data(self, trace_conformance_data):
        """
        Thread safe method to add trace conformance analysis data to the overall conformance analysis data
        :type trace_conformance_data: TraceConformanceAnalysisData that will be added to the overall information
        """
        # Acquire lock for thread safe execution
        self.Lock.acquire()

        if trace_conformance_data.Trace is not None:
            self.ViolatingTraces.append(trace_conformance_data.Trace)
        for activity in trace_conformance_data.ViolatingActivities:
            if activity in self.ViolatedActivities:
                self.ViolatedActivities[activity] += 1
            else:
                self.ViolatedActivities[activity] = 1
        for connection in trace_conformance_data.ViolatingConstraints:
            if connection in self.ViolatedConnections:
                self.ViolatedConnections[connection] += 1
            else:
                self.ViolatedConnections[connection] = 1
        for excluded_execution in trace_conformance_data.ViolatingExcluded:
            if excluded_execution in self.ViolatedExcluded:
                self.ViolatedExcluded[excluded_execution] += 1
            else:
                self.ViolatedExcluded[excluded_execution] = 1
        for pending_marking in trace_conformance_data.ViolatingPending:
            if pending_marking in self.ViolatedPending:
                self.ViolatedPending[pending_marking] += 1
            else:
                self.ViolatedPending[pending_marking] = 1
        for role in trace_conformance_data.ViolatingRoles:
            if role in self.ViolatedRoles:
                self.ViolatedRoles[role] += 1
            else:
                self.ViolatedRoles[role] = 1

        del trace_conformance_data

        # Release lock for next thread
        self.Lock.release()


class TraceConformanceAnalysisData(object):
    """
    This class is used to capture all failing constraints of a trace in a DCR graph
    """

    def __init__(self, trace):
        """
        Default initializer for all violations in a Trace
        :param trace: trace that is added to the trace data
        """
        self.Trace = trace
        self.HasViolations = False
        self.ViolatingActivities = []
        self.ViolatingConstraints = []
        self.ViolatingExcluded = []
        self.ViolatingPending = []
        self.ViolatingRoles = []
        self.ViolatingNestingActivityCall = []
        self.ViolatingNestingActivityBlocked = []

    def add_violating_role(self, role, node_role):
        """
        Adds a violating role to the list of violating role violations
        :param role: the role that was used
        :param node_role: the role that should have been used
        :return: None
        """
        self.HasViolations = True
        self.ViolatingRoles.append('{} instead of {}'.format(role, node_role))

    def add_violating_activity(self, activity):
        """
        Adds an execution of a violating activity
        :param activity: the activity that was executed even though it was excluded
        :return: None
        """
        self.HasViolations = True
        self.ViolatingActivities.append(activity)

    def add_violating_connection(self, connection):
        """
        Adds a violated connection (either Milestone or Condition) to the violations
        :param connection: the violated connection
        :return: None
        """
        self.HasViolations = True
        self.ViolatingConstraints.append(connection)

    def add_violating_pending(self, pending):
        """
        Adds an activity to the list of activities that were still in pending in the process execution
        :param pending: The Activity to add
        :return: None
        """
        self.HasViolations = True
        self.ViolatingPending.append(pending)

    def add_violating_nesting_activity_occurred(self, nesting_activity):
        """
        Adds an activity to the list of nesting activities that occurred in the event log
        :return: None
        """
        self.HasViolations = True
        self.ViolatingNestingActivityCall.append(nesting_activity)

    def add_violating_nesting_activity_blocked(self, activity):
        """
        Add a tuple to
        :param activity:
        :return: None
        """
        self.HasViolations = True
        self.ViolatingNestingActivityBlocked.append(tuple((activity, activity.NestingActivity)))

    def calculate_trace_alignment_fitness(self):
        """
        TODO Future Work
        :return: None
        """
        pass
