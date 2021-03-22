
"""The main file of the dcr-cc that is executed"""
from threading import Thread
import datetime
import process_mining.cmd_parser as cmd_parser
import process_mining.eventlog_parser as eventlog_parser
from process_mining.conf_data import ConformanceAnalysisData, TraceConformanceAnalysisData
from process_mining.graph import DCRGraph
from process_mining.marking import Marking
from process_mining.eventlog import Event


def perform_conformance_checking(trace, ca):
    """
    The perform conformance checking method gets a trace as an input and then simulates the model with
    the constraints retrieved from the DCR graph.
    :param ca: The conformance analysis data object that is used for the overall conformance checking
    :param trace: the trace that is checked within this thread
    :return:
    """
    marking = Marking.get_initial_marking()
    trace_conformance_data = TraceConformanceAnalysisData(trace)
    for event in trace.Events:
        node = dcr_graph.get_node_by_name(event.EventName)
        marking.perform_transition_node(node, event, trace_conformance_data)
    pending_violation = False
    if len(marking.PendingResponse) != 0:
        for pending in marking.PendingResponse:
            if pending in marking.Included:
                trace_conformance_data.add_violating_pending(pending.ActivityName)
                pending_violation = True
    time_stamp = None
    if trace.Events is not []:
        time_stamp = trace.Events[-1].Timestamp + datetime.timedelta(seconds=1)

    trace.append_event(Event.create_end_event(time_stamp, pending_violation))
    if trace_conformance_data.HasViolations:
        ca.append_conformance_data(trace_conformance_data)


def main():
    """
    Program main method starts by parsing the DCR graph afterwards retrieving the Event Log
    subsequently the conformance is checked
    :return:
    """
    global dcr_graph

    # input
    dcr_graph = DCRGraph.get_graph_instance(xml_path)
    event_log = eventlog_parser.get_event_log(data_path, custom_role)
    ca = ConformanceAnalysisData()

    # throughput
    # if parallel is set: a thread pool is created
    if parallel:

        threads = []
        for trace in event_log.Traces:
            t = Thread(target=perform_conformance_checking, args=(trace, ca))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    # sequential conformance checking (Debugging purposes)
    else:
        for trace in event_log.Traces:
            perform_conformance_checking(trace, ca)
    # output
    create_conformance_output(ca, event_log)
    eventlog_parser.write_csv_file_to_disk_shift(event_log, output_pcm)



def create_conformance_output(ca, event_log):
    """
    Creates the console output of the program
    :param ca:
    :param event_log:
    :return:
    """
    if len(ca.ViolatingTraces) > 0:
        # Calculate ratios and replay fitness, Round up to two digits
        violating_case_ratio = len(ca.ViolatingTraces) / len(event_log.Traces)
        replay_fitness = 1 - violating_case_ratio
        replay_fitness = "%.2f" % replay_fitness
        violating_case_ratio *= 100
        violating_case_ratio = "%.2f" % violating_case_ratio
        conformance_ratio = 100 - float(violating_case_ratio)
        # Output
        print('All in all, {} of {} violated the process model'.format(len(ca.ViolatingTraces), len(event_log.Traces)))
        print('The ratio of violating cases is: {}%'.format(violating_case_ratio))
        print("Thus, the conformance ratio is: {}%".format(conformance_ratio))
        print("The replay fitness is: {}%".format(replay_fitness))

        # Sort the dictionaries for the descending order of occurrences
        sorted_including_violation = sorted(ca.ViolatedActivities.items(), key=lambda kv: kv[1], reverse=True)
        sorted_violated_roles = sorted(ca.ViolatedRoles.items(), key=lambda kv: kv[1], reverse=True)
        sorted_violated_pending = sorted(ca.ViolatedPending.items(), key=lambda kv: kv[1], reverse=True)
        sorted_violated_connections = sorted(ca.ViolatedConnections.items(), key=lambda kv: kv[1], reverse=True)
        sorted_violated_cases = sorted(ca.create_violated_traces_dict().items(), key=lambda kv: kv[1], reverse=True)

        # Print all detailed information
        print("\n{} process paths failed the events\n".format(len(sorted_violated_cases)))
        for process_path in sorted_violated_cases:
            print("The process path:\n\"{}\" \t was non-conformant {} times ".format(process_path[0], process_path[1]))
        for included_violation in sorted_including_violation:
            print('The activity \"{}\" has been executed {} times even though it was not included'.format(
                included_violation[0], included_violation[1]))
        for violated_role in sorted_violated_roles:
            print('The role \"{}\" was misused \"{}\" times'.format(violated_role[0], violated_role[1]))
        for violated_pending in sorted_violated_pending:
            print('The activity {} was pending at the end in {} cases'.format(violated_pending[0], violated_pending[1]))
        for violated_connection in sorted_violated_connections:
            print('The {} was violated in {} traces'.format(violated_connection[0], violated_connection[1]))
    else:
        print('The conformance ratio is 100%')


def add_dcr_graph_for_test(test_graph):
    """
    For unit tests with the main class a dcr graph can be added
    :param test_graph: the created test graph
    :return:
    """
    global dcr_graph
    dcr_graph = test_graph


if __name__ == '__main__':
    # input parameters
    args = cmd_parser.parse_args()
    data_path = args.eventLog
    xml_path = args.XmlDcr
    output_pcm = args.outputPathPCM
    custom_role = args.customRole
    parallel = False
    dcr_graph = None
    columns_work = None
    main()
