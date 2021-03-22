
"""
This module is used to parse an event log and afterwards bring it to the data structure contained in ``eventlog.py``
"""
import opyenxes.data_in.XesXmlParser as XesParser
import csv
from process_mining.eventlog import EventLog


def write_csv_file_to_disk(eventlog: EventLog, output_path):
    """
    Gets the EventLog class and writes it into a CSV file
    :param eventlog: Eventlog data structure to be exported
    :return: None
    """
    file = open(output_path + "bpia17_pdm_shift_inserted.csv", 'w', newline='')
    labels = ["case", "event", "timestamp", "conformance"]
    writer = csv.DictWriter(file, labels, dialect="excel")
    writer.writeheader()

    for trace in eventlog.Traces:
        if len(trace.Events) < 3:
            continue
        trace_iter = iter(trace.Events)
        for event in trace_iter:
            event_dict = {"case": trace.TraceId, "event": event.EventName,
                      "timestamp": event.Timestamp.strftime("%d.%m.%y-%H:%M:%S"),
                      "conformance": event.get_violation().value}



def write_csv_file_to_disk_shift(eventlog: EventLog, output_path):
    """
    Gets the EventLog class and writes it into a CSV file
    It shifts the conformance label one to the past,
    so that it is predictable in the future
    :param eventlog: Eventlog data structure to be exported
    :return: None
    """
    file = open(output_path + "train_mobis_pcm_shift.csv", 'w', newline='')
    labels = ["case", "event", "timestamp", "conformance"]
    labels += eventlog.AttributeNames
    writer = csv.DictWriter(file, labels, dialect="excel")
    writer.writeheader()
    vio_no = {}

    for trace in eventlog.Traces:
        if len(trace.Events) < 3:
            continue
        previous = None
        for event in iter(trace.Events):
            if previous is None:
                previous = event
                continue
            event_dict = {"case": trace.TraceId, "event": previous.EventName,
                      "timestamp": previous.Timestamp.strftime("%d.%m.%y-%H:%M:%S"),
                      "conformance": event.get_violation().value}
            event_dict.update(previous.Attributes)
            writer.writerow(event_dict)
            if event.get_violation().value not in vio_no:
                vio_no[event.get_violation().value] = 1
            else:
                vio_no[event.get_violation().value] += 1
            previous = event
            if event.EventName == "End":
                previous = None

    print(vio_no)

def get_event_log(file_path, custom_role):
    """
    Gets the event log data structure from the event log file.
    Dispatches the methods to be used by file tyoe
    :param custom_role: Custom role attribute
    :param file_path: Path to the event-log file
    :return:EventLog data structure
    """

    file_path_lowercase = file_path.lower()
    if file_path_lowercase.endswith(".xes"):
        return __handle_xes_file(file_path, custom_role)
    else:
        raise ValueError('The input file was not a XES file')


def __handle_xes_file(import_path, custom_role):
    """
    Puts an xes file into a common data structure
    :param import_path: Path to the xes file
    :return: Void
    """
    opyenxes_log = __import_event_log_xes(import_path)
    return EventLog.create_event_log_xes(opyenxes_log, custom_role)


def __import_event_log_xes(import_path):
    """
    Import an event log from an xes file
    :param import_path: Path of the event log
    :return: parsed event log
    """
    xml_parser = XesParser.XesXmlParser()
    can_parse = xml_parser.can_parse(import_path)
    if can_parse:
        parsed_log = xml_parser.parse(import_path)
    else:
        raise Exception("Error: Xes-file {} cannot be parsed".format(import_path))
    return parsed_log
