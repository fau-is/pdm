# coding=utf-8
"""
This module is used to parse an event log and afterwards bring it to the data structure contained in ``eventlog.py``
"""
import opyenxes.data_in.XesXmlParser as XesParser
import opyenxes.data_out.XesXmlSerializer as XesSerializer
import csv

from process_mining.eventlog import EventLog


def write_csv_file_to_disk(eventlog: EventLog, output_path):
    """
    Gets the EventLog class and writes it into a CSV file
    :param eventlog: Eventlog data structure to be exported
    :return: None
    """
    file = open(output_path, 'w', newline='')
    labels = ["case", "event", "timestamp", "conformance"]
    writer = csv.DictWriter(file, labels)
    writer.writeheader()

    for trace in eventlog.Traces:
        if len(trace.Events) <= 2:
            continue
        trace_iter = iter(trace.Events)
        for event in trace_iter:
            if event.EventName is not "End":
                event_dict = {"case": trace.TraceId, "event": event.EventName, "timestamp": event.Timestamp,
                              "conformance": next(trace_iter).get_violation().value}
                writer.writerow(event_dict)


def get_event_log(file_path: str = None, use_celonis=False):
    """
    Gets the event log data structure from the event log file.
    Dispatches the methods to be used by file tyoe
    :param use_celonis: If the attribute is set to true the event log will be retrieved from celonis
    :param file_path: Path to the event-log file
    :return:EventLog data structure
    """
    if file_path is None and not use_celonis:
        raise ValueError("Parameters file_path was None and use_celonis was false at the same time."
                         "This behavior is not supported")

    file_path_lowercase = file_path.lower()
    if file_path_lowercase.endswith(".xes"):
        return __handle_xes_file(file_path)
    else:
        raise ValueError('The input file was not a XES file')


def __handle_xes_file(import_path):
    """
    Puts an xes file into a common data structure
    :param import_path: Path to the xes file
    :return: Void
    """
    opyenxes_log = __import_event_log_xes(import_path)
    return EventLog.create_event_log_xes(opyenxes_log)


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