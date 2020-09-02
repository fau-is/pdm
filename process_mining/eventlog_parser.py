# coding=utf-8
"""
This module is used to parse an event log and afterwards bring it to the data structure contained in ``eventlog.py``
"""
import opyenxes.data_in.XesXmlParser as XesParser
import opyenxes.data_out.XesXmlSerializer as XesSerializer
import csv
import datetime
from process_mining.eventlog import EventLog, Violations


def write_csv_file_to_disk(eventlog: EventLog, output_path):
    """
    Gets the EventLog class and writes it into a CSV file
    :param eventlog: Eventlog data structure to be exported
    :return: None
    """
    file = open(output_path + "train_new_hb_pcm.csv", 'w', newline='')
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
            writer.writerow(event_dict)


def write_csv_file_to_disk2(eventlog: EventLog, output_path):
    """
    Gets the EventLog class and writes it into a CSV file
    :param eventlog: Eventlog data structure to be exported
    :return: None
    """
    file = open(output_path + "train_new_hb_pcm_shift.csv", 'w', newline='')
    labels = ["case", "event", "timestamp", "conformance"]
    writer = csv.DictWriter(file, labels, dialect="excel")
    writer.writeheader()

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
            writer.writerow(event_dict)
            previous = event
            if event.EventName == "End":
                previous = None



# def write_csv_pcm2_file_to_disk(eventlog: EventLog, output_path):
#     """
#     Gets the EventLog class and writes it into a CSV file
#     :param eventlog: Eventlog data structure to be exported
#     :return: None
#     """
#     file = open(output_path + "_2_withEnd.csv", 'w', newline='')
#     labels = ["case", "event", "timestamp", "violation"]
#     writer = csv.DictWriter(file, labels, dialect="excel")
#     writer.writeheader()
#
#     for trace in eventlog.Traces:
#         if len(trace.Events) <= 2:
#             continue
#         trace_iter = iter(trace.Events)
#         for event in trace_iter:
#             event_dict = {"case": trace.TraceId, "event": event.EventName,
#                           "timestamp": event.Timestamp.strftime("%d.%m.%y-%H:%M:%S")}
#             if event.get_violation() == Violations.Type2:
#                 event_dict["violation"] = event.get_violation().value
#             else:
#                 event_dict["violation"] = Violations.Type0.value
#             writer.writerow(event_dict)
#
#
# def write_csv_pcm3_file_to_disk(eventlog: EventLog, output_path):
#     """
#     Gets the EventLog class and writes it into a CSV file
#     :param eventlog: Eventlog data structure to be exported
#     :return: None
#     """
#     file = open(output_path + "_1_and_2_withEnd.csv", 'w', newline='')
#     labels = ["case", "event", "timestamp", "violation"]
#     writer = csv.DictWriter(file, labels, dialect="excel")
#     writer.writeheader()
#
#     for trace in eventlog.Traces:
#         if len(trace.Events) <= 2:
#             continue
#         trace_iter = iter(trace.Events)
#         for event in trace_iter:
#             event_dict = {"case": trace.TraceId, "event": event.EventName,
#                           "timestamp": event.Timestamp.strftime("%d.%m.%y-%H:%M:%S"),
#                           "violation": event.get_violation().value}
#             writer.writerow(event_dict)


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
