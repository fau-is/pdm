import pandas as pd
import csv


def load_data(file_path):
    return pd.read_csv(file_path, sep=";").to_numpy()


def load_data_headline(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        headline = reader.fieldnames
    file.close()
    return headline


def write_to_file(file_path, data):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    file.close()


name_event_log = "bpi2019_converted_sample.csv"
data = load_data("../data/%s" % name_event_log)

number_events_instances = []
number_events_instance = []

for row_index in range(data.shape[0]):

    if row_index > 0:

        # calc metrics
        if data[row_index, 0] == data[row_index - 1, 0]:
            number_events_instance.append(1)
        else:
            number_events_instances.append(sum(number_events_instance))
            number_events_instance = [1]

print("Number of events %i\n" % data.shape[0])
print("Number of instances %i\n" % len(set(data[:, 0])))
print("Number of activities %i\n" % len(set(data[:, 1])))
print("Max length of instances %i\n" % max(number_events_instances))
print("Min length of instances %i\n" % min(number_events_instances))
print("Avg length of instances %f\n" % (sum(number_events_instances) / len(number_events_instances)))
