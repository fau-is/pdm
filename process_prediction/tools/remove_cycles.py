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


name_event_log = "bpic2017.csv"
data = load_data("../data/%s" % name_event_log)
data_no_cycles = []

for row_index in range(data.shape[0]):
    # print(data[row_index, :])

    if row_index == 0:
        data_no_cycles.append(load_data_headline("../data/%s" % name_event_log))

    if data[row_index, 0] == data[row_index - 1, 0] and data[row_index, 1] == data[row_index - 1, 1]:
        pass  # print("Cycle found...")
    else:
        data_no_cycles.append(data[row_index, :])

# print(data_no_cycles)
write_to_file("%s_no_cycles.csv" % name_event_log[:-4], data_no_cycles)
