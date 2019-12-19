# Proposed new structure for preprocessing in next-event-API

## Taxonomy

The preprocess script processes an event log. \
An event log consists of process instances. Each process instance consists of a sequence of events and has a unique ID.\
Events have a unique ID and consist of attributes: 
- An event always consist of control flow attributes (process instance ID, event ID, event timestamp). 
- In addition, an event can have context attributes (e.g. resource).


### Renaming variables in preprocess script

old variable name 				| new variable name
--- 							| --- 
caseids 						| ids_process_instances
char_indices 					| map_event_label_to_event_id				
char 							| event_label
chars 							| event_labels
check_additional_features 		| check_of_context_attributes (removed)
csvfile 						| eventlog_csvfile
elems_per_fold 					| elements_per_fold 
features_additional_attributes 	| context_attributes_event 
features_additional_events		| context_attributes_process_instance
features_additional_sequences 	| context_attributes_process_instances 
firstLine 						| first_event_of_process_instance
indices_char 					| map_event_id_to_event_label			
lastcase 						| id_latest_process_instance
line 							| process_instance
lines 							| process_instances, process_instances_train bzw. process_instances_test
lines_add 						| context_attributes_train bzw. context_attributes_test
max_sequence_length 			| max_length_process_instance
next_chars 						| event_labels
num_attributes_standard			| num_attributes_control_flow
num_features_activities 		| num_event_ids
num_features_additional 		| num_attributes_context
num_features_all 				| num_features
numlines 						| num_process_instances
row								| event
sentences 						| cropped_event_sequences
sentences_add					| cropped_context_attribute_sequences
spamreader 						| eventlog_reader
step 							| timestep (removed)
target_chars (remove?)			| event_types 					
target_char_indices (remove?)	| map_event_type_to_event_id		
target_indices_char (remove?)	| map_event_id_to_event_type		
X								| data_set
Y								| labels


Use constant END_OF_PROCESS_INSTANCE = '!' to mark end of process_instance


### Exemplary code block with new taxonomy

~~~python
eventlog_csvfile = open(self.data_dir, 'r')
eventlog_reader = csv.reader(eventlog_csvfile, delimiter=';', quoteevent_label='|')

next(eventlog_reader, None)  
for event in eventlog_reader:
    
    # initial setting of context attributes
    if check_of_context_attributes == True:
        if len(event) == self.num_attributes_control_flow:
            util.llprint("No additional attributes.\n")
        else:
            self.num_attributes_context = len(event) - self.num_attributes_control_flow     
            util.llprint("Number of additional attributes: %d\n" % self.num_attributes_context)
        check_of_context_attributes = False
        
    if event[0]!=id_latest_process_instance:
        ids_all_process_instances.append(event[0])
        id_latest_process_instance = event[0]
        if not first_event_of_process_instance:
            process_instances_all.append(process_instance)
            if self.num_attributes_context > 0:
                context_attributes_all_process_instances.append(context_attributes_process_instance)
        process_instance = ''
        if self.num_attributes_context > 0:
            context_attributes_process_instance = []
        counter_events+=1
        
    # get values of context attributes
    if self.num_attributes_context > 0:
        for index in range(self.num_attributes_control_flow, self.num_attributes_control_flow + self.num_attributes_context):
            context_attributes_event.append(event[index])
        context_attributes_process_instance.append(context_attributes_event)    
        context_attributes_event = []
    
    # add event id to a process instance
    process_instance+=chr(int(event[1])+self.ascii_offset) 
    first_event_of_process_instance = False

process_instances_all.append(process_instance)
if self.num_attributes_context > 0:
    context_attributes_all_process_instances.append(context_attributes_process_instance)
counter_events+=1
~~~
