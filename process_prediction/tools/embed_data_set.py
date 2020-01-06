import csv
import gensim
import numpy as np

if __name__ == '__main__':
    
    file = open("../data/helpdesk.csv", 'r')
    reader = csv.reader(file, delimiter=';', quotechar='|')
    next(reader, None)
    data_set = list()
    embedding_size = 60
    epochs = 1

    # create data set
    for row in reader:
        data_set.append(row[1])

    print(0)

    # train model
    model = gensim.models.Word2Vec(data_set, size=embedding_size, window=3, min_count=0)

    for epoch in range(epochs):
        if epoch % 2 == 0:
            print('Now training epoch %s' % epoch)
        model.train(data_set, total_examples=len(data_set), epochs=epochs)
        model.alpha -= 0.002  # decrease learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
                      
    # save
    model.save('../checkpoints/embeddings.model', sep_limit=2000000000)

    """
    # load model
    model = gensim.models.Doc2Vec.load('checkpoints/'+str(0)+'_context_attributes_doc2vec_2d'+str(embedding_size)+'.model')    
        
    # apply embedding model and save data set
    for document_context_str in documents_context_str: 
        try: documents_context_emb.append(model.infer_vector(document_context_str.words)) 
        except: 
            documents_context_emb.append([0]*embedding_size)
            print(document_context_str.words, 'not found')

    # concate
    data_set_new = np.zeros((len(data_set), 3 + embedding_size), dtype=np.dtype('U20'))

    # fill data
    for index in range(0, len(data_set)):
        # process
        for sub_index_process in range(0,3):
            data_set_new[index, sub_index_process] = data_set[index][sub_index_process]
        # context
        for sub_index_context in range(0, embedding_size):
            data_set_new[index, sub_index_context+3] = documents_context_emb[index][sub_index_context]

    # write dataset
    with open("../data/embeded.csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["case", "event", "time"])
    
        for row in data_set_new:
            try: spamwriter.writerow(['{:f}'.format(cell) for cell in (row)])
            except:
                spamwriter.writerow(['{:s}'.format(cell) for cell in (row)])
    """