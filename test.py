from model.data_utils import CoNLLDataset, get_processing_word, get_glove_vocab
from model.ner_model import NERModel
from model.config import Config


def main():

    config = Config()



    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets [(char_ids), word_id]
    processing_word = get_processing_word(lowercase=False)
    dev   = CoNLLDataset(config.filename_dev, processing_word)
    train = CoNLLDataset(config.filename_train, processing_word)
    test = CoNLLDataset(config.filename_test, processing_word)
    entities = []
    for raw_words, raw_tags in test:
        chunks = get_chunks_from_tags(raw_tags)
        for _, chunk_start, chunk_end in chunks:
            entity = 'ENTITY/'
            for i in range(chunk_start, chunk_end):
                if i == chunk_end-1:
                    entity += raw_words[i]
                else:
                    entity = entity + raw_words[i] + '_'
            entities.append(entity)
    # print(len(entities))
    # print(entities)

    entities = set(entities)
    print(len(entities))
    vocab_glove = get_glove_vocab(config.filename_glove)
    print(len(entities & vocab_glove))




def get_chunks_from_tags(tags):
    '''

    :param tags: [B-LOC,I-LOC,O,B-PER]
    :return: [("LOC",0,2),("PER",3,4)]
    '''
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(tags):
        # End of a chunk 1
        if tok == 'O' and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != 'O':
            tok_chunk_class, tok_chunk_type = tok.split('-')
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(tags))
        chunks.append(chunk)

    return chunks

if __name__ == '__main__':
    main()