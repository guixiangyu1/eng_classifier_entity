import numpy as np
import os


# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)



class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()            #去掉前后的空格
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0],ls[-1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]


    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()

    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            tags = [tag.split('-')[1] for tag in tags if tag!='O']
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags

def entity2vocab(datasets):
    print("Add entity to vocab $@&")
    chunk = ""
    vocab = set()
    for dataset in datasets:
        for words, tags in dataset:
            for word, tag in zip(words, tags):
                tag = tag.split('-')[0]

                if tag == 'B':
                    if len(chunk) == 0:
                        chunk = 'ENTITY/' + word
                    else:
                        vocab.add(chunk)
                        chunk = 'ENTITY/' + word
                if tag == 'I':
                    assert len(chunk) != 0
                    chunk = chunk + "_" + word
                if tag == 'O':
                    if len(chunk) != 0:
                        vocab.add(chunk)
                        chunk = ""
            if len(chunk) != 0:
                vocab.add(chunk)
                chunk = ""

    print("- done. {} entities".format(len(vocab)))
    return vocab

def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()      #若没有这个strip（），则每个词包含一个换行符
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.zeros([len(vocab), dim])   # 对于glove中无定义的word，embedding为全部为0
    print(len(vocab))

    with open(glove_filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    for keyword in vocab:
        embedding_total = []
        if keyword.startswith('ENTITY/'):
            keyword_index = vocab[keyword]
            if (embeddings[keyword_index] == np.zeros([dim])).all():
                keyword = keyword[7:].split("_")
                for word in keyword:
                    word = word.lower()
                    if word in vocab:
                        word_idx = vocab[word]
                    else:
                        word_idx = vocab["$UNK$"]
                    embedding_total.append(embeddings[word_idx])
                embeddings[keyword_index] = np.mean(embedding_total, axis=0)



    np.savez_compressed(trimmed_filename, embeddings=embeddings)    #压缩词嵌入到文件中，并且名字为embeddings


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            if word.startswith('ENTITY/'):
                word_for_char = word[7:]
            else:
                word_for_char = word
            # entity_words = word[7:].split("_")
            for char in word_for_char:
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]      #vocab_chars是个dict， list的+方法，就是往里添加元素[1]+[2]=[1,2]

        # 1. preprocess word
        if lowercase and (word.startswith('ENTITY/') is False):
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            # tuple
            return char_ids, word
        else:
            return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        #提取出最大词长
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch, z_batch = [], [], []
    for (x, y, z) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch, z_batch
            x_batch, y_batch, z_batch = [], [], []
        # data may be (list of (list of char_id, word_id), list of (tags_id))
        if type(x[0]) == tuple:
            x = zip(*x)  # zip(*x) 生成可迭代对象。 x本身是个list，*代表变长参数输入，几个tuple.最后得到x是一个对象,俩tuple
        x_batch += [x]   # 可迭代对象的一个list，每个对象x包含一个tuple(list[char_ids]),tuple(word_ids)的迭代器
        y_batch += [y]
        z_batch += [z]

    if len(x_batch) != 0:
        yield x_batch, y_batch, z_batch


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
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
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

def CoNLLdata4classifier(dataset, processing_word=None, processing_tag=None):
    '''
    transit CoNLLdata into classifier style, (words,tags,masks)
    :param datasets:
    :return:
    '''
    words, tags, masks = [], None, []
    data = []
    for raw_words, raw_tags in dataset:
        chunks = get_chunks_from_tags(raw_tags)
        for ne_type, ne_start, ne_end in chunks:
            tags = processing_tag(ne_type)

            for i, word in enumerate(raw_words):

                if i<ne_start or i>=ne_end:
                    words.append(processing_word(word))
                    masks.append(False)
                else:
                    if i==ne_start:
                        entity = 'ENTITY/' + raw_words[i]
                    else:
                        entity = entity + '_' + raw_words[i]

                    if i==ne_end-1:
                        words.append(processing_word(entity))
                        masks.append(True)
            data.append((words, tags, masks))
            words, tags, masks = [], None, []
    return data






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






