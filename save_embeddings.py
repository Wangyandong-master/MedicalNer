import numpy as np

def read_embeddings_vocab(filename):
    vocab = set()
    with open(filename,encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.strip().split('\t')
            word = line[0]
            vocab.add(word)
    return vocab

def get_embeds_mixed_chars_words(embedding_filename_of_words, embedding_filename_of_chars, vocab_words_mixed_chars, out_embedding_file, npz_filename):
    embeddings_mixed_chars_words = None
    len_vocab = len(vocab_words_mixed_chars)
    dim = 0
    num = 0
    with open(embedding_filename_of_words, encoding='utf-8') as f_word:

        for i, line_word in enumerate(f_word):
            line_word = line_word.strip().split('\t')
            if i == 0:
                _, dim = line_word[0], int(line_word[1])
                embeddings_mixed_chars_words = np.zeros([len_vocab, dim])
                continue

            word = line_word[0]
            embedding_word = [float(x) for x in line_word[1:]]
            if word in vocab_words_mixed_chars:
                word_idx = vocab_words_mixed_chars[word]
                embeddings_mixed_chars_words[word_idx] = np.asarray(embedding_word)

    with open(embedding_filename_of_chars, encoding='utf-8') as f_char:

        for i, line_char in enumerate(f_char):
            line_char = line_char.strip().split('\t')
            if i == 0:
                continue

            char = line_char[0]
            embedding_char = [float(x) for x in line_char[2:]]

            if char in vocab_words_mixed_chars:
                word_idx = vocab_words_mixed_chars[char]
                if embeddings_mixed_chars_words[word_idx][1] > 0 and embeddings_mixed_chars_words[word_idx][2] > 0:
                    num += 1
                    embeddings_mixed_chars_words[word_idx] = (embeddings_mixed_chars_words[word_idx] + np.asarray(
                        embedding_char)) / 2
                else:
                    embeddings_mixed_chars_words[word_idx] = np.asarray(embedding_char)



    with open(out_embedding_file, 'w', encoding='utf-8') as outf:
        outf.write('%d\t%d\n' % (len_vocab, dim))
        for key, i in vocab_words_mixed_chars.items():
            outf.write('%s' % (key))
            if embeddings_mixed_chars_words[i][0] ==0 and embeddings_mixed_chars_words[i][2] == 0:
                embeddings_mixed_chars_words[i] = np.random.normal(0,1,dim)
            for j in range(dim):
                outf.write('\t{:.6f}'.format(embeddings_mixed_chars_words[i][j]))
            outf.write('\n')
    np.savez_compressed(npz_filename, embeddings=embeddings_mixed_chars_words)


def save_word_embeddings(embedding_filename, npz_filename, vocab):
    embeddings = None
    with open(embedding_filename, encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip().split('\t')

            if i == 0:
                _, dim = line[0], int(line[1])
                embeddings = np.zeros([len(vocab), dim])
                continue

            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(npz_filename, embeddings=embeddings)

def load_embeddings(npz_filename):
    print("load embeddings {} ...".format(npz_filename))
    try:
        with np.load(npz_filename) as data:
            return data["embeddings"]

    except IOError:
        raise "Unable to load file:" + npz_filename


def save_char_embeddings(embedding_filename, npz_filename, vocab):
    embeddings = None
    with open(embedding_filename, encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip().split('\t')
            if i == 0:
                _, dim = line[0], int(line[1])
                embeddings = np.random.randn(len(vocab), dim)
                continue
            word = line[0]
            embedding = [float(x) for x in line[2:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    np.savez_compressed(npz_filename, embeddings=embeddings)