import numpy as np


def encode_full_from_midi(s, encoder, vocab):
    return one_hot_encode_from_magenta(encode_into_magenta(s, encoder), vocab)


def decode_full_from_one_hot(s, encoder, vocab):
    """returns path"""
    return decode_from_magenta(one_hot_decode_into_magenta(s, vocab), encoder)


def decode_from_magenta(s, encoder):
    """decode from 1d magenta format into midi. return path to file"""
    return encoder.decode(s)


def encode_into_magenta(f, encoder):
    """encode midi into 1d magenta format. f is a path"""
    return encoder.encode(f)


def one_hot_decode_into_magenta(s, vocab_path):
    vocab = np.load(vocab_path)
    reconstructed_seq = np.repeat(-1, len(s))
    for i, step in enumerate(s):
        instruction_index = np.argmax(step)
        instruction = vocab[instruction_index]
        reconstructed_seq[i] = instruction
    return reconstructed_seq


def one_hot_encode_from_magenta(s, vocab_path):
    vocab = np.load(vocab_path)
    new_seq = np.zeros((len(s), len(vocab)), dtype='uint8')
    for instruction_index in range(len(s)):
        instruction = s[instruction_index]
        try:
            instruction_index_in_vocab = np.where(vocab == instruction)[0][0]
        except IndexError as e:
            print(e)
            print('could not find instruction %s' %instruction)
        new_seq[instruction_index, instruction_index_in_vocab] = 1
    return new_seq
