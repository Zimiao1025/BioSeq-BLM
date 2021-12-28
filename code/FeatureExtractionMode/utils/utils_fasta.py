import sys


class Seq:
    def __init__(self, name, seq, no):
        self.name = name
        self.seq = seq.upper()
        self.no = no
        self.length = len(seq)

    def __str__(self):
        """Output seq when 'print' method is called."""
        return "%s\tNo:%s\tlength:%s\n%s" % (self.name, str(self.no), str(self.length), self.seq)


def is_under_alphabet(s, alphabet):
    """Judge the string is within the scope of the alphabet or not.

    :param s: The string.
    :param alphabet: alphabet.

    Return True or the error character.
    """
    for e in s:
        if e not in alphabet:
            return e

    return True


def is_fasta(seq):
    """Judge the Seq object is in FASTA format.
    Two situation:
    1. No seq name.
    2. Seq name is illegal.
    3. No sequence.

    :param seq: Seq object.
    """
    if not seq.name:
        error_info = 'Error, sequence ' + str(seq.no) + ' has no sequence name.'
        print(seq)
        sys.stderr.write(error_info)
        return False
    if -1 != seq.name.find('>'):
        error_info = 'Error, sequence ' + str(seq.no) + ' name has > character.'
        sys.stderr.write(error_info)
        return False
    if 0 == seq.length:
        error_info = 'Error, sequence ' + str(seq.no) + ' is null.'
        sys.stderr.write(error_info)
        return False

    return True


def read_fasta(f):
    """Read a fasta file.

    :param f: HANDLE to input. e.g. sys.stdin, or open(<file>)

    Return Seq obj list.
    """
    name, seq = '', ''
    count = 0
    seq_list = []
    lines = f.readlines()
    for line in lines:
        if not line:
            break

        if '>' == line[0]:
            if 0 != count or (0 == count and seq != ''):
                if is_fasta(Seq(name, seq, count)):
                    seq_list.append(Seq(name, seq, count))
                else:
                    sys.exit(0)

            seq = ''
            name = line[1:].strip()
            count += 1
        else:
            seq += line.strip()

    count += 1
    if is_fasta(Seq(name, seq, count)):
        seq_list.append(Seq(name, seq, count))
    else:
        sys.exit(0)

    return seq_list


def read_fasta_yield(f):
    """Yields a Seq object.

    :param f: HANDLE to input. e.g. sys.stdin, or open(<file>)
    """
    name, seq = '', ''
    count = 0
    while True:
        line = f.readline()
        if not line:
            break

        if '>' == line[0]:
            if 0 != count or (0 == count and seq != ''):
                if is_fasta(Seq(name, seq, count)):
                    yield Seq(name, seq, count)
                else:
                    sys.exit(0)

            seq = ''
            name = line[1:].strip()
            count += 1
        else:
            seq += line.strip()

    if is_fasta(Seq(name, seq, count)):
        yield Seq(name, seq, count)
    else:
        sys.exit(0)


def read_fasta_check_dna(f, alphabet):
    """Read the fasta file, and check its legality.

    :param alphabet:
    :param f: HANDLE to input. e.g. sys.stdin, or open(<file>)

    Return the seq list.
    """
    seq_list = []
    for e in read_fasta_yield(f):
        res = is_under_alphabet(e.seq, alphabet)
        if res:
            seq_list.append(e)
        else:
            error_info = 'Sorry, sequence ' + str(e.no) \
                         + ' has character ' + str(res) + '.(The character must be ' + alphabet + ').'
            sys.exit(error_info)

    return seq_list


def get_sequence_check_dna(f, alphabet):
    """Read the fasta file.

    Input: f: HANDLE to input. e.g. sys.stdin, or open(<file>)

    Return the sequence list.
    """
    sequence_list = []
    for e in read_fasta_yield(f):
        res = is_under_alphabet(e.seq, alphabet)
        if res is not True:
            print(e.name)
            error_info = 'Error, sequence ' + str(e.no) \
                         + ' has character ' + str(res) + '.(The character must be ' + alphabet + ').'
            sys.exit(error_info)
        else:
            # print(e.no)
            # print(e.name)
            # print(e.seq)
            sequence_list.append(e.seq)

    return sequence_list


def is_sequence_list(sequence_list, alphabet):
    """Judge the sequence list is within the scope of alphabet and change the lowercase to capital."""
    count = 0
    new_sequence_list = []

    for e in sequence_list:
        e = e.upper()
        count += 1
        res = is_under_alphabet(e, alphabet)
        if res is not True:
            error_info = 'Sorry, sequence ' + str(count) \
                         + ' has illegal character ' + str(res) + '.(The character must be A, C, G or T)'
            sys.stderr.write(error_info)
            return False
        else:
            new_sequence_list.append(e)

    return new_sequence_list


def get_seqs(input_file, alphabet, desc=False):
    """Get sequences data from file or list with check.

    :param alphabet: DNA, RNA or Protein
    :param input_file: type file or list
    :param desc: with this option, the return value will be a Seq object list(it only works in file object).
    :return: sequence data or shutdown.
    """
    # modified at 2020/05/10
    if hasattr(input_file, 'read'):
        if desc is False:
            return get_sequence_check_dna(input_file, alphabet)
        else:
            return read_fasta_check_dna(input_file, alphabet)  # return Seq(name, seq, count)
    elif isinstance(input_file, list):
        input_data = is_sequence_list(input_file, alphabet)
        if input_data is not False:
            return input_data
        else:
            sys.exit(0)
    else:
        error_info = 'Sorry, the parameter in get_data method must be list or file type.'
        sys.exit(error_info)
