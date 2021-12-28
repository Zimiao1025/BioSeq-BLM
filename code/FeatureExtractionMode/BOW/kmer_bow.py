import sys
import re
import numpy as np
from ..utils.utils_words import make_km_list
from ..utils.utils_bow import frequency
from ..utils.utils_fasta import get_seqs


def find_rev_comp(sequence, rev_comp_dictionary):
    # Save time by storing reverse complements in a hash.
    if sequence in rev_comp_dictionary:
        return rev_comp_dictionary[sequence]

    # Make a reversed version of the string.
    rev_sequence = list(sequence)
    rev_sequence.reverse()
    rev_sequence = ''.join(rev_sequence)

    return_value = ""
    for letter in rev_sequence:
        if letter == "A":
            return_value += "T"
        elif letter == "C":
            return_value += "G"
        elif letter == "G":
            return_value += "C"
        elif letter == "T":
            return_value += "A"
        elif letter == "N":
            return_value += "N"
        else:
            error_info = ("Unknown DNA character (%s)\n" % letter)
            sys.exit(error_info)

    # Store this value for future use.
    rev_comp_dictionary[sequence] = return_value

    return return_value


def _cmp(a, b):
    return (a > b) - (a < b)


def make_rev_comp_km_list(km_list):
    rev_comp_dictionary = {}
    new_km_list = [km for km in km_list if _cmp(km, find_rev_comp(km, rev_comp_dictionary)) <= 0]
    return new_km_list


def km_bow(input_file, k, alphabet, rev_comp=False):
    """Generate km vector."""

    if rev_comp and re.search(r'[^acgtACGT]', ''.join(alphabet)) is not None:
        sys.exit("Error, Only DNA sequence can be reverse compliment.")

    with open(input_file, 'r') as f:
        seq_list = get_seqs(f, alphabet)
    vector = []
    km_list = make_km_list(k, alphabet)

    for seq in seq_list:
        count_sum = 0

        # Generate the km frequency dict.
        km_count = {}
        for km in km_list:
            temp_count = frequency(seq, km)
            if not rev_comp:
                if km not in km_count:
                    km_count[km] = 0
                km_count[km] += temp_count
            else:
                rev_km = find_rev_comp(km, {})
                if km <= rev_km:
                    if km not in km_count:
                        km_count[km] = 0
                    km_count[km] += temp_count
                else:
                    if rev_km not in km_count:
                        km_count[rev_km] = 0
                    km_count[rev_km] += temp_count

            count_sum += temp_count

        # Normalize.
        if not rev_comp:
            count_vec = [km_count[km] for km in km_list]
        else:
            rev_comp_km_list = make_rev_comp_km_list(km_list)
            count_vec = [km_count[km] for km in rev_comp_km_list]
        count_vec = [round(float(e) / count_sum, 8) for e in count_vec]

        vector.append(count_vec)

    return np.array(vector)
