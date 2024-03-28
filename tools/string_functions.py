import pandas as pd
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import csv
import random
import sys
import gc
import textdistance
import re
import copy
from termcolor import colored

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


def flag_input_stream(input_stream):
    count = 0
    stream_flags = list("·" * len(input_stream))
    stream_counts = list("0" * len(input_stream))
    # Backward-passing the Input Stream
    for i in range(len(input_stream) - 1, -1, -1):
        # Take note of how many deletions there are in a row
        if input_stream[i] == "<":
            stream_counts[i] = str(count)
            count += 1
        else:
            # These characters will appear so they get flagged
            if count == 0:
                stream_flags[i] = str("F")
                stream_counts[i] = str(count)
            else:
                # Do not flag these characters as they'll be deleted
                stream_counts[i] = str(count)
                count -= 1
    # Return the stream flags, the counts and the original Input Stream
    return "".join(stream_flags), "".join(stream_counts), input_stream


# Compute the minimum string distance between two strings
def min_string_distance(str1, str2):
    m, n = len(str1), len(str2)

    # Initialize the matrix with zeros
    distance_matrix = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and column
    for i in range(m + 1):
        distance_matrix[i][0] = i
    for j in range(n + 1):
        distance_matrix[0][j] = j

    # Fill in the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            distance_matrix[i][j] = min(
                distance_matrix[i - 1][j] + 1,  # Deletion
                distance_matrix[i][j - 1] + 1,  # Insertion
                distance_matrix[i - 1][j - 1] + cost  # Substitution
            )

    # Return the value of the Minimum String Distance and the dynamic programming matrix
    return distance_matrix[m][n], distance_matrix


# Compute Optimal Alignments of two Strings
def align(S1, S2, D, x, y, A1, A2, alignments):
    # If both indexes reach zero then the alignment is complete and added to the list
    if x == 0 and y == 0:
        alignments.append([A1, A2])
        return
    # Use the MSD matrix to create the optimal alignments
    if x > 0 and y > 0:
        if D[x][y] == D[x - 1][y - 1] and S1[x - 1] == S2[y - 1]:
            align(S1, S2, D, x - 1, y - 1, S1[x - 1] + A1, S2[y - 1] + A2, alignments)
        if D[x][y] == D[x - 1][y - 1] + 1:
            align(S1, S2, D, x - 1, y - 1, S1[x - 1] + A1, S2[y - 1] + A2, alignments)
    # Insert insertions and omission markers as needed
    if x > 0 and D[x][y] == D[x - 1][y] + 1:
        align(S1, S2, D, x - 1, y, S1[x - 1] + A1, "~" + A2, alignments)
    if y > 0 and D[x][y] == D[x][y - 1] + 1:
        align(S1, S2, D, x, y - 1, "~" + A1, S2[y - 1] + A2, alignments)


# Adapt the alignments to the Input Stream
def stream_align(IS, alignments_list):
    triplets = []
    for alignment_pair in alignments_list:
        new_stream = IS[2]  # Input stream
        new_flags = IS[0]  # Flags
        new_alignments = [alignment_pair[0], alignment_pair[1]]

        # The final length of the alignment triplet will have to include omissions and insertions.
        final_alignment_length = max(len(alignment_pair[1]), len(new_stream))

        for i in range(final_alignment_length):
            if i < len(alignment_pair[1]) and alignment_pair[1][i] == '~':
                # Insert spacer character
                new_stream = new_stream[:i] + "·" + new_stream[i:]
                # Necessary to set every "~" as flag.
                new_flags = new_flags[:i] + "F" + new_flags[i:]
            elif new_flags[i] != "F":
                alignment_pair[0] = alignment_pair[0][:i] + "·" + alignment_pair[0][i:]
                alignment_pair[1] = alignment_pair[1][:i] + "·" + alignment_pair[1][i:]

        triplets += [[alignment_pair[0], alignment_pair[1], [new_stream, new_flags]]]
    # Return the new alignments as a tuple and the flags
    return triplets


# Error Estimation function
def error_estimate(target_phrase, user_phrase, input_stream):
    error_count = 0

    error_count += textdistance.hamming.distance(target_phrase, user_phrase)

    for b in range(len(input_stream)):  # 0 to |IS|-1
        # generic assumed 1 corrected error at the end of a series of "<"
        if b != 0 and input_stream[b] == "<" and input_stream[b - 1] != "<":
            error_count += 1

    return error_count


# Assign position values to properly conduct error detection
def assign_position_values(triplets):
    edited_triplets = []

    for triplet in triplets:
        # Get the two strings and the input stream alignments
        align_0 = triplet[0]
        align_1 = triplet[1]
        align_IS = triplet[2][0]
        # Plus the flags
        IS_flags = triplet[2][1]
        # And initialize the pos values as zero
        IS_pos = list("0" * len(align_IS))

        pos = 0
        for i in range(len(align_IS)):
            if IS_flags[i] == "F":
                pos = 0
                IS_pos[i] = str(pos)
            else:
                if align_IS[i] == "<" and pos > 0:
                    pos -= 1
                IS_pos[i] = str(pos)
                if align_IS[i] != "<":
                    pos += 1

        new_triplet = [[align_0, align_1, [align_IS, IS_flags, "".join(IS_pos)]]]
        edited_triplets += new_triplet

    # Return a copy of the given triplets now with flags and position values.
    return edited_triplets


# Look Ahead function for Error Detection
def look_ahead(string, start, count, condition_function):
    i = start
    while (i >= 0 and i < len(string)) and not condition_function(string[i]):
        i += 1  # Keep looking until the condition is met.
    while count > 0 and i < len(string):
        i += 1
        if i == len(string):
            break
        elif condition_function(string[i]):
            count -= 1
    return min(i, len(string) - 1)


# Lood Behind function for Error Detection
def look_behind(string, start, count, condition_function):
    i = start
    while (i >= 0 and i < len(string)) and not condition_function(string[i]):
        i -= 1  # Keep looking until the condition is met.
    while count > 0 and i >= 0:
        i -= 1
        if i < 0:
            break
        elif condition_function(string[i]):
            count -= 1
    return max(0, i)


# Character detection functions
def check_special_char(char):
    return char == '~' or char == '#' or char == '·'


def check_not_spacer(char):
    return char != "·"


# Error Detection Function
def error_detection(triplets):
    errors_per_triplet = []

    for triplet in triplets:
        errors = []

        P = triplet[0]  # Target Phrase Align
        T = triplet[1]  # User Input Align
        IS = triplet[2][0]  # Input Stream
        IS_flags = triplet[2][1]  # IS Flags
        IS_pos = triplet[2][2]  # IS Position values

        a = 0
        for b in range(len(IS)):  # 0 to |IS|-1
            if T[b] == '~':
                errors += [[0, "o", [P[b], "~"]]]  # uncorrected omission
            elif IS_flags[b] == "F" or b == len(IS) - 1:
                M = set()  # Corrected omissions set
                I = set()  # Corrected insertions set
                for i in range(a, b):  # Iterate over substring determined by flags
                    val = int(IS_pos[i])
                    if IS[i] == "<":
                        if val in M:
                            M.remove(val)
                        if val in I:
                            I.remove(val)
                    elif check_not_spacer(IS[i]):
                        target = look_ahead(P, b, val + len(M) - len(I), check_special_char)
                        next_p = look_ahead(P, target, 1, check_special_char)
                        prev_p = look_behind(P, target, 1, check_special_char)
                        next_is = look_ahead(IS, i, 1, check_not_spacer)
                        prev_is = look_behind(IS, i, 1, check_not_spacer)

                        if IS[i] == P[target]:
                            errors += [[1, "n", [IS[i], IS[i]]]]  # corrected no error
                        elif target >= len(P) - 1 or IS[next_is] == P[target] or (
                                IS[prev_is] == IS[i] and IS[prev_is] == P[i]):
                            errors += [[1, "i", ["~", IS[i]]]]  # corrected insertion
                            I.add(val)
                        elif IS[i] == P[next_p] and not check_special_char(T[target]):
                            errors += [[1, "o", [P[target], "~"]]]  # corrected omission
                            errors += [[1, "n", [IS[i], IS[i]]]]  # corrected no error
                            M.add(val)
                        else:
                            errors += [[1, "s", [P[target], IS[i]]]]  # corrected substitution

                if P[b] == "~":
                    errors += [[0, "i", ["~", T[b]]]]  # uncorrected insertion
                elif P[b] != T[b]:
                    errors += [[0, "s", [P[b], T[b]]]]  # uncorrected substitution
                elif P[b] != "·":
                    errors += [[0, "n", [T[b], T[b]]]]  # uncorreced no error
                a = b + 1
        if len(IS) < len(P):
            for i in range(len(IS), len(P)):
                errors += [[0, "o", [P[i], "~"]]]
        errors_per_triplet += [errors]
    # Returns a series of the errors/non-errors present in the user typed phrase.
    # The series contains items of this type [0, i, [~, z]] where:
    # [1]corrected, [0]uncorrected
    # [i]nsertion, [o]mission, [t]ransposition, [s]ubstitution and [c]apitalization. [n]o error
    # [char expected, char produced]
    return errors_per_triplet


# Specify which kind of substitution error is happening (TRA, CAP or SU)
def specify_errors(error_list):
    updated_errors = []
    last_error = None
    last_last_error = None
    last_last_last_error = None
    for error in error_list:
        if ((error[2][0] != error[2][1]) and  # CAPITALIZATION ERROR
                (error[2][0].lower() == error[2][1].lower()) and
                error[1] == "s"):
            new_error = [error[0], "c", error[2]]
            updated_errors += [new_error]
            last_error = new_error
        else:
            if ((last_error != None) and  # UNCORRECTED TRANSPOSITION IF NOT RE-ALIGNED
                    (error[2][0] == last_error[2][1] and error[2][1] == last_error[2][0]) and
                    (error[0] == last_error[0]) and
                    (error[1] == last_error[1] == "s")):
                updated_errors.pop()
                new_error = [error[0], "t", [error[2][0], error[2][1]]]
                updated_errors += [new_error]
                last_error = new_error
            elif ((
                          last_last_error != None) and  # UNCORRECTED TRANSPOSITION NOT RE-ALIGNED CASE 1 if input is [~,l2][l1,l1][l2,~]
                  (error[2][0] == last_last_error[2][1] != '~') and
                  (last_error[2][0] == last_error[2][1]) and
                  (error[2][1] == last_last_error[2][0] == '~') and
                  (error[1] == "o" and last_error[1] == "n" and last_last_error[1] == "i")):
                updated_errors.pop()
                updated_errors.pop()
                new_error = [error[0], "t", [error[2][0], last_error[2][0]]]
                updated_errors += [new_error]
                last_error = new_error
            elif ((
                          last_last_error != None) and  # UNCORRECTED TRANSPOSITION NOT RE-ALIGNED CASE 2 if input is [l2,~][l1,l1][~,l2]
                  (error[2][0] == last_last_error[2][1] == '~') and
                  (last_error[2][0] == last_error[2][1]) and
                  (error[2][1] == last_last_error[2][0] != '~') and
                  (error[1] == "i" and last_error[1] == "n" and last_last_error[1] == "o")):
                updated_errors.pop()
                updated_errors.pop()
                new_error = [error[0], "t", [error[2][1], last_error[2][0]]]
                updated_errors += [new_error]
                last_error = new_error
            elif ((
                          last_last_last_error != None) and  # CORRECTED TRANSPOSITION (DELETE LAST 4, TWO WRONG INSERTS AND TWO CORRECTIONS -> TRANSPOSITION) if input is [~,l2][~,l1][l2,l2][l1,l1]
                  (last_last_last_error[0] == last_last_error[0] == 1) and
                  (last_last_last_error[1] and last_last_error[1] in ["i", "s"]) and
                  (last_last_last_error[2][1] == error[2][1]) and
                  (last_last_error[2][1] == last_error[2][1])):
                updated_errors.pop()
                updated_errors.pop()
                updated_errors.pop()
                new_error = [1, "t", [error[2][1], last_error[2][1]]]
                updated_errors += [new_error]
                last_error = new_error
            else:  # NO EDIT TO BE MADE
                updated_errors += [error]
                last_error = error
        last_last_last_error = last_last_error
        last_last_error = last_error
    return updated_errors


def visualize_error(error):
    if error[0] == 0:
        if error[1] == "i":
            print("Uncorrected | Type: " + colored("INS-Error", "on_light_green") + " | Letters: {}".format(error[2]))
        elif error[1] == "o":
            print("Uncorrected | Type: " + colored("OMI-Error", "on_light_magenta") + " | Letters: {}".format(error[2]))
        elif error[1] == "s":
            print("Uncorrected | Type: " + colored("SUB-Error", "on_light_cyan") + " | Letters: {}".format(error[2]))
        elif error[1] == "c":
            print("Uncorrected | Type: " + colored("CAP-Error", "on_light_blue") + " | Letters: {}".format(error[2]))
        elif error[1] == "t":
            print("Uncorrected | Type: " + colored("TRA-Error", "on_light_white") + " | Letters: {}".format(error[2]))
        else:
            print("Uncorrected | Type: NON-Error | Letters: {}".format(error[2]))
    else:
        if error[1] == "i":
            print("Corrected   | Type: " + colored("INS-Error", "on_green") + " | Letters: {}".format(error[2]))
        elif error[1] == "o":
            print("Corrected   | Type: " + colored("OMI-Error", "on_magenta") + " | Letters: {}".format(error[2]))
        elif error[1] == "s":
            print("Corrected   | Type: " + colored("SUB-Error", "on_cyan") + " | Letters: {}".format(error[2]))
        elif error[1] == "c":
            print("Corrected   | Type: " + colored("CAP-Error", "on_blue") + " | Letters: {}".format(error[2]))
        elif error[1] == "t":
            print("Corrected   | Type: " + colored("TRA-Error", "on_white") + " | Letters: {}".format(error[2]))
        else:
            print("Corrected   | Type: NON-Error | Letters: {}".format(error[2]))


def count_errors(error_list):
    uncorr_errors = [["i", 0], ["o", 0], ["s", 0], ["t", 0], ["c", 0], ]  # "INS,OMI,SUB,TRA,CAP"
    corr_errors = [["i", 0], ["o", 0], ["s", 0], ["t", 0], ["c", 0], ]
    errors_only_list = []

    for error in error_list:
        if error[0]:
            # Char has been corrected
            for error_type in corr_errors:
                # Increase by one the counter for this kind of error
                if error_type[0] == error[1]:
                    error_type[1] += 1
            if error[1] != 'n':
                errors_only_list.append(error)
        else:
            # Char has not been corrected
            for error_type in uncorr_errors:
                # Increase by one the counter for this kind of error
                if error_type[0] == error[1]:
                    error_type[1] += 1
            if error[1] != 'n':
                errors_only_list.append(error)

    return uncorr_errors, corr_errors, errors_only_list


def count_transpositions(unique_transposition_sets, new_transposition):
    # For each transposition
    # [[exp_1, exp_2], [transp_1, transp_2], count]
    for transposition in unique_transposition_sets:
        # if the error combination is already known
        if new_transposition == transposition[1]:
            # increase the count
            transposition[2] += 1
            break
        # if the combination isn't present then add it with count set to 1
    else:
        unique_transposition_sets.append([[new_transposition[1], new_transposition[0]], new_transposition, 1])


def optimal_error_set(all_error_lists, unique_transposition_sets):
    # Create a list of already known error combinations
    # The objects of the set will have this structure
    # [[uncorr_errors, corr_errors],
    #  [sum_of_errors],
    #  appeareances]
    unique_error_sets = []

    for error_list in all_error_lists:
        # Specify the type of errors
        new_error_list = specify_errors(error_list)
        # Count the total errors per category
        uncorr_errors, corr_errors, errors_only_list = count_errors(new_error_list)
        # Sum uncorrected and corrected errors together so we can filter by highest TRA, lowest INS
        sum_of_errors = []
        for pair1, pair2 in zip(uncorr_errors, corr_errors):
            letter = pair1[0]
            sum_value = pair1[1] + pair2[1]
            sum_of_errors.append([letter, sum_value])

        new_errors = [uncorr_errors, corr_errors]

        for error_set in unique_error_sets:
            # if the error combination is already known
            if new_errors == error_set[0]:
                # increase the count
                error_set[2] += 1
                break
            # if the combination isn't present then add it with count set to 1
        else:
            unique_error_sets.append([new_errors, sum_of_errors, 1, errors_only_list])

        # Sort by max TRA errors and then by minimum sum of all other errors on the sum_of_errors
        best_set = max(unique_error_sets,
                       key=lambda x: (x[1][3][1], - (x[1][0][1] + x[1][1][1] + x[1][2][1] + x[1][4][1])))

        for error in best_set[3]:
            if error[1] == 't':
                count_transpositions(unique_transposition_sets, [error[2][0], error[2][1]])

    return best_set[0], best_set[2]  # THE ERROR WITH HIGHEST COUNT


def estimate_phrase(target_phrase, user_phrase, input_stream):
    min_estimated_errors_for_phrase = error_estimate(target_phrase, user_phrase, input_stream)
    return min_estimated_errors_for_phrase


def return_errors(target_phrase, user_phrase, input_stream, unique_transposition_sets):
    # Get alignments
    flagged_IS = flag_input_stream(input_stream)
    _, MSD = min_string_distance(target_phrase, user_phrase)
    alignments = []
    align(target_phrase, user_phrase, MSD, len(target_phrase), len(user_phrase), "", "", alignments)
    all_triplets = stream_align(flagged_IS, alignments)
    all_edited_triplets = assign_position_values(all_triplets)

    # Get error lists
    all_error_lists = error_detection(all_edited_triplets)

    best_set, occurrences = optimal_error_set(all_error_lists, unique_transposition_sets)

    return best_set[0], best_set[1]


def test_phrase(target_phrase, user_phrase, input_stream, verbose=True):
    flagged_IS = flag_input_stream(input_stream)
    print("Phrase Details:")
    print("Flags: {}\nMoves: {}\nIS   : {}\n".format(*flagged_IS))
    unique_transposition_sets = []
    _, MSD = min_string_distance(target_phrase, user_phrase)

    alignments = []

    align(target_phrase, user_phrase, MSD, len(target_phrase), len(user_phrase), "", "", alignments)

    all_triplets = stream_align(flagged_IS, alignments)
    all_edited_triplets = assign_position_values(all_triplets)
    all_error_lists = error_detection(all_edited_triplets)

    best_set, occurrences = optimal_error_set(all_error_lists, unique_transposition_sets)
    INF, IF, C, F = count_component(all_error_lists[-1])
    print("INF: {} | IF: {} | C: {} | F: {}".format(INF, IF, C, F))
    count = 0
    new_all_error_lists = []
    for error_list in all_error_lists:
        print("Alignment {}:".format(str(count + 1)))
        print("Target: {}\nUser  : {}\n".format(alignments[count][0], alignments[count][1]))
        new_error_list = specify_errors(error_list)
        count += 1

        if verbose:
            print("Unmodified Errors")
            for error in error_list:
                visualize_error(error)
            print("Modified Errors")
            for error in new_error_list:
                visualize_error(error)
            print("\n")
        new_all_error_lists.append(new_error_list)
    uncorr_errors, corr_errors = optimal_error_set(new_all_error_lists, unique_transposition_sets)

    # print("Uncorrected: INS: {} | OMI: {} | SUB: {} | TRA: {} | CAP: {} | NON: {}".format(
    #     *(err for _, err in uncorr_errors)))
    # print("Corrected  : INS: {} | OMI: {} | SUB: {} | TRA: {} | CAP: {} | NON: {}".format(
    #     *(err for _, err in corr_errors)))
    # print("---------------------------------------------------------------------------")
    # print()
    #
    # print("The most common set with highest TRA error prioritization is:")
    # print("Uncorrected: INS: {} | OMI: {} | SUB: {} | TRA: {} | CAP: {} | NON: {}".format(
    #     *(err for _, err in best_set[0])))
    # print("Corrected  : INS: {} | OMI: {} | SUB: {} | TRA: {} | CAP: {} | NON: {}".format(
    #     *(err for _, err in best_set[1])))
    # print("And it occurrs in {} out of {} alignments".format(occurrences, len(all_error_lists)))


def get_input_stream(test_section_df):
    input_stream_string = ''
    last_length = None
    prev_row = None

    for index, row in test_section_df.iterrows():
        # If a deletion operation happened then add a "#" symbol
        if last_length is not None and len(str(row['INPUT'])) == last_length - 1:
            input_stream_string += '<'
        # Sometimes there are double entries, no idea why, so we ignore them and skip over
        elif prev_row is not None and (str(row['DATA']) == prev_row[0]
                                       and str(row['INPUT']) == prev_row[1]):
            pass
        # If all's good then add the newest character to the input stream
        else:
            input_stream_string += str(row['INPUT'])[-1]
        # Update last length and previoous row
        last_length = len(str(row['INPUT']))
        prev_row = [str(row['DATA']), str(row['INPUT'])]

        auto_corrected_if_count, auto_corrected_c_count, \
        auto_corrected_word_count, auto_correct_count = 0, 0, 0, 0

    return input_stream_string, auto_corrected_if_count, auto_corrected_c_count, auto_corrected_word_count, auto_correct_count


def count_component(error_list, verbose=False):
    INF, IF, C, F = 0, 0, 0, 0
    for error in error_list:
        if error[0] == 0:
            if error[1] == "i":
                INF += 1
                if verbose:
                    print(
                        "Uncorrected | Type: " + colored("INS-Error", "on_light_green") + " | Letters: {}".format(
                            error[2]))
            elif error[1] == "o":
                INF += 1
                if verbose:
                    print("Uncorrected | Type: " + colored("OMI-Error", "on_light_magenta") + " | Letters: {}".format(
                        error[2]))
            elif error[1] == "s":
                INF += 1
                if verbose:
                    print(
                        "Uncorrected | Type: " + colored("SUB-Error", "on_light_cyan") + " | Letters: {}".format(
                            error[2]))
            elif error[1] == "c":
                INF += 1
                if verbose:
                    print(
                        "Uncorrected | Type: " + colored("CAP-Error", "on_light_blue") + " | Letters: {}".format(
                            error[2]))
            elif error[1] == "t":
                INF += 1
                if verbose:
                    print(
                        "Uncorrected | Type: " + colored("TRA-Error", "on_light_white") + " | Letters: {}".format(
                            error[2]))
            else:
                if verbose:
                    print("Uncorrected | Type: NON-Error | Letters: {}".format(error[2]))
                C += 1
        else:
            if error[1] == "i":
                IF += 1
                if verbose:
                    print("Corrected   | Type: " + colored("INS-Error", "on_green") + " | Letters: {}".format(error[2]))
            elif error[1] == "o":
                IF += 1
                if verbose:
                    print(
                        "Corrected   | Type: " + colored("OMI-Error", "on_magenta") + " | Letters: {}".format(error[2]))
            elif error[1] == "s":
                IF += 1
                if verbose:
                    print("Corrected   | Type: " + colored("SUB-Error", "on_cyan") + " | Letters: {}".format(error[2]))
            elif error[1] == "c":
                IF += 1
                if verbose:
                    print("Corrected   | Type: " + colored("CAP-Error", "on_blue") + " | Letters: {}".format(error[2]))
            elif error[1] == "t":
                IF += 1
                if verbose:
                    print("Corrected   | Type: " + colored("TRA-Error", "on_white") + " | Letters: {}".format(error[2]))
            else:
                if verbose:
                    print("Corrected   | Type: NON-Error | Letters: {}".format(error[2]))
                # C += 1
    return INF, IF, C, F


if __name__ == "__main__":
    pass
    # reference_sentence = "the quick brown"
    # user_input = "th quick brpown"
    # input_stream = "th quxck<<<ick brpown"
    # user_input = 'Was wondering if you and Natalie connected?'
    # input_stream = 'Was c<wimedrting <<<<<<<<<<ondering if you and Natalie conce<<nected ?<<v<?'
    # reference_sentence = "Was wondering if you and Natalie connected?"
    # test_phrase(reference_sentence, user_input, input_stream)
    reference_sentence = 'lähetä paperit minulle'
    user_input = 'lähetä paperit minulm'
    input_stream = 'lähet<tä paperit minulm'
    test_phrase(reference_sentence, user_input, input_stream)
