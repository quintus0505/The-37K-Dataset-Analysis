def levenshtein_with_details_and_indices(s1, s2):
    """
    Compute the Levenshtein distance between two strings (s1 and s2),
    and return the distance along with counts and indices of insertions, deletions, and substitutions.
    However, in this version, the logic that previously applied to s1 and s2 is switched.
    """
    rows = len(s2) + 1  # Switched
    cols = len(s1) + 1  # Switched
    distance_matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    # Initialize the distance matrix
    for i in range(1, rows):
        distance_matrix[i][0] = i
    for i in range(1, cols):
        distance_matrix[0][i] = i

    # Compute Levenshtein distance
    for col in range(1, cols):
        for row in range(1, rows):
            cost = 0 if s2[row - 1] == s1[col - 1] else 1  # Switched
            distance_matrix[row][col] = min(distance_matrix[row - 1][col] + 1,  # Deletion
                                            distance_matrix[row][col - 1] + 1,  # Insertion
                                            distance_matrix[row - 1][col - 1] + cost)  # Substitution

    # Backtrack to find the number of insertions, deletions, and substitutions
    insertions, deletions, substitutions = 0, 0, 0
    insertion_indices, deletion_indices, substitution_indices = [], [], []
    row, col = rows - 1, cols - 1
    while row > 0 or col > 0:
        if row > 0 and col > 0 and s2[row - 1] == s1[col - 1]:  # Switched
            row -= 1
            col -= 1
        elif row > 0 and col > 0 and distance_matrix[row][col] == distance_matrix[row - 1][col - 1] + 1:
            substitutions += 1
            substitution_indices.append(row - 1)  # Changed to index in s2
            row -= 1
            col -= 1
        elif row > 0 and distance_matrix[row][col] == distance_matrix[row - 1][col] + 1:
            deletions += 1
            deletion_indices.append(row - 1)  # Changed to index in s2
            row -= 1
        elif col > 0 and distance_matrix[row][col] == distance_matrix[row][col - 1] + 1:
            insertions += 1
            insertion_indices.append(row - 1)  # Changed, indicates where in s2 the insertion in s1 should be
            col -= 1
        # Handle cases where we're at the first row or column
        elif row == 0 and col > 0:
            insertions += 1
            insertion_indices.append(row - 1)  # Changed, for clarity in context
            col -= 1
        elif col == 0 and row > 0:
            deletions += 1
            deletion_indices.append(row - 1)  # Changed, for clarity in context
            row -= 1

    distance = distance_matrix[-1][-1]
    return (distance, insertions, deletion_indices, insertion_indices, substitutions, substitution_indices)


def rebuild_committed_sentence(typed):
    committed = []
    for char in typed:
        if char != "<":
            committed.append(char)
        elif committed:
            committed.pop()  # Remove the last character due to backspace
    return ''.join(committed)


def calculate_C_INF(reference, committed):
    # Calculate INF using Levenshtein distance
    INF, insertions, deletion_indices, insertion_indices, substitutions, substitution_indices = levenshtein_with_details_and_indices(
        reference, committed)
    # Directly count matching characters in the same positions
    C = len(committed) - insertions

    return C, INF, deletion_indices, insertion_indices, substitution_indices


def compute_IF_from_indices(reference, typed, deletion_indices, substitution_indices):
    IF = 0
    typed_index = 0
    reference_index = 0
    for char in typed:
        if char == "<":
            # Backspace character, check if it's necessary
            if typed_index - 1 in deletion_indices or typed_index - 1 in substitution_indices:
                # The backspace was necessary to correct a mistake
                pass
            else:
                # The backspace was unnecessary, counting towards IF
                IF += 1
            # Adjust indices based on the mistake type and position
            if typed_index - 1 in deletion_indices:
                deletion_indices.remove(typed_index - 1)
            if typed_index - 1 in substitution_indices:
                substitution_indices.remove(typed_index - 1)
        else:
            typed_index += 1

    return IF


def simplify_typed_text(typed_text):
    """
    Simplify the typed text by applying backspaces ("<") and ignoring characters
    that are backspaced and then retyped at the same position.
    """
    delete_chars = []
    simplify_text = []
    bsp_count = 0
    for i in range(len(typed_text)):
        if typed_text[i] == "<":
            delete_chars.append(typed_text[i - 2 * bsp_count - 1])
            bsp_count += 1
        else:
            if bsp_count == 0:
                simplify_text.append(typed_text[i])
            else:
                # turn the order of  the delete_chars
                delete_chars = delete_chars[::-1]
                # remove the last bsp_count elements from the simplify_text
                simplify_text = simplify_text[:-bsp_count]
                correct_char = list(typed_text[i: i + bsp_count])
                correct_char_used = [False] * len(correct_char)
                for j in range(len(correct_char)):
                    flag = False
                    for k in range(j, len(correct_char)):
                        if correct_char[k] == delete_chars[j] and not correct_char_used[k]:
                            flag = True
                            correct_char_used[k] = True
                            break
                    if not flag:
                        simplify_text.append(delete_chars[j])
                        simplify_text.append('<')
                        simplify_text.append(correct_char[j])
                bsp_count = 0
                delete_chars.clear()
    return ''.join(simplify_text)


def track_typing_errors(reference, typed):
    committed = rebuild_committed_sentence(typed)
    C, INF, deletion_indices, insertion_indices, substitution_indices = calculate_C_INF(reference, committed)
    # add 1 to every element to insertion_indices
    insertion_indices = [i + 1 for i in insertion_indices]
    # print("deletion_indices", deletion_indices)
    # print("substitution_indices", substitution_indices)
    # print("insertion_indices", insertion_indices)
    F = typed.count("<")
    simply_typed = simplify_typed_text(typed)
    # print("simply_typed", simply_typed)
    IF = compute_IF_from_indices(reference, simply_typed, deletion_indices, substitution_indices)
    return C, INF, IF, F


# Example usage
reference_sentence = "the quick brown"
typed_text = "th quxck<<<ick brpown"
# print(simplify_typed_text(typed_text))
# print(track_typing_errors(reference_sentence, typed_text))
