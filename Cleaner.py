import re
import numpy as np
from numpy.linalg import matrix_rank

text = """| 1 x4-8x2y2-12xy3-5y4+4x3z-20x2yz-64xy2z-40y3z-8x2z2-100xyz2-110y2z2-48xz3-120yz3-45z4 2x3y+7x2y2+8xy3+3y4+2x3z+22x2yz+44xy2z+24y3z+15x2z2+72xyz2+66y2z2+36xz3+72yz3+27z4 0                                                                                      x3y2-3xy4-2y5-3x2y2z-24xy3z-21y4z-x3z2-14x2yz2-70xy2z2-84y3z2-11x2z3-88xyz3-158y2z3-39xz4-138yz4-45z5 x2y3+2xy4+y5+3x2y2z+12xy3z+9y4z+3x2yz2+24xy2z2+30y3z2+x2z3+20xyz3+46y2z3+6xz4+33yz4+9z5 |
           | 1 -8x3z-40x2yz-64xy2z-32y3z-48x2z2-160xyz2-128y2z2-96xz3-160yz3-64z4                  4x3z+20x2yz+32xy2z+16y3z+24x2z2+80xyz2+64y2z2+48xz3+80yz3+32z4                     x4+6x3y+13x2y2+12xy3+4y4+6x3z+26x2yz+36xy2z+16y3z+13x2z2+36xyz2+24y2z2+12xz3+16yz3+4z4 -12x3yz-60x2y2z-96xy3z-48y4z+12x3z2-12x2yz2-144xy2z2-144y3z2+44x2z3-16xyz3-160y2z3+32xz4-80yz4-16z5   4x3yz+20x2y2z+32xy3z+16y4z-4x3z2+4x2yz2+48xy2z2+48y3z2-16x2z3+48y2z3-16xz4+16yz4        |
           | 1 0                                                                                   0                                                                                  0                                                                                      0                                                                                                     0                                                                                       |

"""

'''
@Global Parameters:

    maximum_order: the highest number of orders to describe the polynomial space we are working on
    MONOMIAL_SET_LENGTH: = 220

    mdic: the monomial dictionary
    pdic: the reverse of mdic
        @Function conditional_add: helper function for the mdic construction
    {'x': 1, 'y': 2, 'z': 3,
    'x2': 4, 'xy': 5, 'y2': 6, 'xz': 7, 'yz': 8, 'z2': 9,
    'x3': 10, 'x2y': 11, 'xy2': 12, 'y3': 13, 'x2z': 14, 'xyz': 15, 'y2z': 16, 'xz2': 17, 'yz2': 18, 'z3': 19,
    'x4': 20, 'x3y': 21, 'x2y2': 22, 'xy3': 23, 'y4': 24, 'x3z': 25, 'x2yz': 26, 'xy2z': 27, 'y3z': 28, 'x2z2': 29,
        'xyz2': 30, 'y2z2': 31, 'xz3': 32, 'yz3': 33, 'z4': 34,
    'x5': 35, 'x4y': 36, 'x3y2': 37, 'x2y3': 38, 'xy4': 39, 'y5': 40, 'x4z': 41, 'x3yz': 42, 'x2y2z': 43, 'xy3z': 44,
        'y4z': 45, 'x3z2': 46, 'x2yz2': 47, 'xy2z2': 48, 'y3z2': 49, 'x2z3': 50, 'xyz3': 51, 'y2z3': 52, 'xz4': 53,
        'yz4': 54, 'z5': 55,
    'x6': 56, 'x5y': 57, 'x4y2': 58, 'x3y3': 59, 'x2y4': 60, 'xy5': 61, 'y6': 62, 'x5z': 63, 'x4yz': 64, 'x3y2z': 65,
        'x2y3z': 66, 'xy4z': 67, 'y5z': 68, 'x4z2': 69, 'x3yz2': 70, 'x2y2z2': 71, 'xy3z2': 72, 'y4z2': 73, 'x3z3': 74,
        'x2yz3': 75, 'xy2z3': 76, 'y3z3': 77, 'x2z4': 78, 'xyz4': 79, 'y2z4': 80, 'xz5': 81, 'yz5': 82, 'z6': 83,
    'x7': 84, 'x6y': 85, 'x5y2': 86, 'x4y3': 87, 'x3y4': 88, 'x2y5': 89, 'xy6': 90, 'y7': 91, 'x6z': 92, 'x5yz': 93,
        'x4y2z': 94, 'x3y3z': 95, 'x2y4z': 96, 'xy5z': 97, 'y6z': 98, 'x5z2': 99, 'x4yz2': 100, 'x3y2z2': 101,
        'x2y3z2': 102, 'xy4z2': 103, 'y5z2': 104, 'x4z3': 105, 'x3yz3': 106, 'x2y2z3': 107, 'xy3z3': 108, 'y4z3': 109,
        'x3z4': 110, 'x2yz4': 111, 'xy2z4': 112, 'y3z4': 113, 'x2z5': 114, 'xyz5': 115, 'y2z5': 116, 'xz6': 117,
        'yz6': 118, 'z7': 119,
    'x8': 120, 'x7y': 121, 'x6y2': 122, 'x5y3': 123, 'x4y4': 124, 'x3y5': 125, 'x2y6': 126, 'xy7': 127, 'y8': 128,
        'x7z': 129, 'x6yz': 130, 'x5y2z': 131, 'x4y3z': 132, 'x3y4z': 133, 'x2y5z': 134, 'xy6z': 135, 'y7z': 136,
        'x6z2': 137, 'x5yz2': 138, 'x4y2z2': 139, 'x3y3z2': 140, 'x2y4z2': 141, 'xy5z2': 142, 'y6z2': 143, 'x5z3': 144,
        'x4yz3': 145, 'x3y2z3': 146, 'x2y3z3': 147, 'xy4z3': 148, 'y5z3': 149, 'x4z4': 150, 'x3yz4': 151, 'x2y2z4': 152,
        'xy3z4': 153, 'y4z4': 154, 'x3z5': 155, 'x2yz5': 156, 'xy2z5': 157, 'y3z5': 158, 'x2z6': 159, 'xyz6': 160,
        'y2z6': 161, 'xz7': 162, 'yz7': 163, 'z8': 164,
    'x9': 165, 'x8y': 166, 'x7y2': 167, 'x6y3': 168, 'x5y4': 169, 'x4y5': 170, 'x3y6': 171, 'x2y7': 172, 'xy8': 173,
        'y9': 174, 'x8z': 175, 'x7yz': 176, 'x6y2z': 177, 'x5y3z': 178, 'x4y4z': 179, 'x3y5z': 180, 'x2y6z': 181,
        'xy7z': 182, 'y8z': 183, 'x7z2': 184, 'x6yz2': 185, 'x5y2z2': 186, 'x4y3z2': 187, 'x3y4z2': 188, 'x2y5z2': 189,
        'xy6z2': 190, 'y7z2': 191, 'x6z3': 192, 'x5yz3': 193, 'x4y2z3': 194, 'x3y3z3': 195, 'x2y4z3': 196, 'xy5z3': 197,
        'y6z3': 198, 'x5z4': 199, 'x4yz4': 200, 'x3y2z4': 201, 'x2y3z4': 202, 'xy4z4': 203, 'y5z4': 204, 'x4z5': 205,
        'x3yz5': 206, 'x2y2z5': 207, 'xy3z5': 208, 'y4z5': 209, 'x3z6': 210, 'x2yz6': 211, 'xy2z6': 212, 'y3z6': 213,
        'x2z7': 214, 'xyz7': 215, 'y2z7': 216, 'xz8': 217, 'yz8': 218, 'z9': 219,
    '1': 0}
'''

maximum_order = 10
MONOMIAL_SET_LENGTH = 220


def condition_add(target, coordinate, power):
    if power != 0:
        target += coordinate
        if power != 1:
            target += str(power)
    return target


mdic = {}
pdic = {}
pos = 0
for num in range(1, 10):

    for bar2 in range(num, -1, -1):

        for bar1 in range(bar2, -1, -1):
            pos += 1
            mono = ""
            mono = condition_add(mono, "x", bar1)
            mono = condition_add(mono, "y", bar2 - bar1)
            mono = condition_add(mono, "z", num - bar2)
            mdic[mono] = pos
            pdic[pos] = mono

mdic["1"] = 0
pdic[0] = "1"


# ===================================FUNCTIONS==========================================================================


# parse_text: Parse the text into List-of-List form: DONE
#         @Param:
#             Input:
#             input_text: the text to be processed
#             Output:
#             rows: a List of a List of entries, in String form
#             (e.g. "1", "-8x3z-40x2yz-64xy2z-32y3z-48x2z2-160xyz2-128y2z2-96xz3-160yz3-64z4")

def parse_text(input_text):
    rows = []
    rows_index = [i for i in range(len(input_text)) if input_text.find('|', i) == i]  # The indices of '|' signs

    for x in zip(rows_index[::2], rows_index[1::2]):
        row = input_text[x[0] + 2:x[1]]
        row = row.split()
        rows.append(row)

    return rows


# get_column_vector: Get the respective column vectors from the parsed text
#         @Param:
#             Input:
#             input_matrix: the List-of-List form of the text
#             Output:
#             columns: a List of vectors, each of which consists of several entries (similar to above "entries")

def get_column_vector(input_matrix):
    columns = []

    example_row = input_matrix[0]
    columns_count = len(example_row)

    for column_index in range(columns_count):
        column_temp = []
        for row_index in range(len(input_matrix)):
            column_temp.append(input_matrix[row_index][column_index])
        columns.append(column_temp)

    return columns


# parse_vector: Given a particular vector, parse it into coefficients on different monomials
#         @Param:
#             Input:
#             input_vector: the vector input
#             Output:
#             parsed_vector: the vector with length (input_vector.length * monomial_space.length), and right
#                            coefficients

def parse_vector(input_vector):
    parsed_vector = []

    for entry in input_vector:
        parsed_entry = parse_entry(entry)
        parsed_vector.append(parsed_entry)

    return parsed_vector


# monomial_get_coefficient: Given a monomial in string form, get its coefficient so it could be put in the right
#     index.
#         @Param:
#             Input:
#             input_monomial: the monomial input
#             Output:
#             param_list: the list of returns
#             monomial: the monomial returned
#             coefficient: the coefficient that should be put in the right index.

def monomial_get_coefficient(input_monomial):

    param_list = []
    coefficient = re.split('[xyz]', input_monomial)[0]
    monomial = input_monomial[len(coefficient):]

    if coefficient == "" or coefficient == "+":
        coefficient = 1
    elif coefficient == "-":
        coefficient = -1
    else:
        coefficient = int(coefficient)
    if monomial == "":
        monomial = "1"

    param_list.append(coefficient)
    param_list.append(monomial)

    return param_list


# parse_entry: Given a particular entry, parse it into coefficients on different monomials
#         @Param:
#             Input:
#             input_entry: the entry input
#             Output:
#             parsed_entry: a list of numbers representing the projection on the monomial space length of this
#             particular entry.
#         @Using:
#             monomial_str_to_index(monomial in input_entry)

def parse_entry(input_entry):
    parsed_entry = [0] * MONOMIAL_SET_LENGTH
    raw = re.split('([-+])', input_entry)

    # This part re-connects the "+" / "-" signs with the monomials.
    mex = ""
    for mon in raw:
        if mon == "+" or mon == "-":
            mex += mon
        else:
            mex += mon
            if mex != "":
                param_list = monomial_get_coefficient(mex)
                mon_coefficient = param_list[0]
                mon_position = mdic[param_list[1]]
                parsed_entry[mon_position] = mon_coefficient
            mex = ""

    return parsed_entry


# parse_monomial: Given a monomial in string form, return its power for x, y, and z.

def parse_monomial(monomial):
    coeff_list = [0] * 3

    char_list = ["x", "y", "z"]
    pos_list = [0] * 3
    for i in range(3):
        pos_list[i] = monomial.find(char_list[i])
        if pos_list[i] == -1:
            coeff_list[i] = 0

    for i in range(2, -1, -1):
        if pos_list[i] != -1:
            residual = monomial[pos_list[i] + 1:]
            if residual == "":
                coeff_list[i] = 1
            else:
                coeff_list[i] = int(residual)
            monomial = monomial[:pos_list[i]]

    return coeff_list


# monomial_to_string: Give a list of coefficients, generate the monomial string.

def monomial_to_string(coeff_list):
    if coeff_list == [0, 0, 0]:
        return "1"

    monomial = ""
    char_set = ["x", "y", "z"]
    for i in range(3):
        if coeff_list[i] != 0:
            monomial += char_set[i]
            if coeff_list[i] != 1:
                monomial += str(coeff_list[i])

    return monomial


# generate_compensated_vector: Given a vector and a compensator monomial, return the multiplied vector, i.e. each
#     element is compensated by the compensator.
#     @Param:
#         Input:
#         input_vector: the input vector
#         compensator: the monomial that needs to be multiplied to each element
#         Output:
#         compensated_vector: the compensated vector in R^n form.

def generate_compensated_vector(input_vector, compensator):
    compensated_vector = [0] * MONOMIAL_SET_LENGTH
    epos = 0
    for element in input_vector:
        if element != 0:
            coeff_list = parse_monomial(pdic[epos])
            coeff_added_list = parse_monomial(compensator)
            for i in range(len(coeff_list)):
                coeff_list[i] += coeff_added_list[i]
            pos = mdic[monomial_to_string(coeff_list)]
            compensated_vector[pos] = element
        epos += 1

    return compensated_vector


# generate_monomial_space: Given a dimension, generate the respective monomial space.

def generate_monomial_space(dimension):
    mspace = ["1"]
    for num in range(1, dimension + 1):
        for bar2 in range(num, -1, -1):
            for bar1 in range(bar2, -1, -1):
                mono = ""
                mono = condition_add(mono, "x", bar1)
                mono = condition_add(mono, "y", bar2 - bar1)
                mono = condition_add(mono, "z", num - bar2)
                mspace.append(mono)
    return mspace


# generate_ideal: Given a parsed vector, generate its related ideal, given the order-difference.
#     @Param:
#         Input:
#         input_vector: the vector input from parse_vector.
#         order_difference: the number of orders that we have to compensate.
#         Output:
#         generated_ideal: the generated ideal.

def generate_ideal(input_vector, order_difference):
    generated_ideal = []
    compensation_space = generate_monomial_space(order_difference)
    for some_monomial in compensation_space:
        generated_ideal.append(generate_compensated_vector(input_vector, some_monomial))
    return generated_ideal


# generate_column_ideal: Generate the ideal of the entire column, and result in a list of n*220-length vectors.
#      Thoughts:
#      matrix = []
#      concatenated = []
#      for i in range(dimension of compensated space):
#         for j in ideallist:
#             concatenated.append(ideallist[j][i]'s elements NOT THE LIST ITSELF!)
#             # or use another iteration to do it.
#
#         We now get the 1st compensation of entries 1~n combined.
#         matrix.append(concatenated)
#         concatenated = []

def generate_column_ideal(column_vector, order_difference):
    ideal_list = []
    matrix = []
    for entry in column_vector:
        ideal_list.append(generate_ideal(parse_entry(entry), order_difference))
    concatenated = []
    for i in range(len(generate_monomial_space(order_difference))):
        for j in ideal_list:
            for bit in j[i]:
                concatenated.append(bit)
        matrix.append(concatenated)
        concatenated = []

    return matrix


# generate_matrix: Given the parsed vectors, output the big matrix.

def generate_matrix(parsed_columns, order_difference):
    matrix = []
    for col in parsed_columns:
        ideal_rows = generate_column_ideal(col, order_difference)
        for row in ideal_rows:
            matrix.append(row)

    return matrix


# check_matrix: Check if the matrix has property rank(A) = rank(A|b)

def check_matrix(matrix, new_vector):

    mA = np.array(matrix)
    mA = mA.T
    mb = np.array(new_vector)

    mA_b = np.c_[mA, mb]
    rank1 = matrix_rank(mA)
    rank2 = matrix_rank(mA_b)

    return rank1 != rank2


# find_highest_order: Find the highest order in a list.

def find_highest_order(vector):
    order = 0
    for element in vector:
        raw = re.split('([-+])', element)
        mex = ""
        for mon in raw:
            if mon == "+" or mon == "-":
                mex += mon
            else:
                mex += mon
                coeff_list = parse_monomial(mex)
                sum = coeff_list[0] + coeff_list[1] + coeff_list[2]
                if sum > order:
                    order = sum
                mex = ""

    return order


# cleaner: Get the columns of the input matrix, and then for each new column, check if it creates a new rank. If so,
#          take it into the set. If not, remove it.

def cleaner(text):

    column_set = []
    new_order = 0
    curr_order = 0
    columns = get_column_vector(parse_text(text))
    column_set.append(columns[0])

    for col in columns:

        new_order = find_highest_order(col)
        col_long = []
        for entry in col:
            monomial_list = parse_entry(entry)
            for monomial in monomial_list:
                col_long.append(monomial)

        if check_matrix(generate_matrix(column_set, new_order - curr_order), col_long):
            column_set.append(col)
        if curr_order < new_order:
            curr_order = new_order

    return column_set


print(cleaner(text))
