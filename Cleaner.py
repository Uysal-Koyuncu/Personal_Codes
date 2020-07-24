import re

text = """| 1 x4-8x2y2-12xy3-5y4+4x3z-20x2yz-64xy2z-40y3z-8x2z2-100xyz2-110y2z2-48xz3-120yz3-45z4 2x3y+7x2y2+8xy3+3y4+2x3z+22x2yz+44xy2z+24y3z+15x2z2+72xyz2+66y2z2+36xz3+72yz3+27z4 0                                                                                      x3y2-3xy4-2y5-3x2y2z-24xy3z-21y4z-x3z2-14x2yz2-70xy2z2-84y3z2-11x2z3-88xyz3-158y2z3-39xz4-138yz4-45z5 x2y3+2xy4+y5+3x2y2z+12xy3z+9y4z+3x2yz2+24xy2z2+30y3z2+x2z3+20xyz3+46y2z3+6xz4+33yz4+9z5 |
           | 1 -8x3z-40x2yz-64xy2z-32y3z-48x2z2-160xyz2-128y2z2-96xz3-160yz3-64z4                  4x3z+20x2yz+32xy2z+16y3z+24x2z2+80xyz2+64y2z2+48xz3+80yz3+32z4                     x4+6x3y+13x2y2+12xy3+4y4+6x3z+26x2yz+36xy2z+16y3z+13x2z2+36xyz2+24y2z2+12xz3+16yz3+4z4 -12x3yz-60x2y2z-96xy3z-48y4z+12x3z2-12x2yz2-144xy2z2-144y3z2+44x2z3-16xyz3-160y2z3+32xz4-80yz4-16z5   4x3yz+20x2y2z+32xy3z+16y4z-4x3z2+4x2yz2+48xy2z2+48y3z2-16x2z3+48y2z3-16xz4+16yz4        |
           | 1 0                                                                                   0                                                                                  0                                                                                      0                                                                                                     0                                                                                       |

"""

'''
@Global Parameters:

    maximum_order: the highest number of orders to describe the polynomial space we are working on
'''

maximum_order = 10

# parse_text: Parse the text into List-of-List form: DONE
#         @Param:
#             Input:
#             input_text: the text to be processed
#             Output:
#             rows: a List of a List of entries, in String form
#             (e.g. "1", "-8x3z-40x2yz-64xy2z-32y3z-48x2z2-160xyz2-128y2z2-96xz3-160yz3-64z4")

def parse_text(input_text):

    rows = []
    rows_index = [i for i in range(len(text)) if text.find('|', i) == i]    # The indices of '|' signs

    for x in zip(rows_index[::2], rows_index[1::2]):
        row = text[x[0] + 2:x[1]]
        row = row.split()
        rows.append(row)

    return rows
#print(parse_text(text))


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
#print(get_column_vector(parse_text(text)))


# parse_vector: Given a particular vector, parse it into coefficients on different monomials
#         @Param:
#             Input:
#             input_vector: the vector input
#             Output:
#             parsed_vector: the vector with length (input_vector.length * monomial_space.length), and right coefficients

def parse_vector(input_vector):

    parsed_vector = []

    for entry in input_vector:
        parsed_entry = parse_entry(entry)
        parsed_vector.append(parsed_entry)

    return parsed_vector


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

    parsed_entry = []
    raw = re.split('([-+])', input_entry)

    # This part re-connects the "+" / "-" signs with the monomials.
    mex = ""
    for mon in raw:
        if mon == "+" or mon == "-":
            mex += mon
        else:
            mex += mon
            parsed_entry.append(mex)
            mex = ""
        mon_coefficient = parse_monomial(mon)

    return parsed_entry
print(parse_vector(get_column_vector(parse_text(text))[1]))

'''
@Global Parameters:
    
    maximum_order: the highest number of orders to describe the polynomial space we are working on
'''
'''
@Functions

    parse_text: Parse the text into List-of-List form: DONE
        @Param:
            Input:
            input_text: the text to be processed
            Output:
            rows: a List of a List of entries, in String form (e.g. "1", "-8x3z-40x2yz-64xy2z-32y3z-48x2z2-160xyz2-128y2z2-96xz3-160yz3-64z4")
         
    get_column_vector: Get the respective column vectors from the parsed text
        @Param:
            Input:
            input_matrix: the List-of-List form of the text
            Output:
            columns: a List of vectors, each of which consists of several entries (similar to above "entries")
    
    parse_vector: Given a particular vector, parse it into coefficients on different monomials
        @Param:
            Input:
            input_vector: the vector input
            Output:
            parsed_vector: the vector with length (input_vector.length * monomial_space.length), and right coefficients
            
    parse_entry: Given a particular entry, parse it into coefficients on different monomials
        @Param:
            Input:
            input_entry: the entry input
            Output:
            parsed_entry: a list of numbers representing the projection on the monomial space length of this particular 
            entry.
        @Using:
            monomial_str_to_index(monomial in input_entry)
    
    monomial_str_to_index: Given a monomial in string form, parse it to the correct index in the dictionary. Done by the
    mdic[] dictionary.
        @Param:
            Input:
            input_monomial: the monomial input
            Output:
            monomial_index: the index in the mdic[] dictionary. Used to sort the monomials into the correct parts.
            
    monomial_get_coefficient: Given a monomial in string form, get its coefficient so it could be put in the right
    index.
        @Param:
            Input:
            input_monomial: the monomial input
            Output:
            monomial_coefficient: the coefficient that should be put in the right index.
            
    parse_monomial: Given a monomial in string form, return its power for x, y, and z.
    
    monomial_multiplication: Given two monomials, return their product in string form.
    
    monomial_to_string: Given a monomial's respective powers, generate its string form so that the compensated monomial
    could be looked up in the dictionary and now we know where to put the coefficients.
    
    generate_ideal: Given a parsed vector, generate its related ideal, given the order-difference.
        @Param:
            Input:
            input_vector: the vector input from parse_vector.
            order_difference: the number of orders that we have to compensate.
            Output:
            generated_ideal:
        for some_monomial in compensation_space:
            generated_ideal.append(generate_compensated_vector(input_vector, some_monomial))
        
    generate_matrix: Given the parsed vectors, output the big matrix.
        matrix = []
        for vec in parsed_vectors:
            ideal_rows = generate_ideal(vec, order_difference)
            for row in ideal_rows:
                matrix.append(row)
    
    check_zero_row: Given the matrix, remove all rows that are entirely zero.
    
    check_bad_form: Check if 0x=1 form exist in any rows, which could cause no solution.
    
    solvable_matrix: See if the matrix have any non-trivial solutions. Boolean.
    
    cleaner: Given a set of columns, start from the first 2, and step by step, find if a new one is linearly independent
    from the first several. Remove all columns that are not.

'''
