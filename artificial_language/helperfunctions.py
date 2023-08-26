import random
import re


def convert_dict_to_list_of_values(d):
    """
    """
    return [v for k, v in d.items()]


def prod(x):
    """
    Finds product of all elements in array
    """
    prod = 1
    for elem in x:
        prod = elem * prod

    return prod


def convert_df_cols_to_list(df, ):
    """
    """
    return


def _convert_df_to_dict_re(df, key_col, col_start, drop_cols):
    """
    """
    # Drop unneeded columns
    df = df.drop(columns = [col for col in drop_cols if col in df.columns], axis = 1)

    # Find columns matching col_start
    value_cols = [col for col in df.columns if re.match(col_start, col)]
    
    # Only keep key column or columns in value_cols
    df = df.drop(columns = [col for col in df.columns if col not in value_cols or col == key_col])

    # Convert to dictionary
    return df.set_index(key_col).T.to_dict('list')

def _convert_df_to_dict(df,
                        key_col = 'word',
                        drop_cols = ['pos'],
                        drop_rep = []):
    """
    Turns a dataframe into a dictionary, where each key : value pair is a row
    Keys are the values in key_col, and each value is a list of all other column values
    Values that are not wanted should be listed in drop_cols
    """
    # Remove unwanted columns
    df = df.drop(columns = [col for col in drop_cols if col in df.columns], axis = 1)

    # Convert to dictionary
    return df.set_index(key_col).T.to_dict('list')


def _convert_cols_to_dict(df, key_col, value_col):
    """
    """
    return dict(zip(df[key_col], df[value_col]))


def _add_elems_to_elem_num_dict(old_elem_to_index, new_elems):
    """
    """
    new_elem_to_index = old_elem_to_index
    new_index_to_elem = _reverse_dictionary(new_elem_to_index)

    old_indices_max = max([v for k, v in old_elem_to_index.items()])

    for index, elem in enumerate(new_elems):
        index += old_indices_max + 1
        new_elem_to_index[elem] = index
        new_index_to_elem[index] = elem

    return new_elem_to_index, new_index_to_elem


def _convert_list_to_elem_num_dict(l):
    """
    """
    return {elem : num for num, elem in enumerate(l)}

def _reverse_dictionary(d):
    """
    """
    return {v: k for k, v in d.items()}


def _choose_from_probability_dict(probability_dict):
    """
    """
    (options, option_probs) = zip(*probability_dict.items())
    choice = random.choices(population = options, weights = option_probs, k = 1)[0]

    return choice, probability_dict[choice]


def _convert_weights_to_probs(weights):
    """
    """
    try:
        probs = [(weight / sum(weights)) for weight in weights]
    except ZeroDivisionError:
        probs = [0.0 for weight in weights]

    return probs