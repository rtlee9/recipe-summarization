import pandas as pd
import prep_data

MEASURE_TOKENS = [
    'cup', 'can', 'teaspoon', 'tsp', 'tablespoon', 'tbsp', 'pound', 'lb', 'jar', 'bottle',
    'ounce', 'small', 'large', 'medium', 'envelope', 'ear', 'piece', 'drops', 'oz']

def remove_trailing_s(token):
    if token.endswith('s'):
        return token[:-1]
    else:
        return token

def get_max_token_index(tokens, MEASURE_TOKENS):
    max_token_index = -1
    for i, token in enumerate(tokens):
        if remove_trailing_s(token) in MEASURE_TOKENS:
            max_token_index = i
    return max_token_index

def parse_item_and_prep(item_and_prep):
    components = [i.strip() for i in item_and_prep.split(',')]
    return components[0], ' '.join(components[1:])

def parse_item(item, skip_str_list):
    item_clean = item
    for skip_str in skip_str_list:
        item_clean = item_clean.split(skip_str)[0]
    return item_clean

def is_number(s):
    try:
        float(s.replace('/', ''))
        return True
    except ValueError:
        return False

def move_to_quantity(first_component):
    return (is_number(first_component)) | (remove_trailing_s(
        first_component) in ['to', 'plus', 'a', 'several', 'or'] + MEASURE_TOKENS)

def parse_quantity(ingredient, quantity=''):
    components = swap_places_colon(remove_parens(
        ingredient.lower().replace('-', ' '))).split(':')[-1].split()
    if len(components) <= 1:
        return ingredient, quantity
    first_component = components[0]
    if move_to_quantity(first_component):
        quantity += ' ' + first_component
        ingredient = ' '.join(components[1:])
        ingredient, quantity = parse_quantity(ingredient, quantity)
    else:
        return ingredient, quantity
    return ingredient, quantity.strip()

def remove_parens(ingredient):
    split1 = ingredient.split('(')
    if len(split1) == 1:
        return split1[0]
    split2 = ' '.join(split1[1:]).split(') ')
    split2.append('')  # append extra item to list in case parens comes last
    return split1[0] + ' '.join(split2[1:])

def swap_places_colon(ingredient):
    components = ingredient.split(':')
    if len(components) <= 1:
        return ingredient
    else:
        return '{} {}'.format(
            swap_places_colon(':'.join(components[1:])), components[0])

def parse_ingredients(ingredient):
    try:
        item_and_prep, measures = parse_quantity(ingredient.lower())
    except IndexError:
        print('Could not parse "{}"'.format(ingredient))
    item, prep = parse_item_and_prep(item_and_prep)
    item = parse_item(item, [' or ', ' plus ', ' and ', ' as ', ' from '])
    return measures, item, prep

def get_df(data):
    # Collect all ingredients in a single list
    ingredients_all = []
    for ingredients_list in (data.train.ingredients,
                             data.validation.ingredients, data.test.ingredients):
        for ingredient in ingredients_list:
            ingredients_all.extend(ingredient)
    print('{:,} ingredients in total'.format(len(ingredients_all)))

    ingredients_parsed = pd.DataFrame(
        [parse_ingredients(ingredient) for ingredient in ingredients_all])
    ingredients_parsed.columns = ['measure', 'ingredient', 'preparation']
    return ingredients_parsed

def parse_ingredient_list(ingredient_list):
    return [parse_ingredients(i)[1] for i in ingredient_list]

def parse_ingredient_listlist(ingredient_listlist):
    return [parse_ingredient_list(i) for i in ingredient_listlist]

if __name__ == '__main__':
    data = prep_data.main()
    ingredients_parsed = get_df(data)
    print(ingredients_parsed.sample(10))
    print(parse_ingredient_listlist(data.train.ingredients)[:10])
