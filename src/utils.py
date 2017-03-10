def join_ingredients(ingredients_listlist):
    return [' , '.join(i) for i in ingredients_listlist]

def get_flat_ingredients_list(ingredients_joined_train):
    return ' , '.join(ingredients_joined_train).split(' , ')

def section_print():
    '''Memorized function keeping track of section number'''
    section_number = 0

    def __inner(message):
        nonlocal section_number
        section_number += 1
        print('Section {}: {}'.format(section_number, message))
    print('Section {}: initializing section function'.format(section_number))
    return __inner

def is_filename_char(x):
    if x.isalnum():
        return True
    if x in ['-', '_']:
        return True
    return False

def URL_to_filename(filename):
    return "".join(x for x in filename if is_filename_char(x))

if __name__ == '__main__':
    print(URL_to_filename('http://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename'))
