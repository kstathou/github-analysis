def flatten_lists(lst):
    """Remove nested lists."""
    return [item for sublist in lst for item in sublist]

def change_cc(val):
    """Change the country code for Greece and United Kingdom to match the one from EIS."""
    if val == 'gb':
        return 'uk'
    elif val == 'gr':
        return 'el'
    else:
        return val