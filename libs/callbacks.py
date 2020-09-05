"""
callbacks module
"""

def load_cb(cval, fval, title):
    NUM_BARS = 30
    ratio = cval / fval
    ratio *= NUM_BARS
    ratio = int(ratio)
    print('\r|' + '=' * int(ratio) + '>' + ' ' * (NUM_BARS - ratio) +
          '| ' + title, end='')
    if cval == fval:
        print(' Done.')
    return
