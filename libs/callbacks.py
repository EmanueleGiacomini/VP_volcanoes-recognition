"""
callbacks module
"""

def load_cb(cval, fval, title):
    if cval == fval:
        print(' Done.')
        return
    NUM_BARS = 30
    ratio = cval / fval
    ratio *= NUM_BARS
    ratio = int(ratio)
    print('\r|' + '=' * int(ratio) + '>' + ' ' * (NUM_BARS - ratio) +
          '| ' + title, end='')
