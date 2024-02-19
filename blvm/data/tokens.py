import string


# Tokens
START_TOKEN = "<"
END_TOKEN = ">"
DELIMITER_TOKEN = "|"
BLANK_TOKEN = "%"
UNKNOWN_TOKEN = "¿"

SPACE = " "
APOSTROPHE = "'"
PERIOD = "."
COMMA = ","
EXCLAMATION = "!"
QUERY = "?"
HYPHEN = "-"
AMPERSAND = "&"


# Alphabets
ENGLISH_STANDARD = list(string.ascii_lowercase + SPACE + APOSTROPHE)
PENN_TREEBANK_ALPHABET = ENGLISH_STANDARD + [PERIOD, HYPHEN, AMPERSAND, "$", "N"]
TIMIT_ALPHABET = ENGLISH_STANDARD + [PERIOD, COMMA, HYPHEN, EXCLAMATION, QUERY]

# Phonesets
TIMIT_PHONESET = [
    "bcl", "dcl", "gcl", "pcl", "tck", "kcl", "tcl",
    "b", "d", "g", "p", "t", "k", "dx", "q", "jh", "ch", "s", "sh", "z", "zh", "f", "th", "v", "dh", "m", "n", "ng",
    "em", "en", "eng", "nx", "l", "r", "w", "y", "hh", "hv", "el", "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay",
    "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h", "1", "2"
]
