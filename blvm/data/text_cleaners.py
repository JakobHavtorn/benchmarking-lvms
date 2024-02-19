
def clean_timit(txt: str):
    """Clean timit text for ASR. Can be used to clean both .txt, .wrd and .phn files."""
    # Split multiple lines (remove last empty line)
    txt = txt.split("\n")
    if not txt[-1]:
        txt = txt[:-1]

    # Remove alignment annotation "'0 46797 She had your dark suit in greasy wash water all year."
    txt = [" ".join(t.split()[2:]) for t in txt]
    txt = " ".join(txt)

    # Lower case
    txt = txt.lower()

    # Replace rare punctuation
    txt = txt.replace(";", ",")
    txt = txt.replace(":", ".")

    # Replace quotation marks
    txt = txt.replace('"', " ")

    return txt
