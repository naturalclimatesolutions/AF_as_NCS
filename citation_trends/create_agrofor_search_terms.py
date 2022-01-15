import sys
import os

topic = sys.argv[1]

assert topic in ['ncs', 'carbon'], 'VALID TOPICS: "ncs", "carbon"'


def read_search_terms_file(topic):
    """
    Reads in a search-terms file.

    Params
    ------
    topic : str
        can be 'agrofor', 'ncs', or 'carbon'

    Returns
    -------
    txt : str
        the file's search-terms contents, as a block of text

    """
    with open('search_terms_%s.txt' % topic, 'r') as f:
        txt = f.read()
    return txt


if __name__ == '__main__':
    # read in the search-term blocks
    agrofor_terms = read_search_terms_file('agrofor')
    topic_terms = read_search_terms_file(topic)

    # peg them together
    search_terms = 'TS=(\n\n%s\nAND\n\n%s\n)' % (agrofor_terms, topic_terms)

    # save to file
    with open('search_terms_%s_%s.txt' % ('agrofor', topic), 'w') as f:
        f.write(search_terms)
