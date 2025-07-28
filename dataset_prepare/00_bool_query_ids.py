
from utils.pubmed_submission import pubmed_submission

#

def get_pubmed_ids(query, pubmed_ids_file):

    with open(pubmed_ids_file, "w") as f:
        # first run the query
        # then save the ids to a file
        ids, counter_too_many = pubmed_submission(query, None, 0)
        for id in ids:
            f.write(id + "\n")


if __name__ == "__main__":
    query = '"pubmed pmc open access"[filter] AND ("systematic review"[pt])'
    pubmed_ids_file = "../data/pubmed_ids.txt"

    get_pubmed_ids(query, pubmed_ids_file)

