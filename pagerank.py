import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = dict()
    if corpus[page] == set():
        for filename in corpus:
            distribution[filename] = 1 / len(corpus)
    else:
        for filename in corpus:
            if filename in corpus[page]:
                distribution[filename] = damping_factor / len(corpus[page]) + (1 - damping_factor) / len(corpus)
            else:
                distribution[filename] = (1 - damping_factor) / len(corpus)

    return distribution



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    data = dict()
    for filename in corpus:
        data[filename] = 0

    pages = list(corpus.keys())
    page = pages[random.randrange(len(corpus))]

    for i in range(n):
        page = random.choices(pages, transition_model(corpus, page, damping_factor).values())[0]

        data[page] += 1

    for filename in data:
        data[filename] = data[filename] / n

    return data


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    data = dict()

    for filename in corpus:
        data[filename] = 1 / N

    change = 1
    while change > 0.0001 or change < -0.0001:
        filename = list(data.keys())[random.randrange(len(data))]
        previous_value = data[filename]
        for file in data:
            sum_i = 0
            for file_i in data:
                if file in corpus[file_i]:
                    sum_i += data[file_i] / len(corpus[file_i])
            data[file] = (1 - damping_factor) / N + damping_factor * sum_i

        change = data[filename] - previous_value

    return data


if __name__ == "__main__":
    main()
