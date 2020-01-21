import requests


def get_text_from_wikipedia(title, wiki_domain='en.wikipedia.org'):
    """
    Retrieve Wikipedia article content as plain-text

    :param title: Title of Wikipedia article
    :param wiki_domain: API domain (e.g., en.wikipedia.org)
    :return: Article content as plain-text
    """

    res = requests.get(f'https://{wiki_domain}/w/api.php', params={
        "action":"query",
        "prop":"revisions",
        "rvprop":"content",
        "format":"json",
        "titles":title,
        "rvslots":"main"
    })

    from gensim.corpora.wikicorpus import filter_wiki
    pages = [p for p_id, p in res.json()['query']['pages'].items()]

    if len(pages) == 0:
        raise ValueError(f'Cannot find Wikipedia article: {title}')
    elif len(pages) > 1:
        raise ValueError(f'Wikipedia article title is unambigious. Multiple articles returned from API: {title}')
    else:
        p = pages[0]

    wikitext = p['revisions'][0]['slots']['main']['*']  #
    text = filter_wiki(wikitext).strip()

    return text
