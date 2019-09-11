from stop_words import get_stop_words

stop_words = set(get_stop_words('en'))


def convert_text_to_set_without_stop(text):
    _set = set()
    for token in text.split(' '):
        if token and token.lower() not in stop_words:
            _set.add(token)
    return _set