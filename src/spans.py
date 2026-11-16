"""IOB2 span utilities shared by EDA, evaluation and the database layer."""


def get_spans(tags):
    """Convert an IOB2 tag sequence into (type, start, end_exclusive) spans.

    Follows conlleval/seqeval leniency: an I- tag that does not continue an
    entity of the same type starts a new entity.
    """
    spans = []
    cur_type, start = None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag == "O" or tag.startswith("B-") or (tag.startswith("I-") and tag[2:] != cur_type):
            if cur_type is not None:
                spans.append((cur_type, start, i))
                cur_type, start = None, None
            if tag != "O":
                cur_type, start = tag[2:], i
    return spans


def span_text(tokens, start, end):
    return " ".join(tokens[start:end])
