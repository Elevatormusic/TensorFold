"""A lane finds a request by identity: ``list.remove`` and ``in`` on its lists must not compare the fields of two
streams. Image requests carry tensors, and a field comparison of tensors raises ("Boolean value of Tensor with more
than one value is ambiguous"). Two text requests with equal fields must stay two requests."""

from __future__ import annotations

from tensorfold.cuda.streams import Stream


class Ambiguous:
    """Compares like a tensor with more than one value: the result has no truth value."""

    def __eq__(self, other):
        return self

    def __bool__(self):
        raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")

    __hash__ = object.__hash__


def test_removing_an_image_stream_does_not_compare_its_tensors():
    first = Stream(prompt=[1, 2], count=4, vision=Ambiguous())
    second = Stream(prompt=[1, 2], count=4, vision=Ambiguous())
    filling = [first, second]
    filling.remove(second)
    assert filling == [first]
    assert second not in filling


def test_equal_text_streams_stay_separate_requests():
    first = Stream(prompt=[5, 6, 7], count=8, sid=0)
    second = Stream(prompt=[5, 6, 7], count=8, sid=0)
    filling = [first, second]
    filling.remove(second)
    assert len(filling) == 1 and filling[0] is first
    assert first != second
    assert len({first, second}) == 2
