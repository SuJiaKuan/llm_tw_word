import requests

from llm_tw_word.compute import memory


@memory.cache
def _request_fanhuaji(
    text,
    converter,
    url="https://api.zhconvert.org/convert",
):
    return requests.post(url, data={
        "text": text,
        "converter": converter,
    }).json()["data"]["text"]



def simp2data(text_simp):
    text_trad = _request_fanhuaji(text_simp, "Traditional")
    text_tw = _request_fanhuaji(text_trad, "Taiwan")

    return (text_trad, text_tw)
