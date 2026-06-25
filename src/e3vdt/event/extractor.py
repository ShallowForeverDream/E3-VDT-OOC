from __future__ import annotations
import re
from typing import Iterable, List
from e3vdt.schemas import EventTuple

TIME_RE = re.compile(r"(\b(?:19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})\b|\b\d{1,2}-\d{1,2}-(?:\d{2}|\d{4})\b|\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b|\d{4}年|\d{1,2}月\d{1,2}日|周[一二三四五六日天]|星期[一二三四五六日天])", re.I)
CN_LOCATION_RE = re.compile(r"([\u4e00-\u9fa5]{2,8}(?:国|省|市|县|区|州|岛|镇|村|港|机场|广场|法院|学校|医院))")
KNOWN_LOCATIONS = {"paris","london","new york","washington","washington dc","beijing","shanghai","tokyo","moscow","ukraine","russia","gaza","israel","france","germany","china","india","usa","united states","u.s.","britain","england","europe","africa","asia"}
EVENT_KEYWORDS = {
    "protest": ["protest","demonstration","rally","march","抗议","游行","示威"],
    "war/conflict": ["war","attack","strike","missile","soldier","conflict","战争","袭击","导弹","冲突"],
    "disaster": ["earthquake","flood","fire","storm","rescue","灾害","地震","洪水","火灾","救援"],
    "politics": ["election","president","minister","government","vote","选举","总统","政府","部长"],
    "court/crime": ["court","trial","arrest","police","crime","法院","审判","逮捕","警方"],
    "sports": ["match","game","cup","team","player","比赛","球队","冠军"],
    "health": ["covid","hospital","vaccine","health","疫情","医院","疫苗"],
    "finance": ["market","stock","bank","economy","finance","市场","股票","银行","经济"],
}
RELATION_KEYWORDS = {
    "gather": ["gather","rally","march","protest","聚集","游行","抗议"],
    "attack": ["attack","strike","bomb","hit","袭击","打击","轰炸"],
    "meet": ["meet","met","talk","summit","visit","会见","访问","峰会"],
    "arrest": ["arrest","detain","charge","逮捕","拘留","起诉"],
    "rescue": ["rescue","evacuate","救援","撤离"],
    "win": ["win","beat","defeat","获胜","击败"],
}
STOP_ENTITIES = {"The", "A", "An", "This", "That", "People", "Image", "Photo", "News"}

def _norm_item(x: str) -> str:
    return re.sub(r"\s+", " ", x.strip().strip(".,;:!?，。；：！？'\"()[]{}"))
def _unique(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for item in items:
        item = _norm_item(str(item))
        if not item: continue
        key = item.lower()
        if key not in seen:
            seen.add(key); out.append(item)
    return out

def extract_times(text: str) -> List[str]:
    return _unique(m.group(0) for m in TIME_RE.finditer(text or ""))
def extract_locations(text: str) -> List[str]:
    text = text or ""; lowered = text.lower(); found = []
    for loc in KNOWN_LOCATIONS:
        if re.search(r"\b" + re.escape(loc) + r"\b", lowered): found.append(loc)
    found.extend(m.group(1) for m in CN_LOCATION_RE.finditer(text))
    return _unique(found)
def extract_event_types(text: str) -> List[str]:
    lowered=(text or "").lower(); found=[]
    for event_type,kws in EVENT_KEYWORDS.items():
        if any(kw.lower() in lowered for kw in kws): found.append(event_type)
    return _unique(found)
def extract_relations(text: str) -> List[str]:
    lowered=(text or "").lower(); found=[]
    for rel,kws in RELATION_KEYWORDS.items():
        if any(kw.lower() in lowered for kw in kws): found.append(rel)
    return _unique(found)
def extract_entities(text: str) -> List[str]:
    text=text or ""
    spans=re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b", text)
    spans=[s for s in spans if s.split()[0] not in STOP_ENTITIES]
    spans += re.findall(r"([\u4e00-\u9fa5]{2,12}(?:公司|政府|警方|法院|总统|部长|队|组织|集团))", text)
    locs={x.lower() for x in extract_locations(text)}; times={x.lower() for x in extract_times(text)}
    return _unique(s for s in spans if s.lower() not in locs and s.lower() not in times)
def extract_event_tuple(text: str, source: str="text") -> EventTuple:
    return EventTuple(source=source, raw_text=text or "", entities=extract_entities(text), locations=extract_locations(text), times=extract_times(text), event_types=extract_event_types(text), relations=extract_relations(text))
