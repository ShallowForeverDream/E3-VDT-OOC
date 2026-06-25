from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Iterable, List, Dict, Any
from tqdm import tqdm

TIME_RE = re.compile(r"(\b(?:19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})\b|\b\d{1,2}-\d{1,2}-(?:\d{2}|\d{4})\b|\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b|\d{4}年|\d{1,2}月\d{1,2}日|周[一二三四五六日天]|星期[一二三四五六日天])", re.I)
KNOWN_LOCATIONS = {"paris","london","new york","washington","washington dc","beijing","shanghai","tokyo","moscow","ukraine","russia","gaza","israel","france","germany","china","india","usa","united states","u.s.","britain","england","europe","africa","asia","iraq","iran","syria","afghanistan","pakistan","japan","korea","hong kong","taiwan"}
CN_LOCATION_RE = re.compile(r"([\u4e00-\u9fa5]{2,8}(?:国|省|市|县|区|州|岛|镇|村|港|机场|广场|法院|学校|医院))")
EVENT_KEYWORDS = {
    "protest": ["protest","demonstration","rally","march","抗议","游行","示威"],
    "war/conflict": ["war","attack","strike","missile","soldier","conflict","战争","袭击","导弹","冲突","blast","explosion"],
    "disaster": ["earthquake","flood","fire","storm","rescue","灾害","地震","洪水","火灾","救援"],
    "politics": ["election","president","minister","government","vote","选举","总统","政府","部长","parliament"],
    "court/crime": ["court","trial","arrest","police","crime","法院","审判","逮捕","警方"],
    "sports": ["match","game","cup","team","player","比赛","球队","冠军"],
    "health": ["covid","hospital","vaccine","health","疫情","医院","疫苗"],
    "finance": ["market","stock","bank","economy","finance","市场","股票","银行","经济"],
}
RELATION_KEYWORDS = {
    "gather": ["gather","rally","march","protest","聚集","游行","抗议"],
    "attack": ["attack","strike","bomb","hit","袭击","打击","轰炸"],
    "meet": ["meet","talk","summit","visit","会见","访问","峰会"],
    "arrest": ["arrest","detain","charge","逮捕","拘留","起诉"],
    "rescue": ["rescue","evacuate","救援","撤离"],
    "win": ["win","beat","defeat","获胜","击败"],
}
STOP_ENTITIES = {"The","A","An","This","That","People","Image","Photo","News","Reuters","Getty"}


def norm_item(x: str) -> str:
    return re.sub(r"\s+", " ", x.strip().strip(".,;:!?，。；：！？'\"()[]{}"))


def unique(items: Iterable[str]) -> List[str]:
    seen=set(); out=[]
    for x in items:
        x=norm_item(str(x))
        if not x: continue
        k=x.lower()
        if k not in seen:
            seen.add(k); out.append(x)
    return out


def rule_extract(text: str) -> Dict[str, Any]:
    text = text or ""; lowered=text.lower()
    times = unique(m.group(0) for m in TIME_RE.finditer(text))
    locs = []
    for loc in KNOWN_LOCATIONS:
        if re.search(r"\b" + re.escape(loc) + r"\b", lowered):
            locs.append(loc)
    locs.extend(m.group(1) for m in CN_LOCATION_RE.finditer(text))
    locs = unique(locs)
    event_types=[]
    for et,kws in EVENT_KEYWORDS.items():
        if any(kw.lower() in lowered for kw in kws): event_types.append(et)
    relations=[]
    for rel,kws in RELATION_KEYWORDS.items():
        if any(kw.lower() in lowered for kw in kws): relations.append(rel)
    spans=re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b", text)
    spans=[s for s in spans if s.split()[0] not in STOP_ENTITIES]
    spans += re.findall(r"([\u4e00-\u9fa5]{2,12}(?:公司|政府|警方|法院|总统|部长|队|组织|集团))", text)
    loc_set={x.lower() for x in locs}; time_set={x.lower() for x in times}
    entities=unique(s for s in spans if s.lower() not in loc_set and s.lower() not in time_set)
    return {"entities": entities, "locations": locs, "times": times, "event_types": unique(event_types), "relations": unique(relations)}


def spacy_extract(text: str, model: str) -> Dict[str, Any]:
    try:
        import spacy
        nlp = spacy.load(model)
    except Exception as e:
        raise RuntimeError(f"spaCy model unavailable: {model}. Install with: python -m spacy download {model}. Error={e}")
    doc=nlp(text or "")
    entities=[]; locs=[]; times=[]
    for ent in doc.ents:
        if ent.label_ in {"PERSON","ORG","NORP"}: entities.append(ent.text)
        elif ent.label_ in {"GPE","LOC","FAC"}: locs.append(ent.text)
        elif ent.label_ in {"DATE","TIME"}: times.append(ent.text)
    base=rule_extract(text)
    return {
        "entities": unique(entities + base["entities"]),
        "locations": unique(locs + base["locations"]),
        "times": unique(times + base["times"]),
        "event_types": base["event_types"],
        "relations": base["relations"],
    }


def main():
    ap=argparse.ArgumentParser(description="Extract event tuples from COVE-lite context pairs.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--extractor", choices=["rule","spacy"], default="rule")
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    args=ap.parse_args()
    n=0; stats={"records":0,"extractor":args.extractor,"empty_current":0,"empty_true":0}
    out_path=Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, encoding="utf-8") as f, open(out_path,"w",encoding="utf-8") as out:
        for line in tqdm(f, desc="extract events"):
            if not line.strip(): continue
            rec=json.loads(line)
            cur=rec.get("current_caption","") or ""
            tru=rec.get("true_image_context","") or ""
            if args.extractor=="spacy":
                cur_evt=spacy_extract(cur, args.spacy_model)
                tru_evt=spacy_extract(tru, args.spacy_model)
            else:
                cur_evt=rule_extract(cur); tru_evt=rule_extract(tru)
            if not any(cur_evt.values()): stats["empty_current"] += 1
            if not any(tru_evt.values()): stats["empty_true"] += 1
            row=dict(rec); row["current_event_tuple"]=cur_evt; row["true_event_tuple"]=tru_evt
            out.write(json.dumps(row, ensure_ascii=False)+"\n")
            stats["records"] += 1
    Path(str(out_path)+".stats.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats,indent=2,ensure_ascii=False))

if __name__ == "__main__":
    main()
