"""Versioned Vietnamese hotel abbreviations; no labels or test-specific rules."""
import hashlib
import json
import re
import unicodedata

LEXICON = {
 'ko':'không','k':'không','khong':'không','khg':'không','khôg':'không',
 'hok':'không','hông':'không','kg':'không',
 'ks':'khách sạn','ksan':'khách sạn','ksạn':'khách sạn','khachsan':'khách sạn',
 'nv':'nhân viên','nviên':'nhân viên','nhanvien':'nhân viên',
 'p':'phòng','phg':'phòng','bf':'bữa sáng','bfast':'bữa sáng',
 'bth':'bình thường','bt':'bình thường','đc':'được','dc':'được',
 'wc':'phòng vệ sinh','wi-fi':'wifi','wi fi':'wifi'
}
NORMALIZER_ID = 'hotel_abbreviations_v1'
SIGNATURE = hashlib.sha256(json.dumps(LEXICON,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
PATTERN = re.compile(r'(?<!\w)('+'|'.join(re.escape(k) for k in sorted(LEXICON,key=len,reverse=True))+r')(?!\w)',re.IGNORECASE)


def normalize(text):
    text=unicodedata.normalize('NFC',str(text))
    return PATTERN.sub(lambda m:LEXICON[m.group(0).lower()],text)


def prepare_texts(cfg,texts):
    kind=cfg.get('text_normalizer')
    if kind in (None,'none'):return texts
    if kind!=NORMALIZER_ID:raise ValueError('Unknown text normalizer: '+str(kind))
    assert cfg.get('text_normalizer_sha256')==SIGNATURE,'Normalizer/model version mismatch'
    return [normalize(s) for s in texts]
