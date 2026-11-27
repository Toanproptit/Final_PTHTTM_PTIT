from pathlib import Path
import json
from collections import Counter
import pdfplumber
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
QA=ROOT/'tmp/pdfs/toan_v2_review'
f=pdfplumber.open(ROOT/'output/pdf/Tieu_luan_Nguyen_Trong_Toan_B23DCCN833_v2.pdf')
colors=Counter();bad=[];count=0
for n,p in enumerate(f.pages,1):
    for c in p.chars:
        color=c.get('non_stroking_color');key=str(color);colors[key]+=1;count+=1
        if color is not None and color!=0 and color!=(0,) and color!=(0,0,0) and color!=(0,0,0,1):
            bad.append((n,c['text'],color))
    p.close()
assert not bad,str(bad[:12])
summary=json.loads((QA/'summary.json').read_text(encoding='utf-8'))
doc=pdfium.PdfDocument(str(ROOT/'output/pdf/Tieu_luan_Nguyen_Trong_Toan_B23DCCN833_v2.pdf'))
pages=summary['review_pages']
for i in range(0,len(pages),4):
    canvas=Image.new('RGB',(1200,1730),'white');draw=ImageDraw.Draw(canvas)
    for j,n in enumerate(pages[i:i+4]):
        im=Image.open(QA/f'page-{n}.png');im.thumbnail((590,835))
        x=(j%2)*600;y=(j//2)*865
        canvas.paste(im,(x,y+25));draw.text((x+5,y+5),f'Trang {n}',fill='black')
    canvas.save(QA/f'sheet-{i//4+1}.png')
result={'pages':len(f.pages),'text_characters':count,'non_black_text':len(bad),'colors':dict(colors),'review_sheets':(len(pages)+3)//4}
(QA/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
