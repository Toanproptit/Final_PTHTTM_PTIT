from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

files = sorted(Path('tmp/pdfs').glob('report-page-*.png'))
thumbs = []
for f in files:
    im = Image.open(f).convert('RGB')
    im.thumbnail((420, 594))
    canvas = Image.new('RGB', (440, 630), 'white')
    canvas.paste(im, ((440-im.width)//2, 12))
    ImageDraw.Draw(canvas).text((12, 608), f.stem, fill='black')
    thumbs.append(canvas)
cols = 3
rows = (len(thumbs)+cols-1)//cols
sheet = Image.new('RGB', (cols*440, rows*630), '#dddddd')
for i, im in enumerate(thumbs):
    sheet.paste(im, ((i%cols)*440, (i//cols)*630))
sheet.save('tmp/pdfs/report-montage.png')
print(f'Created montage for {len(files)} pages')
