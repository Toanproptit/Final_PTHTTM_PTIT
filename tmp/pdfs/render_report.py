from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, PageBreak, Paragraph, Preformatted,
    Spacer, Table, TableStyle, Flowable, KeepTogether,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "report" / "FINAL_REPORT.md"
OUTPUT = ROOT / "output" / "pdf" / "Diabetes_Agent_Lab_Report.pdf"
FONT_DIR = Path(r"C:\Windows\Fonts")

pdfmetrics.registerFont(TTFont("Arial", str(FONT_DIR / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONT_DIR / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Italic", str(FONT_DIR / "ariali.ttf")))
pdfmetrics.registerFontFamily(
    "Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic",
    boldItalic="Arial-Bold",
)

NAVY = colors.HexColor("#18324B")
BLUE = colors.HexColor("#2B668F")
PALE = colors.HexColor("#EAF2F7")
INK = colors.HexColor("#202B35")
MUTED = colors.HexColor("#526271")


def inline_markup(text: str) -> str:
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
    links: list[tuple[str, str]] = []

    def save_link(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2)
        if target.startswith(("http://", "https://")):
            replacement = f'<link href="{html.escape(target, quote=True)}" color="#2B668F">{html.escape(label)}</link>'
        else:
            replacement = html.escape(label)
        token = f"@@LINK{len(links)}@@"
        links.append((token, replacement))
        return token

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", save_link, text)
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="Arial-Bold" color="#18324B">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    for token, replacement in links:
        text = text.replace(token, replacement)
    return text


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="CoverTitle", fontName="Arial-Bold", fontSize=22, leading=29,
    alignment=TA_CENTER, textColor=NAVY, spaceAfter=15 * mm,
))
styles.add(ParagraphStyle(
    name="CoverSubtitle", fontName="Arial", fontSize=15, leading=22,
    alignment=TA_CENTER, textColor=BLUE, spaceAfter=7 * mm,
))
styles.add(ParagraphStyle(
    name="CoverMeta", fontName="Arial", fontSize=11, leading=18,
    alignment=TA_CENTER, textColor=MUTED,
))
styles.add(ParagraphStyle(
    name="H1Custom", fontName="Arial-Bold", fontSize=17, leading=22,
    textColor=NAVY, spaceBefore=7 * mm, spaceAfter=4 * mm, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="H2Custom", fontName="Arial-Bold", fontSize=13, leading=18,
    textColor=BLUE, spaceBefore=5 * mm, spaceAfter=2.5 * mm, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="H3Custom", fontName="Arial-Bold", fontSize=11, leading=15,
    textColor=INK, spaceBefore=3.5 * mm, spaceAfter=2 * mm, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="BodyCustom", fontName="Arial", fontSize=9.2, leading=14,
    textColor=INK, alignment=TA_JUSTIFY, spaceAfter=2.5 * mm,
    splitLongWords=True,
))
styles.add(ParagraphStyle(
    name="BulletCustom", fontName="Arial", fontSize=9.2, leading=13.5,
    textColor=INK, leftIndent=5 * mm, firstLineIndent=-4 * mm,
    spaceAfter=1.2 * mm,
))
styles.add(ParagraphStyle(
    name="QuoteCustom", fontName="Arial-Italic", fontSize=9, leading=13.5,
    textColor=MUTED, leftIndent=5 * mm, rightIndent=4 * mm,
    borderColor=BLUE, borderWidth=1, borderPadding=5, spaceBefore=2 * mm,
    spaceAfter=3 * mm,
))
styles.add(ParagraphStyle(
    name="SmallCustom", fontName="Arial", fontSize=8, leading=10,
    textColor=INK,
))


class ArchitectureDiagram(Flowable):
    def __init__(self, width: float):
        super().__init__()
        self.width = width
        self.height = 48 * mm

    def draw(self) -> None:
        c = self.canv
        labels = [
            ("Web UI", "React / Vite"), ("FastAPI", "API + static UI"),
            ("LangGraph", "orchestration"), ("Agents", "validate / safety / predict"),
            ("Knowledge", "TF-IDF fallback"), ("Storage", "MySQL / SQLite"),
        ]
        gap = 3 * mm
        box_w = (self.width - 5 * gap) / 6
        box_h = 18 * mm
        y = 22 * mm
        for i, (title, subtitle) in enumerate(labels):
            x = i * (box_w + gap)
            c.setFillColor(PALE if i not in (2, 3) else colors.HexColor("#DCEAF4"))
            c.setStrokeColor(colors.HexColor("#A9BECD"))
            c.roundRect(x, y, box_w, box_h, 2 * mm, fill=1, stroke=1)
            c.setFillColor(NAVY)
            c.setFont("Arial-Bold", 7.5)
            c.drawCentredString(x + box_w / 2, y + 11 * mm, title)
            c.setFillColor(MUTED)
            c.setFont("Arial", 5.6)
            c.drawCentredString(x + box_w / 2, y + 6 * mm, subtitle)
            if i < len(labels) - 1:
                c.setStrokeColor(BLUE)
                c.setLineWidth(1)
                c.line(x + box_w, y + box_h / 2, x + box_w + gap - 1.2 * mm, y + box_h / 2)
                c.line(x + box_w + gap - 2 * mm, y + box_h / 2, x + box_w + gap - 3.2 * mm, y + box_h / 2 + 1 * mm)
                c.line(x + box_w + gap - 2 * mm, y + box_h / 2, x + box_w + gap - 3.2 * mm, y + box_h / 2 - 1 * mm)
        c.setFont("Arial", 7.5)
        c.setFillColor(MUTED)
        c.drawString(0, 12 * mm, "Neo4j/vector retrieval khi cấu hình; OpenRouter chỉ cho tab LLM tùy chọn.")
        c.drawString(0, 7 * mm, "SQLite là fallback local/test; MySQL là system-of-record trong cấu hình Docker.")


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(filename, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
                         topMargin=20 * mm, bottomMargin=20 * mm, title="Diabetes Agent Lab - Bao cao do an",
                         author="Diabetes Agent Lab")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=self._decorate)])

    def _decorate(self, canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        w, h = A4
        canvas.setStrokeColor(colors.HexColor("#D9E3EA"))
        canvas.line(22 * mm, h - 14 * mm, w - 22 * mm, h - 14 * mm)
        canvas.setFont("Arial", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(22 * mm, h - 11 * mm, "DIABETES AGENT LAB | BÁO CÁO ĐỒ ÁN")
        canvas.line(22 * mm, 14 * mm, w - 22 * mm, 14 * mm)
        canvas.drawString(22 * mm, 9 * mm, "Bản dựa trên mã nguồn và artifact trong project")
        canvas.drawRightString(w - 22 * mm, 9 * mm, str(doc.page))
        canvas.restoreState()


def md_table(lines: list[str], available_width: float) -> Table:
    rows = []
    for line in lines:
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append([Paragraph(inline_markup(cell), styles["SmallCustom"]) for cell in cells])
    if not rows:
        return Table([[]])
    count = max(len(row) for row in rows)
    for row in rows:
        row.extend([Paragraph("", styles["SmallCustom"])] * (count - len(row)))
    col_width = available_width / count
    table = Table(rows, colWidths=[col_width] * count, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1DCE4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FA")]),
    ]))
    return table


def parse_markdown(source: str, width: float):
    lines = source.splitlines()
    story = []
    i = 0
    cover = True
    first_title_seen = False
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            i += 1
            continue
        if line == "---":
            story.append(Spacer(1, 3 * mm))
            i += 1
            continue
        if line.startswith("```"):
            language = line[3:].strip().lower()
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            if language == "mermaid":
                story.append(ArchitectureDiagram(width))
            else:
                escaped = "\n".join(block).replace("–", "-").replace("—", "-").replace("‑", "-")
                story.append(Preformatted(html.escape(escaped, quote=False), ParagraphStyle(
                    "Code", fontName="Courier", fontSize=7.2, leading=9.5,
                    backColor=colors.HexColor("#F2F5F7"), borderColor=colors.HexColor("#D8E1E7"),
                    borderWidth=0.4, borderPadding=5, leftIndent=4, rightIndent=4,
                    spaceBefore=2 * mm, spaceAfter=3 * mm, splitLongWords=1,
                )))
            continue
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            group = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                group.append(lines[i])
                i += 1
            story.append(md_table(group, width))
            story.append(Spacer(1, 3 * mm))
            continue
        if line.startswith("# "):
            title = line[2:].strip()
            if first_title_seen:
                story.append(PageBreak())
                cover = False
            first_title_seen = True
            story.append(Paragraph(inline_markup(title), styles["CoverTitle"] if cover else styles["H1Custom"]))
            i += 1
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            if title == "Hệ thống đa tác tử hỗ trợ tìm hiểu và phân loại dữ liệu tiểu đường":
                story.append(Paragraph(inline_markup(title), styles["CoverSubtitle"]))
            elif cover and title == "Tóm tắt":
                story.extend([Spacer(1, 4 * mm), PageBreak()])
                cover = False
                story.append(Paragraph(inline_markup(title), styles["H1Custom"]))
            else:
                story.append(Paragraph(inline_markup(title), styles["H1Custom"]))
            i += 1
            continue
        if line.startswith("### "):
            title = line[4:].strip()
            if title.startswith("5.2 "):
                story.append(PageBreak())
            if cover and title.startswith(("**Dự án:**", "**Kiến trúc:**")):
                story.append(Paragraph(inline_markup(title), styles["CoverMeta"]))
            else:
                story.append(Paragraph(inline_markup(title), styles["H2Custom"]))
            i += 1
            continue
        if line.startswith("#### "):
            story.append(Paragraph(inline_markup(line[5:].strip()), styles["H3Custom"]))
            i += 1
            continue
        if line.startswith("> "):
            story.append(Paragraph(inline_markup(line[2:].strip()), styles["QuoteCustom"]))
            i += 1
            continue
        if re.match(r"^[-*] ", line):
            story.append(Paragraph("• " + inline_markup(line[2:]), styles["BulletCustom"]))
            i += 1
            continue
        if re.match(r"^[0-9]+\. ", line):
            story.append(Paragraph(inline_markup(line), styles["BulletCustom"]))
            i += 1
            continue
        if line == "**Từ khóa:**" or line.startswith("**Dự án:**") or line.startswith("**Kiến trúc:**"):
            style = styles["CoverMeta"] if cover else styles["BodyCustom"]
            story.append(Paragraph(inline_markup(line), style))
            i += 1
            continue
        story.append(Paragraph(inline_markup(line), styles["BodyCustom"]))
        i += 1
    return story


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDocTemplate(str(OUTPUT))
    story = parse_markdown(SOURCE.read_text(encoding="utf-8"), doc.width)
    doc.build(story)
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
