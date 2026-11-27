from pathlib import Path
from io import BytesIO
import json
import re

import pdfplumber
import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, NameObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(r'C:\Users\Fptshop123\Downloads\tieuluan_final_06_03_longth_v3.pdf')
DEST = ROOT / 'output/pdf/Tieu_luan_Nguyen_Trong_Toan_B23DCCN833.pdf'
QA = ROOT / 'tmp/pdfs/toan_review'
QA.mkdir(parents=True, exist_ok=True)
DEST.parent.mkdir(parents=True, exist_ok=True)
pdfmetrics.registerFont(TTFont('ReportTimes', r'C:\Windows\Fonts\times.ttf'))
pdfmetrics.registerFont(TTFont('ReportTimesBold', r'C:\Windows\Fonts\timesbd.ttf'))

# Page number, first original line, last original line, revised paragraph.
# Keep all measurements, code, figures, references and headings in their original places.
EDITS = [
 (8, 'Kết quả được so sánh', 'dụng web và API Python.',
  'Tiểu luận kết nối cơ sở lý thuyết AI với quá trình xây dựng, đánh giá và triển khai mô hình. Các phương pháp được đối chiếu trên cùng tập kiểm tra của từng bộ dữ liệu; kết quả được trình bày cùng notebook, mã nguồn, trọng số, giao diện web và API Python.'),
 (10, 'AI xây dựng hệ thống', 'tầng.',
  'Trí tuệ nhân tạo nghiên cứu cách xây dựng hệ thống có khả năng nhận biết, suy luận và lựa chọn hành động. Trong phạm vi đó, học máy ước lượng quy luật từ dữ liệu, còn học sâu sử dụng mạng nơ-ron nhiều tầng để học các biểu diễn phục vụ nhiệm vụ.'),
 (10, 'AI thay đổi cả cách', 'sang học, công việc',
  'Sự phát triển của AI gắn với thay đổi trong cách biểu diễn tri thức và kiểm chứng kết quả. Hệ luật mã hóa điều kiện do con người xác định; học máy điều chỉnh tham số theo dữ liệu và hàm mục tiêu. Chất lượng dữ liệu vì thế trở thành yêu cầu xuyên suốt.'),
 (10, 'Kiến trúc được chọn', 'nối giữa lý thuyết',
  'Cấu trúc đầu vào định hướng lựa chọn kiến trúc: MLP phù hợp với các thuộc tính dạng bảng của Iris, CNN khai thác lân cận pixel trong MNIST, còn RNN mô hình hóa thứ tự quan sát của FPT. Các ví dụ này giúp liên hệ nguyên lý với thiết kế thực nghiệm.'),
 (13, 'Mỗi hàng là một quan sát', 'thống kê, không phải',
  'Dữ liệu bảng tổ chức các quan sát theo hàng và các thuộc tính theo cột. Một hàng của Iris mô tả mẫu hoa, của Breast Cancer mô tả mẫu khối u, còn Housing mô tả khu vực thống kê. Xác định đúng đơn vị quan sát là cơ sở để diễn giải đầu ra của mô hình.'),
 (13, 'Học có giám sát dùng', 'loài, không phải',
  'Trong học có giám sát, mô hình học ánh xạ từ đầu vào x tới mục tiêu y. Phân loại sử dụng nhãn lớp và thường tối ưu cross-entropy; hồi quy dự báo đại lượng liên tục và có thể dùng MSE. Với Iris, các giá trị 0, 1, 2 biểu thị loài, không biểu thị mức độ lớn nhỏ.'),
 (13, 'Hồi quy tuyến tính tính', 'tuyến.',
  'Hồi quy tuyến tính kết hợp tuyến tính các đặc trưng với một hệ số chặn để ước lượng mục tiêu. Tính đơn giản giúp mô hình trở thành mốc đối chiếu dễ phân tích. Sai số trên tập kiểm tra vẫn cần được đo, nhất là khi quan hệ giữa các biến có tính phi tuyến.'),
 (13, 'Ridge thêm mức phạt', 'tiền xử lý chung',
  'Ridge bổ sung phạt L2 vào hàm mất mát nhằm kiểm soát độ lớn trọng số. Các đặc trưng cần được chuẩn hóa để chênh lệch đơn vị đo không chi phối mức phạt. Cấu hình Housing giữ alpha bằng 1 và sử dụng cùng quy trình tiền xử lý với các mô hình đối chứng.'),
 (46, 'Ảnh số được lưu', 'cấu trúc không gian',
  'Ảnh được biểu diễn dưới dạng tensor gồm chiều cao, chiều rộng và số kênh. Ảnh xám sử dụng một kênh, trong khi RGB có ba kênh màu. Vì nhiệm vụ là phân loại ảnh, cách tổ chức tensor phải bảo toàn quan hệ không gian giữa các pixel trước khi đưa vào CNN.'),
 (46, 'Conv trong các thư viện', 'vị trí lấy tổng',
  'Phép Conv thường được thư viện hiện thực bằng tương quan chéo: kernel dịch trên ảnh, nhân các phần tử tương ứng rồi cộng bias. Các vị trí cùng dùng một bộ trọng số để tạo feature map.'),
 (46, 'Pooling gộp một vùng', 'chỉ chuyển gradient',
  'Pooling rút gọn feature map bằng cách tổng hợp từng vùng lân cận. Average pooling tính giá trị trung bình; max pooling chọn phần tử lớn nhất. Khi lan truyền ngược, gradient được chia đều ở average pooling hoặc chuyển tới vị trí cực đại ở max pooling.'),
 (67, 'Chuỗi thời gian có thứ tự', 'thiếu theo lịch',
  'Phân tích chuỗi giá cần xác định rõ một bước quan sát. Với cổ phiếu, bước thời gian tương ứng một phiên giao dịch có dữ liệu, không bao gồm phiên giả vào cuối tuần. Với vàng SJC, mỗi bước là ngày quan sát sau khi loại bản ghi trùng; số ngày thiếu theo lịch được ghi nhận riêng.'),
 (67, 'Cấu hình many to one', 'điều kiện để tránh',
  'Mô hình many to one sử dụng cửa sổ 30 mức giá để ước lượng giá tại quan sát tiếp theo, với tensor đầu vào dạng (batch, 30, 1). Ở mẫu i, cửa sổ chỉ chứa các chỉ số từ i - 30 tới i - 1; giá tại i là mục tiêu. Ranh giới này cần được giữ nhất quán để tránh rò rỉ thông tin tương lai.'),
 (67, 'Simple RNN cập nhật', 'chiều dài cửa sổ',
  'Ở mỗi bước, Simple RNN kết hợp đầu vào hiện tại, trạng thái ẩn trước đó và bias qua hàm tanh. Các bước thời gian sử dụng chung bộ trọng số. Kéo dài cửa sổ làm tăng số phép tính và đường truyền gradient, nhưng không làm tăng số tham số của lớp hồi quy.'),
 (89, 'Chương này xây dựng', 'nguồn và ứng dụng web.',
  'Nội dung chương đi từ tổ chức dữ liệu và đồ thị tri thức đến truy xuất bằng chứng, hình thành câu trả lời có nguồn và xây dựng giao diện tra cứu trên web.'),
 (89, 'Câu hỏi có thể dùng', 'Vì vậy cần phối hợp',
  'Người dùng có thể hỏi bằng tiếng Việt, sử dụng tên khác của HighBP hoặc giới hạn câu hỏi theo năm và đơn vị. Bộ truy xuất cần nhận diện đúng thực thể, đồng thời tìm được phần giải thích liên quan. Kết hợp văn bản với đồ thị giúp xử lý các yêu cầu này.'),
 (89, 'KG lưu các quan hệ', 'thay được các thông tin',
  'Knowledge Graph biểu diễn những quan hệ có thể kiểm tra, chẳng hạn miền dữ liệu và năm của tài liệu. Quá trình trả lời cần gắn nội dung với mã bằng chứng và nguồn tương ứng. Điểm tương đồng chỉ hỗ trợ xếp hạng tài liệu, không xác nhận độ đúng của kết luận.'),
 (89, 'KG làm rõ ràng buộc', 'quốc tế phải tách',
  'Đồ thị ràng buộc thực thể, thời gian và đơn vị. Tài liệu năm 2020 gắn với year:2020; vàng quốc tế và vàng SJC được phân biệt để tránh chọn sai bằng chứng.'),
 (117, 'Chương trình bày khái niệm', 'trúc phối hợp agent',
  'Chương giới thiệu mô hình agent, hai bài thực hành Q-learning và hệ Diabetes_Agent ở mục 6.6. Bài mê cung và mô phỏng cạnh tranh giảm giá làm rõ cơ chế học từ phần thưởng; phần ứng dụng tiểu đường trình bày cách phân vai và phối hợp các agent chuyên môn.'),
 (117, 'AI Agent, hay agent', 'mục tiêu và quyền',
  'AI Agent là hệ thống quan sát môi trường và lựa chọn hành động hướng tới mục tiêu được giao. Vòng tương tác gồm tiếp nhận thông tin, ra quyết định, thực hiện hành động và nhận phản hồi. Mức tự chủ cần gắn với nhiệm vụ cùng giới hạn quyền hành động, đồng thời cho phép kiểm tra kết quả từng bước.'),
 (117, 'Trạng thái cần biểu diễn', 'được cập nhật từ',
  'Biểu diễn trạng thái phải giữ các thông tin cần cho quyết định. Trong mê cung, đó là tọa độ robot; trong mô phỏng giảm giá, đó là nhu cầu cùng hành động trước của đối thủ. Trạng thái mô tả tình huống đang gặp, còn bảng Q tích lũy kinh nghiệm để điều chỉnh đánh giá hành động.'),
 (150, 'Báo cáo trình bày sáu', 'môi trường agent.',
  'Sáu chương liên kết tiến trình phát triển AI với các phương pháp ML, CNN, RNN, Knowledge Graph, RAG và agent. Qua dữ liệu bảng, ảnh, chuỗi giá và các môi trường mô phỏng, tiểu luận làm rõ quan hệ giữa giả định mô hình, cách hiện thực và phạm vi đánh giá.'),
 (150, 'Các chỉ số được đánh giá', 'xuất đường đi, khám phá',
  'Việc đánh giá cần bám sát từng nhiệm vụ và mô hình đối chứng. Điểm số tốt trên bộ hỏi đáp nhỏ chưa đủ chứng minh khả năng tổng quát hóa. Với agent, kết quả còn chịu ảnh hưởng của seed, cơ chế khám phá, trạng thái đã gặp và phương pháp trích xuất chính sách.'),
]

reader = PdfReader(SOURCE)
writer = PdfWriter()
writer.clone_document_from_reader(reader)
plumber = pdfplumber.open(SOURCE)
audit = []
groups = {}
for page_no, start, end, text in EDITS:
    lines = plumber.pages[page_no-1].extract_text_lines()
    first = next(i for i, l in enumerate(lines) if l['text'].startswith(start))
    last = next(i for i in range(first, len(lines)) if lines[i]['text'].startswith(end))
    selected = lines[first:last+1]
    y_first = float(selected[0]['chars'][0]['matrix'][5])
    y_last = float(selected[-1]['chars'][0]['matrix'][5])
    size = float(selected[0]['chars'][0]['size'])
    x = min(l['x0'] for l in selected)
    # Match the body column of the retained reference.
    width = 524.4 - x
    wrapped = simpleSplit(text, 'ReportTimes', size, width)
    if len(wrapped) > len(selected):
        raise ValueError(f'Paragraph overflow on page {page_no}: {len(wrapped)} > {len(selected)}')
    groups.setdefault(page_no, []).append(dict(x=x, width=width, y_first=y_first, y_last=y_last,
        size=size, wrapped=wrapped, baselines=[float(l['chars'][0]['matrix'][5]) for l in selected],
        xmin=60, xmax=550))
    audit.append(dict(page=page_no, before=' '.join(l['text'] for l in selected), after=text,
                      original_lines=len(selected), revised_lines=len(wrapped)))

# Replace entire cover rows, preserving the original label and column positions.
groups[1] = [dict(xmin=140, xmax=500, y_first=375.29, y_last=375.29, cover=True,
                  baseline=375.29, label='Họ và tên sinh viên', value='Nguyễn Trọng Toàn'),
             dict(xmin=140, xmax=500, y_first=350.09, y_last=350.09, cover=True,
                  baseline=350.09, label='Mã sinh viên', value='B23DCCN833')]

for page_no, regions in groups.items():
    page = writer.pages[page_no-1]
    stream = ContentStream(page.get_contents(), writer)
    clean = []
    tx = ty = None
    removed = 0
    for operands, op in stream.operations:
        if op == b'BT':
            tx = ty = None
        elif op == b'Tm':
            tx, ty = float(operands[4]), float(operands[5])
        elif op in (b'Td', b'TD') and tx is not None:
            tx += float(operands[0]); ty += float(operands[1])
        if op in (b'Tj', b'TJ', b"'", b'"') and tx is not None and any(
                r['xmin'] <= tx <= r['xmax'] and r['y_last']-0.8 <= ty <= r['y_first']+0.8
                for r in regions):
            removed += 1
            continue
        clean.append((operands, op))
    assert removed > 0
    stream.operations = clean
    page[NameObject('/Contents')] = writer._add_object(stream)
    overlay = BytesIO()
    c = canvas.Canvas(overlay, pagesize=(float(page.mediabox.width), float(page.mediabox.height)))
    for r in regions:
        c.setFont('ReportTimes', r.get('size', 12))
        if r.get('cover'):
            c.setFont('ReportTimesBold', 12)
            c.drawString(148.7, r['baseline'], r['label'])
            c.drawString(304.61, r['baseline'], ':')
            c.drawString(311.606, r['baseline'], r['value'])
        else:
            for line, baseline in zip(r['wrapped'], r['baselines']):
                c.drawString(r['x'], baseline, line)
    c.save()
    page.merge_page(PdfReader(overlay).pages[0])

# Remove the stale accessibility tree, whose /ActualText could retain replaced strings.
# Bookmarks, hyperlinks and page labels remain attached to the original pages.
writer._root_object.pop('/StructTreeRoot', None)
writer._root_object.pop('/MarkInfo', None)
writer.add_metadata({'/Title': 'Tiểu luận tổng hợp Phát triển các hệ thống thông minh',
                     '/Author': 'Nguyễn Trọng Toàn', '/Subject': 'B23DCCN833'})
with DEST.open('wb') as f:
    writer.write(f)

result = PdfReader(DEST)
assert len(result.pages) == len(reader.pages) == 151
assert 'Nguyễn Trọng Toàn' in result.pages[0].extract_text()
assert 'B23DCCN833' in result.pages[0].extract_text()
assert 'Trần Hải Long' not in result.pages[0].extract_text()
assert 'B23DCCN510' not in result.pages[0].extract_text()
for i in range(1, 7):
    assert result.pages[i].extract_text() == reader.pages[i].extract_text(), f'TOC changed: {i+1}'
for i in range(151):
    if i+1 not in groups:
        assert result.pages[i].get_contents().get_data() == reader.pages[i].get_contents().get_data(), i+1
(QA/'changes.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')

doc = pdfium.PdfDocument(str(DEST))
for page_no in sorted(groups):
    p = doc[page_no-1]
    p.render(scale=1.5).to_pil().save(QA/f'page-{page_no}.png')
    p.close()
print(json.dumps({'output': str(DEST), 'pages': len(result.pages), 'changed_pages': sorted(groups),
                  'revised_paragraphs': len(EDITS), 'toc_unchanged': True}, ensure_ascii=False))
