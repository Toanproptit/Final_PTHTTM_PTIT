from pathlib import Path
from io import BytesIO
import json, re, math
import pdfplumber
import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, NameObject, NumberObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'output/pdf/Tieu_luan_Nguyen_Trong_Toan_B23DCCN833.pdf'
DEST=ROOT/'output/pdf/Tieu_luan_Nguyen_Trong_Toan_B23DCCN833_v2.pdf'
QA=ROOT/'tmp/pdfs/toan_v2_review'
QA.mkdir(parents=True,exist_ok=True)
for name,file in [('T','times.ttf'),('TB','timesbd.ttf'),('TI','timesi.ttf'),('TBI','timesbi.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(Path(r'C:\Windows\Fonts')/file)))

# Each edit covers whole original lines and is constrained to their existing space.
EDITS=[
(9,'Quy trình gồm khảo cứu','và test dùng đánh giá cuối.',
 'Quy trình nghiên cứu gồm khảo sát tài liệu, phân tích dữ liệu, kiểm tra đạo hàm và so sánh mô hình. Nguồn được nhận diện bằng URL, kích thước và SHA-256. Tiền xử lý học trên train; validation chọn checkpoint; test dành cho đánh giá cuối.'),
(9,'Độ trễ là trung vị','một seed và một cách chia tập',
 'Chi phí suy luận được đo bằng trung vị 20 lượt sau warm-up, chưa gồm nạp mô hình, HTTP và dựng giao diện. Dung lượng phụ thuộc định dạng checkpoint; số tham số được đếm trên kiến trúc. Kết quả chỉ đại diện cho cấu hình và cách chia tập đã sử dụng.'),
(11,'Hệ chuyên gia gồm','cập nhật khi nghiệp vụ',
 'Hệ chuyên gia tổ chức tri thức thành cơ sở luật và bộ suy diễn, phù hợp nhiệm vụ hẹp với quy tắc ổn định. Khó khăn triển khai nằm ở thu thập tri thức, giải quyết luật xung đột và duy trì tính nhất quán khi nghiệp vụ thay đổi.'),
(11,'Mùa đông AI là','quá cao, không chỉ',
 'Các mùa đông AI phản ánh khoảng cách giữa kỳ vọng và khả năng thực hiện. Khi hiệu quả chưa đáp ứng cam kết, đầu tư và nguồn lực suy giảm. Nguyên nhân gồm hạn chế tính toán, dữ liệu chưa đủ và chi phí ứng dụng cao.'),
(11,'Học máy tìm tham số','mẫu mới, không chỉ',
 'Mục tiêu của học máy là dự báo trên quan sát chưa dùng để học. Vì thế, sai số huấn luyện cần được đọc cùng sai số ngoài mẫu. Một mô hình khớp rất tốt tập train vẫn có thể thất bại khi phân bố đầu vào thay đổi.'),
(11,'Mô hình tuyến tính có','ích thực tế của mạng',
 'Các mô hình đối chứng cung cấp nhiều cách tiếp cận: tuyến tính ưu tiên sự đơn giản, cây phân hoạch không gian đặc trưng, Random Forest tổng hợp nhiều cây và SVM sử dụng biên phân tách. So sánh này giúp xác định khi nào mạng sâu đem lại lợi ích.'),
(11,'Validation dùng chọn','thống kê.',
 'Validation phục vụ chọn cấu hình; test chỉ dùng để chấm kết quả cuối. Điều chỉnh nhiều lần theo test làm suy yếu tính độc lập của đánh giá. Các framework cần dùng cùng mẫu kiểm tra, cùng tiền xử lý và seed được ghi lại; khác biệt nhỏ cần thêm nhiều lần chạy để kiểm chứng.'),
(12,'RNN cập nhật trạng thái','ngôn ngữ, encoder-decoder',
 'RNN xử lý chuỗi bằng trạng thái ẩn theo từng bước. LSTM và GRU dùng cổng để điều tiết thông tin. Với ngôn ngữ, encoder-decoder và attention mở rộng liên kết giữa các vị trí.'),
(12,'Transformer dùng attention','và công cụ. Báo cáo',
 'Transformer liên kết các vị trí bằng attention. Mô hình nền tảng mở rộng quy mô học; RAG bổ sung tài liệu tại thời điểm truy vấn; agent tổ chức việc sử dụng công cụ. Các hướng này cung cấp bối cảnh cho nội dung thực hành trong tiểu luận.'),
(14,'Logistic Regression biến','đây không có nghĩa',
 'Logistic Regression chuyển điểm tuyến tính thành xác suất nhị phân qua sigmoid. Với nhiều lớp, softmax chuẩn hóa vector điểm thành phân bố xác suất. Mặc dù có tên regression, mô hình ở mục này được dùng cho nhiệm vụ phân loại.'),
(14,'Ngưỡng hoặc argmax','mẫu; đánh giá chất lượng',
 'Quyết định phân lớp được tạo bằng ngưỡng hoặc argmax. Độ tin cậy của xác suất cần được đánh giá riêng với độ đúng của nhãn. Trong Breast Cancer, macro F1 bổ sung góc nhìn về cả hai lớp thay vì chỉ dựa vào accuracy.'),
(14,'Cây quyết định chia','thay đổi mạnh khi',
 'Cây quyết định học các điều kiện chia trên từng thuộc tính và đưa ra dự báo tại nút lá. Cách chia này biểu diễn được quan hệ phi tuyến. Cây quá sâu dễ khớp cả nhiễu, khiến dự báo nhạy với thay đổi của tập huấn luyện.'),
(15,'SVM dùng các điểm','của SVM cần được',
 'Biên quyết định của SVM được xác định bởi các điểm hỗ trợ. Với kernel RBF, độ gần phụ thuộc khoảng cách trong không gian đặc trưng. Chuẩn hóa giúp hạn chế việc một thuộc tính có trị số lớn chi phối khoảng cách.'),
(15,'SVM bổ sung đối chứng','các thư viện.',
 'SVM được đặt cạnh mô hình tuyến tính, Random Forest và MLP để so sánh các giả định học khác nhau. Ba bản MLP bằng NumPy, Keras và PyTorch bổ sung phép đối chiếu cách hiện thực cùng kiến trúc.'),
(16,'ANN nhận đặc trưng','optimizer dùng gradient',
 'Mạng ANN biến đổi đặc trưng qua các tầng ẩn tới đầu ra. Mỗi tầng tính z = W a + b rồi áp dụng hàm kích hoạt. Backprop truyền gradient về các trọng số; optimizer dùng gradient để cập nhật tham số.'),
(16,'Ba hàm mất mát thường','BCE trực tiếp từ logits',
 'Lựa chọn loss phụ thuộc kiểu mục tiêu như Bảng trên: MSE cho giá trị liên tục, BCE cho hai lớp và CCE cho nhiều lớp. Nhãn one-hot chỉ giữ hạng ứng với lớp đúng trong CCE. Với BCEWithLogitsLoss, đầu vào là logits, không áp dụng sigmoid trước.'),
(17,'Batch là nhóm mẫu','batch cuối nhỏ hơn',
 'Batch xác định số mẫu trong một lần cập nhật; epoch là một lượt qua tập train. Mẫu được hoán vị với seed cố định. Loss của epoch lấy trung bình theo số mẫu, nhờ đó batch cuối có kích thước nhỏ hơn vẫn được tính đúng trọng số.'),
(47,'Kết nối phần dư','Kết nối phần dư',None),
(106,'Gói Tieu_luan_02_Netlify.zip','LLM hoặc API key',
 'Gói triển khai gồm trang HTML, stylesheet, mô-đun giao diện, bộ truy xuất và dữ liệu JSON. Máy chủ tệp tĩnh cung cấp các thành phần này; nhánh trả lời trích xuất chạy từ chỉ mục đã xuất, không cần khóa API trên trình duyệt.'),
(106,'Hình mục 5.7.2','triển khai.',
 'Các hình ở mục 5.7 mô tả lại luồng dữ liệu và bằng chứng của từng miền. Đây là sơ đồ chức năng, dùng để giải thích thiết kế và đối chiếu nguồn trong câu trả lời.'),
(106,'Thao tác sử dụng. Người dùng','hỏi và bằng chứng',
 'Luồng tra cứu tiểu đường nhận câu hỏi về nhãn khảo sát, BMI, huyết áp hoặc nhóm tuổi. Bộ truy xuất kết hợp chọn tài liệu theo văn bản và thực thể. Đầu ra gắn câu trả lời với ID nguồn, tài liệu liên quan và điểm xếp hạng để kiểm tra.'),
(107,'Thao tác sử dụng. Quy trình','hôm nay.',
 'Tra cứu chứng khoán cần nêu rõ mã và thời điểm. Tài liệu s_FPT có mốc cuối 14/03/2025 với giá 131.400 VND/cổ phiếu; VNM ghi 62.100 VND/cổ phiếu. Ngày cuối của IMP khác các mã còn lại, vì vậy từng kết quả cần hiển thị mốc riêng.'),
(108,'Thao tác sử dụng. Ví dụ','nguồn để kiểm tra',
 'Với câu hỏi về vàng năm 2020, hệ thống nhận diện year:2020 rồi truy xuất g_2020. Tài liệu và đồ thị phải dùng cùng thống kê theo năm; ID nguồn cho phép đối chiếu lại phép tổng hợp trong notebook.'),
(109,'Phần 5.1–5.8 trình bày','cấu trúc và số liệu',
 'Hệ tra cứu thống kê dùng NetworkX và TF-IDF được phân biệt với KB y khoa của Diabetes_Agent. Mục này tập trung vào lưu trữ, truy vấn và truy nguồn tri thức trong Neo4j, dựa trên mã nguồn, bản seed [20] và báo cáo [21].'),
(118,'Epsilon-greedy chọn ngẫu','hưởng ngẫu nhiên.',
 'Epsilon-greedy dành xác suất ε cho khám phá và chọn hành động có Q lớn nhất ở phần còn lại. Giảm ε chỉ thay đổi mức khám phá. Cần đánh giá chính sách sau huấn luyện trên các seed và kịch bản chung, tách khỏi phần thưởng thu trong lúc còn thử ngẫu nhiên.'),
(143,'Một agent đơn có thể','ngân sách, nên lợi ích',
 'Phân vai tách dự báo, truy xuất và bộ nhớ thành các trách nhiệm có đầu vào, đầu ra rõ ràng. Trace giúp theo dõi bước phát sinh lỗi. Lợi ích ở đây thuộc tổ chức phần mềm; chưa có đối chứng trên cùng ngân sách để kết luận hệ đa agent chính xác hơn agent đơn.'),
(144,'Quy trình dưới đây dùng','[21]. Cả hai chưa xác nhận',
 'Kịch bản trình diễn yêu cầu mô hình, cấu hình kết nối và KB đã nạp. Các sơ đồ tiếp theo minh họa cấu trúc kết quả, bằng chứng và dấu vết xử lý; số liệu ba lượt API được trích riêng từ [21]. Chúng không đại diện cho một lần chạy mới trong bản biên tập.'),
]
EDITS=[e for e in EDITS if e[3] is not None]

# Redraw application illustrations as diagrams, not fabricated screenshots.
FIGURES={
8:[('pipeline','Hình M.1 Quy trình từ dữ liệu gốc đến hệ thống suy luận')],
36:[('iris','Hình 2.18 Luồng suy luận phân loại Iris')],
39:[('breast','Hình 2.23 Luồng phân loại đặc trưng Breast Cancer Wisconsin')],
42:[('housing','Hình 2.28 Luồng ước lượng giá trung vị khu vực California')],
59:[('mnist','Hình 3.15 Luồng xử lý một ảnh chữ số MNIST')],
62:[('fashion','Hình 3.20 Luồng xử lý ảnh trang phục Fashion-MNIST')],
65:[('cifar','Hình 3.25 Luồng phân loại ảnh màu CIFAR-10'),('cnn','Hình 3.26 Dòng tensor của CNN sáu lớp tích chập')],
82:[('fpt','Hình 4.17 Luồng dự báo FPT từ cửa sổ 30 quan sát')],
85:[('vnm','Hình 4.22 Luồng dự báo VNM từ cửa sổ 30 quan sát')],
88:[('sjc','Hình 4.27 Luồng dự báo vàng SJC từ cửa sổ 30 quan sát')],
107:[('diabetes_kg','Hình 5.13 Tra cứu BMI và đối chiếu bằng chứng khảo sát')],
108:[('stocks_kg','Hình 5.14 Tra cứu mã FPT và nguồn dữ liệu lịch sử'),('gold_kg','Hình 5.15 Truy xuất tài liệu vàng theo thực thể năm')],
112:[('kb','Hình 5.17 Năm quan hệ nghiệp vụ được biểu diễn lại từ seed [20]')],
115:[('evidence','Hình 5.18 Minh họa đường truy nguồn tri thức cho một hồ sơ')],
116:[('chat','Hình 5.19 Minh họa quy trình hỏi đáp có trích dẫn')],
146:[('output','Hình 6.8 Minh họa cấu trúc kết quả hỗ trợ sàng lọc')],
147:[('safety','Hình 6.9 Minh họa kết hợp bằng chứng và kiểm tra luật')],
148:[('trace','Hình 6.10 Minh họa các bước xử lý và thông tin phiên bản')],
}

APP_PROSE={
36:('Kết quả suy luận:','Ứng dụng chỉ được huấn luyện',
 'Đầu vào gồm bốn số đo hoa theo centimet, được chuẩn hóa bằng thống kê của train. MLP sử dụng lớp ẩn ReLU 32 nơ-ron và đầu ra softmax để phân biệt ba loài Iris.\nKết quả cần được đọc trong phạm vi ba loài đã học. Softmax cao không xác nhận mẫu thuộc phân bố huấn luyện; ảnh hoa không phải đầu vào của mô hình này.'),
39:('Kết quả suy luận:','Phạm vi kết quả.',
 'Mô hình nhận vector 30 đặc trưng số theo đúng thứ tự đã lưu, loại mã định danh và dùng bộ chuẩn hóa của train. Đầu ra phân biệt nhãn B và M; các bảng kế tiếp đánh giá mô hình trên cùng tập kiểm tra.\nỨng dụng phục vụ nghiên cứu dữ liệu đặc trưng. Kết quả chưa được kiểm định lâm sàng và không thay thế đánh giá của người có chuyên môn.'),
42:('Kết quả suy luận:','Dữ liệu thuộc California',
 'Đầu vào mô tả khu vực bằng tọa độ, tuổi nhà, số phòng, dân số, số hộ, thu nhập và vị trí so với biển. Đặc trưng số được chuẩn hóa; ocean_proximity được one-hot. Dự báo MLP được chuyển ngược về USD.\nDữ liệu thuộc California năm 1990 và mục tiêu là giá trung vị khu vực. Sai số test không phải khoảng tin cậy cho một căn nhà, cũng không mô tả giá thị trường hiện tại.'),
59:('Kết quả suy luận:','Phạm vi kết quả.',
 'Một ảnh chữ số được đưa về một kênh 28 × 28 rồi chia pixel cho 255. CNN cơ sở thực hiện tích chập, ReLU, average pooling và phân loại bằng trọng số đã huấn luyện.\nPhân bố xác suất dùng để so sánh các lớp trong MNIST. Hệ thống xử lý một chữ số; ảnh nhiều ký tự, nét lệch hoặc nền khác dữ liệu nguồn có thể làm dự báo sai. Đây không phải quy trình OCR cả tài liệu.'),
62:('Kết quả suy luận:','Phạm vi kết quả.',
 'Ảnh trang phục được biểu diễn bằng tensor xám 1 × 28 × 28. CNN sử dụng kiến trúc cơ sở nhưng có bộ trọng số riêng cho Fashion-MNIST; đầu ra gồm mười lớp trang phục.\nCác lớp Shirt, Coat và Pullover có thể khó phân biệt ở độ phân giải nhỏ. Ảnh có người mặc, nền phức tạp hoặc nhiều vật thể khác phân bố nguồn nên cần được đánh giá bổ sung.'),
65:('Kết quả suy luận:','Phạm vi kết quả.',
 'Ảnh CIFAR-10 giữ đủ ba kênh RGB với kích thước 32 × 32 và pixel trong [0,1]. Mỗi kernel tích chập nhận cả ba kênh; mô hình dùng bộ trọng số riêng cho miền dữ liệu này.\nMười lớp tạo thành tập nhãn đóng, chưa có lớp unknown. Vì vậy, ảnh ngoài danh sách vẫn có thể nhận một nhãn; xác suất đầu ra cần được đọc cùng giới hạn này.'),
82:('Kết quả suy luận:','Chuỗi ví dụ kết thúc',
 'Đầu vào là 30 giá FPT theo thứ tự thời gian. Chuỗi được chuẩn hóa bằng thống kê train; mô hình dự báo chênh lệch, khôi phục đơn vị rồi cộng với mức giá cuối.\nNguồn ví dụ kết thúc 30/06/2023. Kết quả mô tả dự báo một bước trên lịch sử, chưa đánh giá lợi nhuận giao dịch và chưa xác nhận đầy đủ điều chỉnh cổ tức, chia tách.'),
}

reader=PdfReader(SOURCE); writer=PdfWriter(); writer.clone_document_from_reader(reader)
pl=pdfplumber.open(SOURCE)
regions={}; log=[]; changed=set(); removed_urls=[]

def add_region(n,lines,text,font='T',size=None):
    if not lines:return
    ys=[float(l['chars'][0]['matrix'][5]) for l in lines]
    x=min(l['x0'] for l in lines); sz=size or float(lines[0]['chars'][0]['size'])
    wrapped=[]
    if text:
        for p in text.split('\n'):
            wrapped.extend(simpleSplit(p,font,sz,524.4-x))
    if len(wrapped)>len(lines):
        raise ValueError(f'Overflow page {n}: {len(wrapped)} > {len(lines)}: {text[:65]}')
    regions.setdefault(n,[]).append(dict(y0=min(ys)-0.8,y1=max(ys)+0.8,x0=60,x1=555,
        x=x,size=sz,font=font,wrapped=wrapped,ys=ys))
    log.append(dict(page=n,old=' '.join(l['text'] for l in lines),new=text or ''))
    changed.add(n)

def select(n,start,end):
    lines=pl.pages[n-1].extract_text_lines()
    a=next(i for i,l in enumerate(lines) if l['text'].startswith(start))
    b=next((i for i in range(a,len(lines)) if lines[i]['text'].startswith(end)),None)
    if b is None:raise ValueError(f'Missing end on page {n}: {end}; available: {[l["text"] for l in lines[a:]]}')
    return lines[a:b+1]

for n,start,end,text in EDITS:
    add_region(n,select(n,start,end),text)

for n,(start,end,text) in APP_PROSE.items():
    lines=pl.pages[n-1].extract_text_lines()
    a=next(i for i,l in enumerate(lines) if l['text'].startswith(start))
    # Stop before the next numbered section; these paragraphs describe the preceding illustration.
    b=next(i for i in range(a,len(lines)) if re.match(r'^[1-6]\.\d{1,2} ',lines[i]['text']))
    add_region(n,lines[a:b],text)

# Remove demo URLs, including their link labels and wrapped continuation lines.
for n in range(8,151):
    lines=pl.pages[n-1].extract_text_lines()
    for i,l in enumerate(lines):
        if l['text'].startswith(('Link sử dụng:','Đường dẫn sau khi deploy:')):
            add_region(n,[l],None); removed_urls.append(dict(page=n,text=l['text']))

# Captions and adjacent explanations must describe the new illustration honestly.
caption_regions={}
for n,specs in FIGURES.items():
    lines=pl.pages[n-1].extract_text_lines()
    captions=[(i,l) for i,l in enumerate(lines) if re.match(r'^Hình (?:M\.|\d+\.)\d+ ',l['text'])]
    for (kind,newcap),(i,l) in zip(specs,captions):
        selected=[l]
        # The source sometimes wraps a screenshot attribution over two lines.
        if i+1<len(lines) and lines[i+1]['text'].startswith(('40].','dụng chạy cục bộ.','chạy cục bộ.')):
            selected.append(lines[i+1])
        add_region(n,selected,newcap,'TBI',min(float(l['chars'][0]['size']),11))
    caption_regions[n]=captions

add_region(115,select(115,'Giao diện KG cho biết','KB.'),
 'Sơ đồ trình bày cách nối yếu tố, quan hệ, Evidence và Source. Việc kiểm tra một kết luận cần theo được ID bằng chứng tới tài liệu gốc; hình minh họa không đại diện cho một phiên truy vấn vừa thực hiện.')
add_region(116,select(116,'Chat hiển thị đoạn trả lời','chứng → tài liệu gốc.'),
 'Luồng hỏi đáp liên kết khái niệm, quan hệ, đoạn bằng chứng và tài liệu gốc. Sau truy xuất, cần kiểm tra nội dung trả lời có giữ đúng điều kiện và đơn vị của nguồn. Chương 6 kết hợp đầu ra này với dự báo và luật hỗ trợ.')

add_region(148,select(148,'Ba hình trên được chụp','khai.'),
 'Ba sơ đồ trên tóm tắt cấu trúc xử lý: kiểm tra hồ sơ, dự báo theo nhánh, truy xuất bằng chứng, áp dụng luật và lưu dấu vết. Chúng minh họa thiết kế, không biểu diễn một lượt chạy mới.\nKết quả mô hình, ngưỡng và thời gian phản hồi phải được đọc từ artifact hoặc log của cùng phiên bản. Các lượt API ở Bảng 6.17 được trích từ báo cáo [21] và cần được phân biệt với sơ đồ chức năng.\nKhi đánh giá triển khai, cần ghi cấu hình, môi trường, đầu vào và thời điểm đo. Một ảnh giao diện hay một lượt phản hồi chưa đủ để xác nhận độ trễ hoặc chất lượng của toàn hệ thống.')

add_region(85,select(85,'Kết quả suy luận:','chiếu với baseline'),
 'Chuỗi VNM sử dụng 30 quan sát từ cũ tới mới, với bộ trọng số và thống kê chuẩn hóa riêng. Mô hình dự báo delta rồi khôi phục giá ở bước kế tiếp.\nFPT và VNM cùng thuộc nguồn VN30; chuỗi ví dụ kết thúc 30/06/2023. Chất lượng dự báo cần được đối chiếu với baseline giữ giá cuối trên cùng tập test.')
add_region(88,select(88,'Kết quả suy luận:','ngày lịch tiếp theo.'),
 'Đầu vào SJC gồm 30 giá bán dương theo đơn vị triệu đồng/lượng. RNN chuẩn hóa theo train, dự báo delta và khôi phục giá. Dữ liệu mẫu kết thúc 08/10/2026.\nKhoảng trống dữ liệu được giữ nguyên, nên bước dự báo là quan sát kế tiếp, không nhất thiết là ngày lịch kế tiếp. Trong phép đánh giá đã lưu, các mạng chưa vượt baseline giá gần nhất.')

def arrow(c,x1,y1,x2,y2):
    c.setStrokeColorRGB(0,0,0);c.setFillColorRGB(0,0,0);c.setLineWidth(.8)
    c.line(x1,y1,x2,y2)
    a=math.atan2(y2-y1,x2-x1);s=4
    p=c.beginPath();p.moveTo(x2,y2);p.lineTo(x2-s*math.cos(a-.45),y2-s*math.sin(a-.45));p.lineTo(x2-s*math.cos(a+.45),y2-s*math.sin(a+.45));p.close();c.drawPath(p,fill=1,stroke=0)

def box(c,x,y,w,h,title,sub='',sz=10):
    c.setFillColorRGB(.965,.965,.965);c.setStrokeColorRGB(0,0,0);c.setLineWidth(.6)
    c.roundRect(x,y,w,h,4,fill=1,stroke=1)
    c.setFillColorRGB(0,0,0);c.setFont('TB',sz)
    ts=simpleSplit(title,'TB',sz,w-14); ss=simpleSplit(sub,'T',sz-1,w-14) if sub else []
    leading=sz+3;top=y+h/2+(len(ts)+len(ss)-1)*leading/2
    for s in ts:c.drawCentredString(x+w/2,top,s);top-=leading
    c.setFont('T',sz-1)
    for s in ss:c.drawCentredString(x+w/2,top,s);top-=leading

FLOW={
'pipeline':('TỪ DỮ LIỆU ĐẾN SUY LUẬN', [('Nguồn dữ liệu','Kiểm tra schema và chất lượng'),('Chia tập','Train / validation / test'),('Tiền xử lý','Chỉ fit trên train'),('Huấn luyện','Chọn checkpoint bằng validation'),('Đánh giá','Chấm trên test và đối chứng'),('Suy luận','Đóng gói mô hình cùng tiền xử lý')]),
'iris':('PHÂN LOẠI IRIS',[('Đặc trưng đầu vào','Bốn số đo theo centimet'),('Chuẩn hóa','Tham số học từ train'),('MLP','Dense 32 và ReLU'),('Softmax','Ba xác suất lớp'),('Quyết định','Setosa / Versicolor / Virginica')]),
'breast':('PHÂN LOẠI ĐẶC TRƯNG KHỐI U',[('Vector đầu vào','30 đặc trưng số theo schema'),('Tiền xử lý','Loại id, dùng scaler đã lưu'),('MLP','Suy luận từ checkpoint'),('Hai lớp','B / M và xác suất'),('Phạm vi','Dữ liệu nghiên cứu, chưa kiểm định lâm sàng')]),
'housing':('ƯỚC LƯỢNG GIÁ TRUNG VỊ',[('Khu vực California','Đặc trưng số và ocean_proximity'),('Tiền xử lý','Điền thiếu, chuẩn hóa, one-hot'),('MLP hồi quy','Đầu ra ở thang đã chuẩn hóa'),('Khôi phục đơn vị','Inverse transform về USD'),('Diễn giải','Giá trung vị khu vực, dữ liệu năm 1990')]),
'mnist':('PHÂN LOẠI CHỮ SỐ',[('Ảnh đầu vào','Một chữ số, ảnh xám'),('Tensor','1 × 28 × 28, pixel / 255'),('CNN','Conv → ReLU → average pooling'),('Đầu ra','Mười logits và softmax'),('Nhãn dự báo','Một lớp trong các chữ số 0–9')]),
'fashion':('PHÂN LOẠI TRANG PHỤC',[('Ảnh đầu vào','Một vật thể trang phục'),('Tensor','1 × 28 × 28, pixel / 255'),('CNN','Bộ trọng số Fashion-MNIST'),('Softmax','Mười lớp trang phục'),('Đọc kết quả','Chú ý nhầm Shirt / Coat / Pullover')]),
'cifar':('PHÂN LOẠI ẢNH MÀU',[('Ảnh đầu vào','RGB, một đối tượng'),('Tensor','3 × 32 × 32, pixel / 255'),('CNN','Kernel nhận đủ ba kênh'),('Softmax','Mười lớp CIFAR-10'),('Giới hạn','Chưa có lớp unknown')]),
'fpt':('DỰ BÁO MỘT BƯỚC FPT',[('Cửa sổ lịch sử','30 giá, thứ tự cũ → mới'),('Chuẩn hóa','Thống kê của tập train'),('Mô hình hồi quy','Ước lượng chênh lệch'),('Khôi phục giá','Giá cuối + chênh lệch'),('Đầu ra','Giá ở quan sát kế tiếp')]),
'vnm':('DỰ BÁO MỘT BƯỚC VNM',[('Chuỗi đầu vào','30 quan sát VNM theo thời gian'),('Chuẩn hóa','Scaler của tập train'),('RNN / LSTM / GRU','Suy luận từ trọng số đã lưu'),('Khôi phục','Đưa delta về đơn vị giá'),('So sánh','Đối chiếu baseline giữ giá cuối')]),
'sjc':('DỰ BÁO MỘT BƯỚC SJC',[('Chuỗi đầu vào','30 giá vàng SJC'),('Chuẩn hóa','Dùng thống kê train'),('Mô hình chuỗi','Ước lượng biến động kế tiếp'),('Khôi phục giá','Cộng delta với quan sát cuối'),('Đơn vị','Triệu đồng/lượng; không phải LBMA')]),
'diabetes_kg':('TRUY XUẤT THỐNG KÊ BMI',[('Câu hỏi','BMI khác nhau giữa các nhãn thế nào?'),('Thực thể','Nhận diện bmi và miền diabetes'),('Bộ truy xuất','TF-IDF kết hợp điểm thực thể'),('Bằng chứng','Tài liệu d_bmi và nguồn khảo sát'),('Trả lời','Trung vị 27 / 30 / 31 theo nhóm nhãn')]),
'stocks_kg':('TRUY XUẤT CỔ PHIẾU',[('Câu hỏi','Mã FPT và mốc dữ liệu cần tra'),('Thực thể','ticker:FPT, miền stocks'),('Tài liệu','s_FPT và nguồn chuỗi giá'),('Đối chiếu','14/03/2025; 131.400 VND/cổ phiếu'),('Phạm vi','Dữ liệu lịch sử, không phải giá trực tiếp')]),
'gold_kg':('TRUY XUẤT VÀNG THEO NĂM',[('Câu hỏi','Thống kê giá vàng năm 2020'),('Thực thể','year:2020, miền gold'),('Tài liệu','g_2020, g_units, g_quality'),('Kiểm tra','Đúng năm, mẫu số và USD/ounce'),('Đầu ra','Câu trả lời kèm ID bằng chứng')]),
'evidence':('ĐƯỜNG TRUY NGUỒN TRI THỨC',[('Hồ sơ và câu hỏi','Xác định khái niệm liên quan'),('Entity','RiskFactor / Condition / Test / Intervention'),('Quan hệ nghiệp vụ','Kiểm tra kiểu quan hệ và evidence_ids'),('Evidence','Đọc nguyên văn và điều kiện của đoạn'),('Source','Kiểm tra tài liệu gốc, mục và phiên bản')]),
'chat':('HỎI ĐÁP CÓ BẰNG CHỨNG',[('Câu hỏi người dùng','Nhận diện chủ đề và thực thể'),('Kiểm tra phạm vi','Từ chối câu ngoài miền theo quy tắc'),('Truy xuất','Ngữ nghĩa và quan hệ trong KG'),('Đọc bằng chứng','Giữ điều kiện của nguồn'),('Trả lời','Nội dung + trích dẫn + vết truy xuất')]),
'output':('CẤU TRÚC KẾT QUẢ SÀNG LỌC',[('Hồ sơ hợp lệ','Đặc trưng, nhánh sàng lọc và đơn vị'),('Kết quả mô hình','prediction, model_version và ngưỡng'),('Đối chiếu an toàn','Đọc số đo và các luật phù hợp'),('Hướng hỗ trợ','Khuyến nghị trong phạm vi bằng chứng'),('Thông tin trả về','Nguồn, trường điền thiếu và degraded')]),
'safety':('BẰNG CHỨNG VÀ LUẬT',[('Dữ liệu đầu vào','Đối chiếu số đo, schema và đơn vị'),('Dự báo mô hình','Xác suất và mức phân tầng'),('Tri thức truy xuất','Đoạn Evidence cùng Source'),('SafetyGuard','Kiểm tra luật và xử lý xung đột'),('Kết quả tổng hợp','Giữ dấu vết quy tắc đã tác động')]),
'trace':('DẤU VẾT XỬ LÝ',[('Tiếp nhận yêu cầu','Schema, screening_mode và patient_id'),('DiagnosisAgent','Mô hình và phiên bản artifact'),('AdvisorAgent','Bằng chứng và nội dung tư vấn'),('Kiểm tra điều phối','Luật, policy tùy cấu hình và lỗi'),('MemoryAgent','Lưu kết quả, lịch sử và trace')]),
}

def diagram(c,rect,kind):
    x0,top,x1,bottom=rect; W=x1-x0;H=bottom-top;y0=841.92-bottom
    c.saveState();c.translate(x0,y0)
    # Use a normalized drawing space, independent of the previous image resolution.
    if kind=='cnn':
        scale=min(W/450,H/145);c.translate((W-450*scale)/2,(H-145*scale)/2);c.scale(scale,scale)
        steps=[('Đầu vào','C × H × W'),('Khối 1','2 Conv, 4 kênh'),('Khối 2','2 Conv, 6 kênh'),('Khối 3','2 Conv, 8 kênh'),('Dense','Logits')]
        for i,(t,s) in enumerate(steps):
            box(c,i*91,54,82,55,t,s,9)
            if i<4:arrow(c,i*91+82,81,i*91+90,81)
        c.setFont('T',10);c.drawCentredString(225,27,'Conv 3 × 3, padding 1; ReLU; pooling 2 × 2 sau mỗi khối')
    elif kind=='kb':
        scale=min(W/400,H/310);c.translate((W-400*scale)/2,(H-310*scale)/2);c.scale(scale,scale)
        nodes=[(0,140,115,55,'lifestyle_program'),(140,140,115,55,'high_bmi'),(285,140,115,55,'type2_diabetes'),(140,20,115,55,'a1c_test')]
        for x,y,w,h,t in nodes:box(c,x,y,w,h,t,'',10)
        edges=[(115,167,140,167,'ADDRESSES',127,205),(255,167,285,167,'INCREASES_RISK_OF',270,125),(197,140,197,75,'SHOULD_TEST_WITH',260,105)]
        for a,b,d,e,label,tx,ty in edges:
            arrow(c,a,b,d,e);c.setFont('T',7);c.drawCentredString(tx,ty,label)
        c.line(57,195,57,260);c.line(57,260,342,260);arrow(c,342,260,342,195)
        c.setFont('T',8);c.drawCentredString(200,271,'REDUCES_RISK_OF')
        c.line(255,47,342,47);arrow(c,342,47,342,140)
        c.setFont('T',8);c.drawCentredString(350,91,'DIAGNOSES')
    else:
        title,steps=FLOW[kind]
        if W/H>2.5:
            nw,nh=450,130;scale=min(W/nw,H/nh);c.translate((W-nw*scale)/2,(H-nh*scale)/2);c.scale(scale,scale)
            c.setFont('TB',12);c.drawCentredString(225,117,title)
            for i,(t,s) in enumerate(steps):
                row=i//3;col=i%3 if row==0 else 2-i%3;xx=col*153;yy=64-row*59
                box(c,xx,yy,143,45,t,s,9)
                if row==0 and col<2 and i+1<len(steps):arrow(c,xx+143,yy+22,xx+152,yy+22)
                elif row==0 and i+1<len(steps):arrow(c,xx+71,yy,xx+71,yy-14)
                elif col>0 and i+1<len(steps):arrow(c,xx,yy+22,xx-10,yy+22)
        else:
            nw,nh=310,250 if H<270 else 360;scale=min(W/nw,H/nh)
            c.translate((W-nw*scale)/2,(H-nh*scale)/2);c.scale(scale,scale)
            c.setFont('TB',12);c.drawCentredString(nw/2,nh-16,title)
            gap=11; bh=(nh-42-(len(steps)-1)*gap)/len(steps)
            for i,(t,s) in enumerate(steps):
                yy=nh-34-(i+1)*bh-i*gap
                box(c,12,yy,nw-24,bh,t,s,10)
                if i<len(steps)-1:arrow(c,nw/2,yy,nw/2,yy-gap+2)
    c.restoreState()

figure_count=0
for n in sorted(set(regions)|set(FIGURES)):
    page=writer.pages[n-1];rs=regions.get(n,[])
    im=pl.pages[n-1].images if n in FIGURES else []
    names={z['name'] for z in im[:len(FIGURES.get(n,[]))]}
    stream=ContentStream(page.get_contents(),writer);ops=[];tx=ty=None
    for args,op in stream.operations:
        if op==b'BT':tx=ty=None
        elif op==b'Tm':tx,ty=map(float,args[4:6])
        elif op in (b'Td',b'TD') and tx is not None:tx+=float(args[0]);ty+=float(args[1])
        if op==b'Do' and str(args[0]).lstrip('/') in names:continue
        if op in (b'Tj',b'TJ',b"'",b'"') and tx is not None and any(r['x0']<=tx<=r['x1'] and r['y0']<=ty<=r['y1'] for r in rs):continue
        ops.append((args,op))
    stream.operations=ops;page[NameObject('/Contents')]=writer._add_object(stream)
    b=BytesIO();c=canvas.Canvas(b,pagesize=(595.32,841.92));c.setFillColorRGB(0,0,0)
    for r in rs:
        if not r['wrapped']:
            c.setFillColorRGB(1,1,1)
            c.rect(60,min(r['ys'])-3,495,max(r['ys'])-min(r['ys'])+15,stroke=0,fill=1)
    c.setFillColorRGB(0,0,0)
    for r in rs:
        if not r['wrapped']:
            c.setFillColorRGB(1,1,1)
            c.rect(r['x0'],r['y0']-4,r['x1']-r['x0'],r['y1']-r['y0']+16,fill=1,stroke=0)
            c.setFillColorRGB(0,0,0)
        c.setFont(r['font'],r['size'])
        for s,y in zip(r['wrapped'],r['ys']):c.drawString(r['x'],y,s)
    for z,(kind,cap) in zip(im,FIGURES.get(n,[])):
        diagram(c,(float(z['x0']),float(z['top']),float(z['x1']),float(z['bottom'])),kind);figure_count+=1
    c.save();page.merge_page(PdfReader(b).pages[0]);changed.add(n)

# Force black for PDF text, recursively including text inside form XObjects.
seen=set()
def blacken(obj,resources):
    st=ContentStream(obj,writer);out=[]
    for args,op in st.operations:
        if op in (b'Tj',b'TJ',b"'",b'"'):
            out.extend([([NumberObject(0)],b'g'),([NumberObject(0)],b'G')])
        out.append((args,op))
    st.operations=out
    for name,xref in (resources.get('/XObject',{}) or {}).items():
        x=xref.get_object();key=getattr(xref,'idnum',id(x))
        if x.get('/Subtype')=='/Form' and key not in seen:
            seen.add(key);sub=blacken(x,x.get('/Resources',resources));x.set_data(sub.get_data())
    return st

removed_annotations=0
for p in writer.pages:
    st=blacken(p.get_contents(),p.get('/Resources',{}));p[NameObject('/Contents')]=writer._add_object(st)
    if p.get('/Annots'):
        kept=[]
        for a in p['/Annots']:
            obj=a.get_object();uri=str(obj.get('/A',{}).get('/URI',''))
            if uri and any(s in uri.lower() for s in ['netlify.app','localhost','127.0.0.1','?domain=']):removed_annotations+=1
            else:kept.append(a)
        from pypdf.generic import ArrayObject
        p[NameObject('/Annots')]=ArrayObject(kept)

writer.add_metadata({'/Title':'Tiểu luận tổng hợp Phát triển các hệ thống thông minh','/Author':'Nguyễn Trọng Toàn','/Subject':'B23DCCN833'})
writer._root_object.pop('/StructTreeRoot',None);writer._root_object.pop('/MarkInfo',None)
with DEST.open('wb') as f:writer.write(f)
result=PdfReader(DEST)
assert len(result.pages)==151
for i in range(1,7):assert result.pages[i].extract_text()==reader.pages[i].extract_text(),f'TOC mismatch {i+1}'
assert 'zingy-sorbet' not in '\n'.join(p.extract_text() or '' for p in result.pages)
summary={'output':str(DEST),'pages':151,'new_figures':figure_count,'rewritten_regions':len(log),'removed_demo_lines':len(removed_urls),'removed_demo_annotations':removed_annotations,'review_pages':sorted(changed),'toc_unchanged':True}
(QA/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(QA/'changes.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
doc=pdfium.PdfDocument(str(DEST))
for n in sorted(changed):
    p=doc[n-1];p.render(scale=1.4).to_pil().save(QA/f'page-{n}.png');p.close()
print(json.dumps(summary,ensure_ascii=False))
