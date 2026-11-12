// Demo graph for Vinmec-based diabetes knowledge.
// Run in Neo4j Browser. MERGE makes this script safe to re-run.
// For application vector retrieval and complete current chunks, run:
// python scripts/import_knowledge_graph.py --force

CREATE CONSTRAINT source_document_id IF NOT EXISTS
FOR (d:SourceDocument) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT document_chunk_id IF NOT EXISTS
FOR (c:DocumentChunk) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT entity_key IF NOT EXISTS
FOR (e:Entity) REQUIRE e.key IS UNIQUE;
CREATE CONSTRAINT fact_id IF NOT EXISTS
FOR (f:Fact) REQUIRE f.id IS UNIQUE;

UNWIND [
  {id:'overview', title:'Tổng quan về bệnh đái tháo đường', publisher:'Vinmec',
   url:'https://www.vinmec.com/vie/bai-viet/tong-quan-ve-benh-dai-thao-duong-vi', accessed:'2026-11-11',
   text:'Đái tháo đường là rối loạn chuyển hóa với tình trạng tăng glucose máu, liên quan đến khiếm khuyết trong tiết insulin, tác động của insulin hoặc cả hai. Tăng glucose kéo dài có thể gây tổn thương ở tim mạch, thận, mắt và thần kinh. Các thể thường được nhắc đến gồm type 1, type 2 và đái tháo đường thai kỳ.'},
  {id:'symptoms', title:'Triệu chứng có thể gặp', publisher:'Vinmec',
   url:'https://www.vinmec.com/vie/bai-viet/tong-quan-ve-benh-dai-thao-duong-vi', accessed:'2026-11-11',
   text:'Dấu hiệu có thể gồm đi tiểu thường xuyên, rất khát hoặc đói, mệt mỏi, nhìn mờ, vết thương chậm lành, sụt cân không chủ ý hoặc tê đau ran tay chân. Người mắc type 2 đôi khi có triệu chứng nhẹ hoặc không nhận ra. Triệu chứng riêng lẻ không xác nhận cũng không loại trừ bệnh.'},
  {id:'risk', title:'Yếu tố nguy cơ đái tháo đường type 2', publisher:'Vinmec',
   url:'https://www.vinmec.com/vie/bai-viet/cac-yeu-nguy-co-chinh-gay-benh-dai-thao-duong-type-2-o-viet-nam-vi', accessed:'2026-11-11',
   text:'Các yếu tố nguy cơ được Vinmec nêu gồm thừa cân, ít hoạt động thể lực, tiền sử gia đình mắc đái tháo đường, tiền sử đái tháo đường thai kỳ, tuổi cao hơn, tăng huyết áp và rối loạn lipid máu. Có yếu tố nguy cơ không đồng nghĩa với đã mắc bệnh.'},
  {id:'testing', title:'Xét nghiệm chẩn đoán đái tháo đường', publisher:'Vinmec',
   url:'https://www.vinmec.com/vie/bai-viet/xet-nghiem-chan-doan-dai-thao-duong-vi', accessed:'2026-11-11',
   text:'Các xét nghiệm thường dùng gồm glucose huyết tương lúc đói, nghiệm pháp dung nạp glucose đường uống, HbA1c và glucose máu ngẫu nhiên trong bối cảnh có triệu chứng phù hợp. HbA1c phản ánh mức glucose trung bình trong khoảng hai đến ba tháng. Kết quả cần được diễn giải theo hướng dẫn chuyên môn và bối cảnh sức khỏe.'},
  {id:'prevention', title:'Phát hiện và phòng ngừa tiền đái tháo đường', publisher:'Vinmec',
   url:'https://www.vinmec.com/vie/bai-viet/phat-hien-va-phong-ngua-tien-dai-thao-duong-vi', accessed:'2026-11-11',
   text:'Tiền đái tháo đường thường ít biểu hiện rõ; việc đánh giá có thể dùng HbA1c, glucose lúc đói và nghiệm pháp dung nạp glucose. Duy trì cân nặng phù hợp, ăn uống cân đối và hoạt động thể lực thường xuyên là những cách có thể giúp giảm nguy cơ tiến triển type 2. Đây là thông tin chung, không phải kế hoạch điều trị cá nhân.'}
] AS doc
MERGE (d:SourceDocument {id:doc.id})
SET d.title=doc.title, d.publisher=doc.publisher, d.url=doc.url, d.accessed=doc.accessed
MERGE (c:DocumentChunk {id:doc.id + ':cypher'})
SET c.text=doc.text, c.title=doc.title, c.publisher=doc.publisher,
    c.url=doc.url, c.accessed=doc.accessed
MERGE (d)-[:HAS_CHUNK]->(c);

UNWIND [
  {id:'fact-diabetes-insulin', subject_label:'Disease', subject:'Đái tháo đường', subject_key:'Disease:dai thao duong', predicate:'ASSOCIATED_WITH', object_label:'Biomarker', object:'Insulin', object_key:'Biomarker:insulin', document_id:'overview'},
  {id:'fact-diabetes-glucose', subject_label:'Disease', subject:'Đái tháo đường', subject_key:'Disease:dai thao duong', predicate:'ASSOCIATED_WITH', object_label:'Biomarker', object:'Glucose máu', object_key:'Biomarker:glucose mau', document_id:'overview'},
  {id:'fact-diabetes-polydipsia', subject_label:'Disease', subject:'Đái tháo đường', subject_key:'Disease:dai thao duong', predicate:'HAS_SYMPTOM', object_label:'ClinicalCondition', object:'Khát nhiều', object_key:'ClinicalCondition:khat nhieu', document_id:'symptoms'},
  {id:'fact-diabetes-polyuria', subject_label:'Disease', subject:'Đái tháo đường', subject_key:'Disease:dai thao duong', predicate:'HAS_SYMPTOM', object_label:'ClinicalCondition', object:'Tiểu nhiều', object_key:'ClinicalCondition:tieu nhieu', document_id:'symptoms'},
  {id:'fact-type2-overweight', subject_label:'Disease', subject:'Đái tháo đường type 2', subject_key:'Disease:dai thao duong type 2', predicate:'ASSOCIATED_WITH', object_label:'RiskFactor', object:'Thừa cân', object_key:'RiskFactor:thua can', document_id:'risk'},
  {id:'fact-type2-inactivity', subject_label:'Disease', subject:'Đái tháo đường type 2', subject_key:'Disease:dai thao duong type 2', predicate:'ASSOCIATED_WITH', object_label:'RiskFactor', object:'Ít hoạt động thể lực', object_key:'RiskFactor:it hoat dong the luc', document_id:'risk'},
  {id:'fact-hba1c-average', subject_label:'Biomarker', subject:'HbA1c', subject_key:'Biomarker:hba1c', predicate:'REFLECTS', object_label:'Biomarker', object:'Glucose máu trung bình 2–3 tháng', object_key:'Biomarker:glucose mau trung binh 2 3 thang', document_id:'testing'},
  {id:'fact-testing-blood', subject_label:'Disease', subject:'Đái tháo đường', subject_key:'Disease:dai thao duong', predicate:'MEASURED_BY', object_label:'MedicalGuideline', object:'Xét nghiệm glucose máu và HbA1c', object_key:'MedicalGuideline:xet nghiem glucose mau va hba1c', document_id:'testing'},
  {id:'fact-prevention-activity', subject_label:'Recommendation', subject:'Hoạt động thể lực thường xuyên', subject_key:'Recommendation:hoat dong the luc thuong xuyen', predicate:'RECOMMENDED_FOR', object_label:'Disease', object:'Giảm nguy cơ đái tháo đường type 2', object_key:'Disease:giam nguy co dai thao duong type 2', document_id:'prevention'},
  {id:'fact-prevention-weight', subject_label:'Recommendation', subject:'Duy trì cân nặng phù hợp', subject_key:'Recommendation:duy tri can nang phu hop', predicate:'RECOMMENDED_FOR', object_label:'Disease', object:'Giảm nguy cơ đái tháo đường type 2', object_key:'Disease:giam nguy co dai thao duong type 2', document_id:'prevention'}
] AS row
MATCH (c:DocumentChunk {id:row.document_id + ':cypher'})
MERGE (s:Entity {key:row.subject_key})
SET s.name=row.subject, s.normalized_name=substring(row.subject_key, size(row.subject_label) + 1), s.kind=row.subject_label
MERGE (o:Entity {key:row.object_key})
SET o.name=row.object, o.normalized_name=substring(row.object_key, size(row.object_label) + 1), o.kind=row.object_label
MERGE (f:Fact {id:row.id})
SET f.predicate=row.predicate
MERGE (s)-[:HAS_FACT]->(f)
MERGE (f)-[:RELATES_TO]->(o)
MERGE (f)-[:SUPPORTED_BY]->(c);

// Preview imported facts:
// MATCH (s:Entity)-[:HAS_FACT]->(f:Fact)-[:RELATES_TO]->(o:Entity)
// MATCH (f)-[:SUPPORTED_BY]->(c:DocumentChunk)
// RETURN s.name AS subject, f.predicate AS relation, o.name AS object,
//        c.title AS source, c.url AS source_url LIMIT 50;
