import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity, ArrowDownRight, ArrowRight, ArrowUpRight, BookOpen, BrainCircuit,
  Check, ChevronDown, ChevronRight, CircleHelp, Clock3, Database, ExternalLink,
  FileText, FlaskConical, HeartPulse, History, LayoutDashboard, LoaderCircle,
  MessageCircle, MoreHorizontal, Play, Plus, RefreshCw, Send, Settings2,
  ShieldCheck, Sparkles, Trash2, Workflow, X,
} from 'lucide-react';

const NAV = [
  { id: 'overview', label: 'Tổng quan', icon: LayoutDashboard },
  { id: 'prediction', label: 'Dự đoán nghiên cứu', icon: Activity },
  { id: 'knowledge', label: 'Kho tri thức', icon: BookOpen },
  { id: 'llm-agent', label: 'LLM Agent', icon: Sparkles, badge: 'NEW' },
  { id: 'workflow', label: 'Workflow', icon: Workflow },
  { id: 'experiments', label: 'Thực nghiệm', icon: FlaskConical },
];
const TITLES = Object.fromEntries(NAV.map((item) => [item.id, item.label]));
const SAMPLE_QUESTIONS = [
  'HbA1c phản ánh điều gì?',
  'Khát nhiều và tiểu nhiều có thể là triệu chứng gì?',
  'Làm sao phòng ngừa type 2?',
];
const DEMO_PROMPTS = [
  'HbA1c phản ánh điều gì? Hãy tìm nguồn trong kho kiến thức.',
  'Tóm tắt kết quả thực nghiệm hiện có của dự án.',
  'Chạy mô hình trên hồ sơ demo mặc định và giải thích giới hạn.',
];

function cx(...parts) { return parts.filter(Boolean).join(' '); }
function escapeMessage(value) { return String(value ?? ''); }
function pct(value) { return value == null ? '—' : `${(Number(value) * 100).toFixed(1)}%`; }
function pretty(value) {
  if (value == null) return '—';
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString('vi-VN') : value.toFixed(3);
  return String(value);
}
function dateTime(value) {
  return value ? new Date(value).toLocaleString('vi-VN', { dateStyle: 'medium', timeStyle: 'short' }) : '';
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(body.detail)
      ? body.detail.map((item) => `${item.loc?.slice(1).join('.')}: ${item.msg}`).join('; ')
      : body.detail;
    throw new Error(detail || `Lỗi HTTP ${response.status}`);
  }
  return body;
}

function IconButton({ title, onClick, children, className = '' }) {
  return <button className={cx('icon-button', className)} title={title} aria-label={title} onClick={onClick}>{children}</button>;
}

function App() {
  const [page, setPage] = useState(() => location.hash.slice(1) || 'overview');
  const [health, setHealth] = useState(null);
  const [schema, setSchema] = useState(null);
  const [experiments, setExperiments] = useState(null);
  const [error, setError] = useState('');
  const [mobileNav, setMobileNav] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  const go = useCallback((next) => {
    setPage(next);
    setMobileNav(false);
    setError('');
    if (location.hash !== `#${next}`) history.replaceState(null, '', `#${next}`);
  }, []);

  const refreshExperiments = useCallback(async () => {
    setError('');
    try { setExperiments(await api('/api/experiments')); }
    catch (e) { setError(e.message); }
  }, []);

  useEffect(() => {
    let active = true;
    Promise.all([api('/api/health'), api('/api/schema'), api('/api/experiments')])
      .then(([nextHealth, nextSchema, nextExperiments]) => {
        if (!active) return;
        setHealth(nextHealth); setSchema(nextSchema); setExperiments(nextExperiments);
      })
      .catch((e) => { if (active) setError(e.message); });
    const onHash = () => setPage(location.hash.slice(1) || 'overview');
    window.addEventListener('hashchange', onHash);
    return () => { active = false; window.removeEventListener('hashchange', onHash); };
  }, []);

  const metrics = useMemo(() => {
    const metadata = experiments?.ml?.metadata;
    const modelRows = experiments?.ml?.models || experiments?.ml?.metrics || {};
    const mlp = Array.isArray(modelRows)
      ? modelRows.find((row) => String(row.model || '').toLowerCase().includes('mlp'))?.metrics || modelRows.find((row) => String(row.model || '').toLowerCase().includes('mlp'))
      : modelRows.MLP || modelRows.mlp || modelRows;
    const retrieval = experiments?.retrieval;
    return { metadata, mlp: mlp || {}, retrieval };
  }, [experiments]);

  const pageProps = { schema, health, experiments, metrics, go, setError, error, refreshExperiments };

  return (
    <div className="app-shell">
      {mobileNav && <button className="mobile-scrim" aria-label="Đóng menu" onClick={() => setMobileNav(false)} />}
      <aside className={cx('sidebar', mobileNav && 'sidebar-open')}>
        <div className="brand-row">
          <div className="brand-icon"><HeartPulse size={21} strokeWidth={2.3} /></div>
          <div className="brand-copy"><b>Diabetes<span>Lab</span></b><small>RESEARCH WORKSPACE</small></div>
          <button className="mobile-close" onClick={() => setMobileNav(false)} aria-label="Đóng menu"><X size={18} /></button>
        </div>
        <div className="workspace-switcher">
          <div className="workspace-avatar">DL</div>
          <div><b>Diabetes Agent Lab</b><small>Workspace cá nhân</small></div>
          <ChevronDown size={15} />
        </div>
        <div className="nav-caption">KHÔNG GIAN LÀM VIỆC</div>
        <nav className="side-nav" aria-label="Điều hướng chính">
          {NAV.map(({ id, label, icon: Icon, badge }) => (
            <button key={id} className={cx('nav-item', page === id && 'active')} onClick={() => go(id)}>
              <Icon size={18} strokeWidth={1.8} /><span>{label}</span>{badge && <small className="nav-badge">{badge}</small>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-status"><span className={cx('status-dot', health && 'online')} /><span>{health ? 'API đang hoạt động' : 'Đang kết nối API'}</span><MoreHorizontal size={17} /></div>
          <div className="profile-row"><div className="profile-avatar">N</div><div><b>Researcher</b><small>Local demo</small></div><Settings2 size={17} /></div>
          <p className="sidebar-note">Demo nghiên cứu · Chỉ dùng hồ sơ giả định</p>
        </div>
      </aside>

      <main className="main-shell">
        <header className="topbar">
          <div className="topbar-left">
            <button className="mobile-menu" onClick={() => setMobileNav(true)} aria-label="Mở menu"><span /></button>
            <div className="breadcrumbs"><span>Workspace</span><ChevronRight size={14} /><b>{TITLES[page] || 'Tổng quan'}</b></div>
          </div>
          <div className="topbar-actions">
            <div className="api-indicator"><span className={cx('status-dot', health && 'online')} />{health ? 'Local API connected' : 'Connecting...'}</div>
            <div className="top-menu-wrap">
              <IconButton title="Tùy chọn" onClick={() => setMenuOpen((open) => !open)}><MoreHorizontal size={19} /></IconButton>
              {menuOpen && <div className="top-menu"><button onClick={() => { setMenuOpen(false); go('workflow'); }}>Lịch sử và trace</button><button onClick={() => { setMenuOpen(false); go('experiments'); }}>Artifact thực nghiệm</button></div>}
            </div>
            <div className="top-avatar">N</div>
          </div>
        </header>

        {error && <div className="global-error"><CircleHelp size={17} /><span>{error}</span><button onClick={() => setError('')} aria-label="Đóng lỗi"><X size={16} /></button></div>}
        {!health && !error && <div className="loading-strip"><LoaderCircle size={16} className="spin" /> Đang tải dữ liệu dự án…</div>}

        <div className="content-area">
          {page === 'overview' && <OverviewPage {...pageProps} />}
          {page === 'prediction' && <PredictionPage {...pageProps} />}
          {page === 'knowledge' && <KnowledgePage {...pageProps} />}
          {page === 'llm-agent' && <LLMAgentPage {...pageProps} />}
          {page === 'workflow' && <WorkflowPage {...pageProps} />}
          {page === 'experiments' && <ExperimentsPage {...pageProps} />}
        </div>
      </main>
    </div>
  );
}

function PageHeading({ eyebrow, title, description, action }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1>{description && <p>{description}</p>}</div>{action}</div>;
}

function OverviewPage({ health, metrics, go }) {
  const overviewStats = [
    { label: 'DỮ LIỆU NGHIÊN CỨU', value: metrics.metadata?.dataset?.unique_rows ?? '100K', detail: 'mẫu sau xử lý', icon: Database, tone: 'mint' },
    { label: 'MÔ HÌNH DỰ ĐOÁN', value: 'MLP', detail: health?.prediction_ready ? 'đã sẵn sàng' : 'chưa có artifact', icon: BrainCircuit, tone: 'violet' },
    { label: 'KHO TRI THỨC', value: health?.knowledge_chunks ?? '—', detail: 'đoạn có thể truy xuất', icon: BookOpen, tone: 'blue' },
    { label: 'MÔI TRƯỜNG', value: health?.orchestrator?.framework || 'LangGraph', detail: `FastAPI · ${health?.database?.backend === 'mysql' ? 'MySQL' : 'SQLite dev'}`, icon: ShieldCheck, tone: 'amber' },
  ];
  return <>
    <section className="hero-panel">
      <div className="hero-copy"><div className="hero-kicker"><span className="hero-kicker-dot" /> RESEARCH WORKSPACE <span className="hero-kicker-line" /> AGENT SYSTEM</div>
        <h1>Hiểu dữ liệu.<br /><span>Theo dõi từng agent.</span></h1>
        <p>Khám phá bộ dữ liệu, mô hình nghiên cứu và cách các agent phối hợp trong một không gian trực quan.</p>
        <div className="hero-actions"><button className="button button-light" onClick={() => go('prediction')}>Bắt đầu demo <ArrowRight size={16} /></button><button className="button button-ghost" onClick={() => go('workflow')}>Xem workflow <Workflow size={16} /></button></div>
      </div>
      <div className="hero-art" aria-hidden="true"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="orbit orbit-three" /><div className="hero-core"><HeartPulse size={35} /></div><div className="float-node node-a"><BrainCircuit size={17} /></div><div className="float-node node-b"><BookOpen size={16} /></div><div className="float-node node-c"><Activity size={17} /></div><span className="hero-spark spark-a" /><span className="hero-spark spark-b" /></div>
      <div className="hero-index">01 <span>/ 06</span></div>
    </section>

    <div className="section-title-row"><div><div className="eyebrow">TỔNG QUAN DỰ ÁN</div><h2>Workspace snapshot</h2></div><button className="quiet-button" onClick={() => go('experiments')}>Mở báo cáo <ArrowUpRight size={15} /></button></div>
    <div className="stats-grid">{overviewStats.map(({ label, value, detail, icon: Icon, tone }) => <article className="stat-card" key={label}><div className="stat-top"><span>{label}</span><div className={`stat-icon ${tone}`}><Icon size={18} /></div></div><div className="stat-value">{typeof value === 'number' ? value.toLocaleString('vi-VN') : value}</div><div className="stat-detail">{detail}</div></article>)}</div>

    <div className="overview-grid">
      <article className="panel model-panel"><div className="panel-heading"><div><div className="eyebrow">MODEL PERFORMANCE</div><h2>Hiệu năng mô hình</h2></div><button className="square-action" onClick={() => go('experiments')} aria-label="Mở thực nghiệm"><ArrowUpRight size={17} /></button></div>
        <div className="performance-main"><div className="performance-score">{pct(metrics.mlp?.accuracy)}<small>accuracy</small></div><div className="performance-note"><span className="trend-pill"><Activity size={13} /> TEST SET</span><p>Đọc Accuracy cùng Recall và F1 để hiểu rõ hơn kết quả phân loại.</p></div></div>
        <div className="metric-bars">{[['Recall', metrics.mlp?.recall], ['F1 score', metrics.mlp?.f1], ['ROC-AUC', metrics.mlp?.roc_auc]].map(([label, value]) => <div className="metric-bar-row" key={label}><div><span>{label}</span><b>{pct(value)}</b></div><div className="bar-track"><i style={{ width: `${Math.max(0, Math.min(100, Number(value || 0) * 100))}%` }} /></div></div>)}</div>
        <button className="panel-link" onClick={() => go('experiments')}>Xem toàn bộ kết quả <ArrowRight size={15} /></button>
      </article>
      <article className="panel pipeline-panel"><div className="panel-heading"><div><div className="eyebrow">AGENT PIPELINE</div><h2>Các thành phần chính</h2></div><span className="live-tag"><i /> READY</span></div>
        <div className="pipeline-list">{[
          { name: 'Validation Agent', text: 'Kiểm tra schema và miền dữ liệu', icon: Check, n: '01' },
          { name: 'Safety Agent', text: 'Phân luồng cảnh báo độc lập', icon: ShieldCheck, n: '02' },
          { name: 'Prediction Agent', text: 'MLP phân loại nhãn trong dataset', icon: BrainCircuit, n: '03' },
          { name: 'Knowledge Agent', text: 'GraphRAG có dẫn nguồn khi cần', icon: BookOpen, n: '04' },
          { name: 'LLM Tool Agent', text: 'Tự lập kế hoạch và chọn công cụ', icon: Sparkles, n: '05', accent: true },
        ].map(({ name, text, icon: Icon, n, accent }, index, list) => <div className={cx('pipeline-item', accent && 'pipeline-accent')} key={name}><div className="pipeline-num">{n}</div><div className="pipeline-icon"><Icon size={17} /></div><div className="pipeline-text"><b>{name}</b><small>{text}</small></div>{index < list.length - 1 && <div className="pipeline-connector" />}</div>)}</div>
        <button className="panel-link" onClick={() => go('llm-agent')}>Thử LLM Agent <ArrowRight size={15} /></button>
      </article>
    </div>
    <div className="notice-bar"><div className="notice-icon"><ShieldCheck size={17} /></div><p><b>Phạm vi nghiên cứu</b><span>Điểm mô hình là nhãn của dataset Kaggle, không phải chẩn đoán hay nguy cơ bệnh trong tương lai. Chỉ dùng hồ sơ giả định.</span></p><button onClick={() => go('knowledge')}>Đọc thêm <ArrowRight size={14} /></button></div>
  </>;
}

function PredictionPage({ schema, health }) {
  const [values, setValues] = useState({});
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState('');
  const fields = schema?.features || [];
  useEffect(() => { if (schema?.demo_patient) setValues(schema.demo_patient); }, [schema]);
  const setField = (name, value, type) => setValues((old) => ({ ...old, [name]: type === 'number' ? (value === '' ? '' : Number(value)) : type === 'boolean' ? value === 'true' : value }));
  const submit = async (event) => {
    event.preventDefault(); setBusy(true); setLocalError('');
    try { setResult(await api('/api/predict', { method: 'POST', body: JSON.stringify(values) })); }
    catch (e) { setLocalError(e.message); }
    finally { setBusy(false); }
  };
  return <>
    <PageHeading eyebrow="RESEARCH PREDICTION" title="Phân loại hồ sơ nghiên cứu" description="Nhập 8 trường để xem orchestrator tự chọn Validation, Safety, Prediction và GraphRAG. Sử dụng hồ sơ giả định." action={<span className="soft-tag"><span className={cx('status-dot', health?.prediction_ready && 'online')} />{health?.prediction_ready ? 'Model sẵn sàng' : 'Thiếu model artifact'}</span>} />
    {localError && <div className="inline-error"><CircleHelp size={16} />{localError}</div>}
    <div className="predict-layout"><form className="panel form-panel" onSubmit={submit}><div className="panel-heading"><div><div className="eyebrow">INPUT PROFILE</div><h2>Thông tin giả định</h2></div><button className="quiet-button" type="button" onClick={() => setValues(schema?.demo_patient || {})}><RefreshCw size={14} /> Điền mẫu</button></div>
      <div className="form-grid">{fields.map((field) => <label className="form-field" key={field.name}><span>{field.label}</span>{field.type === 'category' ? <select required value={values[field.name] ?? ''} onChange={(e) => setField(field.name, e.target.value, field.type)}><option value="" disabled>Chọn thông tin</option>{field.options?.map((option) => <option key={option} value={option}>{option}</option>)}</select> : field.type === 'boolean' ? <select required value={values[field.name] === true ? 'true' : values[field.name] === false ? 'false' : ''} onChange={(e) => setField(field.name, e.target.value, field.type)}><option value="" disabled>Chọn thông tin</option><option value="true">Có</option><option value="false">Không</option></select> : <input required type="number" min={field.min} max={field.max} step={field.step === 'any' ? 'any' : field.step} value={values[field.name] ?? ''} onChange={(e) => setField(field.name, e.target.value, field.type)} />}</label>)}</div>
      <div className="form-disclaimer"><ShieldCheck size={15} /><span>Không nhập tên, địa chỉ hoặc thông tin định danh. Mọi kết quả chỉ phục vụ demo nghiên cứu.</span></div>
      <button className="button button-primary full-button" disabled={busy}>{busy ? <><LoaderCircle size={16} className="spin" /> Đang chạy agent…</> : <>Chạy phân loại <ArrowRight size={16} /></>}</button>
    </form>
    <div className="result-column">{result ? <PredictionResult result={result} /> : <article className="panel result-empty"><div className="empty-orbit"><BrainCircuit size={27} /></div><span className="eyebrow">MODEL OUTPUT</span><h2>Kết quả sẽ xuất hiện tại đây</h2><p>Hồ sơ được kiểm tra bằng luật trước khi gửi tới mô hình. Trace chi tiết nằm trong trang Workflow.</p><div className="result-empty-foot"><span><ShieldCheck size={14} /> Validation trước inference</span><span><Clock3 size={14} /> Thường chỉ vài giây</span></div></article>}
      <div className="notice-bar compact"><div className="notice-icon"><CircleHelp size={16} /></div><p><b>Giới hạn mô hình</b><span>Điểm sigmoid không được hiệu chuẩn thành nguy cơ cá nhân.</span></p></div></div></div>
  </>;
}

function PredictionResult({ result }) {
  const prediction = result.prediction;
  return <article className="panel prediction-result"><div className="panel-heading"><div><div className="eyebrow">MODEL OUTPUT</div><h2>{prediction ? 'Kết quả phân loại' : 'Ngoài miền dữ liệu'}</h2></div><span className={cx('result-status', prediction ? 'positive' : 'muted-status')}>{result.status === 'completed' ? 'COMPLETED' : 'OUT OF DOMAIN'}</span></div>
    {prediction ? <><div className="big-score">{pct(prediction.positive_class_score)}<small>điểm cho nhãn diabetes = 1</small></div><div className="score-track"><i style={{ width: `${prediction.positive_class_score * 100}%` }} /></div><p className="body-copy">{result.explanation.model_explanation}</p><div className="mini-metrics"><div><small>Logistic baseline</small><b>{pct(prediction.baseline_positive_class_score)}</b></div><div><small>Ngưỡng kỹ thuật</small><b>{pretty(prediction.threshold)}</b></div></div></> : <p className="body-copy">{result.explanation.model_explanation}</p>}
    <div className="result-section"><h3><ShieldCheck size={16} /> Kiểm tra đầu vào</h3>{result.rules.warnings.map((warning) => <p className="warning-line" key={warning}>{warning}</p>)}</div>
    <div className="result-section"><h3><HeartPulse size={16} /> Safety Agent · {result.safety?.level}</h3>{result.safety?.messages?.map((message) => <p className="warning-line" key={message}>{message}</p>)}</div>
    {result.explanation.general_information?.length > 0 && <div className="result-section"><h3><BookOpen size={16} /> Thông tin có nguồn</h3><EvidenceList items={result.explanation.general_information} /></div>}
    <div className="clinical-caveat">{result.explanation.limitation}</div>
  </article>;
}

function EvidenceList({ items = [] }) {
  if (!items.length) return <p className="muted-copy">Chưa có đoạn tài liệu phù hợp.</p>;
  return <div className="evidence-list">{items.map((item, index) => <article className="evidence-card" key={item.chunk_id || index}><div className="evidence-meta"><span>{item.publisher} · {item.chunk_id}</span><span>#{index + 1}</span></div><h3>{item.title}</h3><p>{item.text}</p><a href={item.url} target="_blank" rel="noreferrer">Mở nguồn gốc <ExternalLink size={13} /></a></article>)}</div>;
}

function KnowledgePage({ health }) {
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState('');
  const submit = async (event) => {
    event.preventDefault(); setBusy(true); setLocalError('');
    try { setResult(await api('/api/ask', { method: 'POST', body: JSON.stringify({ question }) })); }
    catch (e) { setLocalError(e.message); }
    finally { setBusy(false); }
  };
  return <>
    <PageHeading eyebrow="CURATED KNOWLEDGE BASE" title="Hỏi đáp từ kho tri thức" description="Kết hợp semantic retrieval, đồ thị Neo4j và nguồn tham khảo; tự fallback về TF-IDF khi graph chưa sẵn sàng." action={<span className="soft-tag"><BookOpen size={14} /> {health?.qa_mode || 'Retrieval'}</span>} />
    {localError && <div className="inline-error"><CircleHelp size={16} />{localError}</div>}
    <div className="knowledge-layout"><article className="panel ask-panel"><div className="panel-heading"><div><div className="eyebrow">ASK THE KNOWLEDGE BASE</div><h2>Bạn muốn tìm hiểu điều gì?</h2></div><div className="ask-mark"><BookOpen size={20} /></div></div>
      <form onSubmit={submit}><textarea className="knowledge-input" value={question} onChange={(e) => setQuestion(e.target.value)} rows={4} maxLength={1000} minLength={3} placeholder="Ví dụ: HbA1c phản ánh đường huyết trong khoảng thời gian nào?" required /><div className="question-chips">{SAMPLE_QUESTIONS.map((sample) => <button type="button" className="question-chip" key={sample} onClick={() => setQuestion(sample)}>{sample}<ArrowUpRight size={12} /></button>)}</div><button className="button button-primary" disabled={busy}>{busy ? <><LoaderCircle size={16} className="spin" /> Đang truy xuất…</> : <>Tìm trong kho tri thức <ArrowRight size={16} /></>}</button></form>
      <div className="knowledge-foot"><ShieldCheck size={14} /> Nội dung được truy xuất từ các tài liệu biên soạn có dẫn nguồn.</div>
    </article>
      <aside className="knowledge-aside"><div className="aside-note"><span className="aside-note-icon"><Sparkles size={17} /></span><div><b>Grounded answers</b><p>Các đoạn nguồn được hiển thị riêng để bạn tự kiểm tra ngữ cảnh.</p></div></div><div className="aside-stat"><span className="eyebrow">CURRENT INDEX</span><div className="aside-stat-value">5 <small>tài liệu</small></div><p>WHO · CDC · Tài liệu sức khỏe cộng đồng</p><div className="aside-stat-bottom"><span><span className="status-dot online" /> Local TF-IDF index</span><ChevronRight size={15} /></div></div></aside>
    </div>
    {result && <article className="panel knowledge-result"><div className="panel-heading"><div><div className="eyebrow">RETRIEVAL RESULT</div><h2>{result.status === 'answered' ? 'Đã tìm thấy thông tin' : 'Chưa có đủ bằng chứng'}</h2></div><span className="soft-tag">{result.mode}</span></div><div className="answer-box"><MessageCircle size={16} /><p>{result.answer}</p></div><EvidenceList items={result.evidence} /></article>}
  </>;
}

function LLMAgentPage() {
  const [status, setStatus] = useState(null);
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState('');
  useEffect(() => { api('/api/agent/status').then(setStatus).catch((e) => setLocalError(e.message)); }, []);
  const submit = async (event) => {
    event?.preventDefault();
    const message = text.trim(); if (!message || busy) return;
    const prior = messages.filter((m) => m.role === 'user' || m.role === 'assistant').slice(-8).map(({ role, content }) => ({ role, content }));
    setMessages((old) => [...old, { role: 'user', content: message }]); setText(''); setBusy(true); setLocalError('');
    try {
      const response = await api('/api/agent/chat', { method: 'POST', body: JSON.stringify({ message, history: prior }) });
      setMessages((old) => [...old, { role: 'assistant', content: response.answer, calls: response.tool_calls || [], model: response.model }]);
    } catch (e) { setLocalError(e.message); setMessages((old) => [...old, { role: 'error', content: e.message }]); }
    finally { setBusy(false); }
  };
  const configured = status?.configured;
  return <>
    <PageHeading eyebrow="MODEL-PLANNED TOOLS" title="LLM Agent workspace" description="Đặt câu hỏi tự nhiên. Model sẽ tự quyết định có cần tìm kiến thức, chạy mô hình nghiên cứu hay đọc kết quả dự án." action={<span className={cx('soft-tag', configured ? 'soft-green' : 'soft-amber')}><span className={cx('status-dot', configured && 'online')} />{configured ? 'OpenRouter connected' : 'Cần cấu hình API'}</span>} />
    <div className="agent-layout"><section className="panel chat-panel"><div className="chat-header"><div className="assistant-avatar"><Sparkles size={18} /></div><div><b>Diabetes Research Assistant</b><small>{configured ? `${status.model} · ${status.tools?.length || 3} tools available` : 'Local tool-calling agent'}</small></div><span className="live-tag"><i /> READY</span></div>
      <div className="chat-scroll"><div className="chat-intro"><span className="chat-orb"><Sparkles size={22} /></span><h2>Xin chào, mình có thể giúp gì?</h2><p>Mình có thể tra cứu tài liệu, xem kết quả thực nghiệm hoặc chạy mô hình trên hồ sơ giả định.</p><div className="prompt-grid">{DEMO_PROMPTS.map((prompt, i) => <button key={prompt} className="prompt-card" onClick={() => setText(prompt)}><span>{i === 0 ? <BookOpen size={15} /> : i === 1 ? <FlaskConical size={15} /> : <BrainCircuit size={15} />}</span>{prompt}<ArrowUpRight size={14} /></button>)}</div></div>
        {messages.map((message, index) => <div key={index} className={cx('chat-message-row', message.role)}><div className={cx('chat-bubble', message.role)}>{message.role !== 'user' && <div className="bubble-label"><Sparkles size={13} /> {message.role === 'error' ? 'Kết nối gặp lỗi' : 'Research Assistant'}</div>}<p>{message.content}</p>{message.calls?.length > 0 && <div className="tool-trace"><div className="trace-caption"><Workflow size={14} /> TOOL CALLS · {message.calls.length}</div>{message.calls.map((call, callIndex) => <details className="tool-call-card" key={`${call.name}-${callIndex}`}><summary><span className="tool-success"><Check size={12} /></span><b>{call.name}</b><span className="trace-state">{call.status}</span><ChevronDown size={14} /></summary><div className="tool-call-content"><small>Tham số</small><pre>{JSON.stringify(call.arguments, null, 2)}</pre><small>Kết quả tóm tắt</small><pre>{JSON.stringify(call.summary, null, 2)}</pre></div></details>)}</div>}</div></div>)}
        {busy && <div className="chat-message-row assistant"><div className="chat-bubble assistant"><div className="typing-dots"><i /><i /><i /><span>Đang suy nghĩ và chọn công cụ…</span></div></div></div>}
      </div>
      <form className="chat-composer" onSubmit={submit}><textarea value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(e); } }} rows={2} maxLength={2000} placeholder="Nhắn tin cho agent…" /><div className="composer-bottom"><span><ShieldCheck size={13} /> Không nhập dữ liệu định danh</span><button className="send-button" disabled={busy || !text.trim()} aria-label="Gửi"><Send size={17} /></button></div></form>
      <div className="chat-footnote">AI có thể trả lời chưa chính xác. Kiểm tra nguồn và giới hạn của từng kết quả.</div>
    </section>
      <aside className="agent-side"><article className="panel agent-status-panel"><div className="eyebrow">AGENT CONFIGURATION</div><h3>Runtime status</h3><div className="runtime-row"><span>Framework</span><b>{status?.framework || 'LangChain'}</b></div><div className="runtime-row"><span>Integration</span><b>{status?.integration || 'ChatOpenRouter'}</b></div><div className="runtime-row"><span>Provider</span><b>{status?.provider || 'OpenRouter'}</b></div><div className="runtime-row"><span>Model</span><b className="model-name">{status?.model || 'Chưa cấu hình'}</b></div><div className="runtime-row"><span>API key</span><b className={configured ? 'text-green' : 'text-amber'}>{configured ? 'Đã nạp ở backend' : 'Chưa phát hiện'}</b></div><div className="tool-list-title">TOOLS AVAILABLE</div>{(status?.tools || ['search_health_knowledge', 'run_research_prediction', 'get_experiment_summary']).map((tool) => <div className="tool-list-item" key={tool}><span><Check size={12} /></span>{tool}</div>)}{!configured && <div className="config-hint">Đặt <code>OPENROUTER_API_KEY</code> trong file `.env` ở thư mục dự án rồi khởi động lại server.</div>}</article><article className="agent-safety-card"><ShieldCheck size={17} /><div><b>Tool access is limited</b><p>Agent chỉ gọi ba hàm nội bộ được khai báo. Không có quyền chạy lệnh hay truy cập file tùy ý.</p></div></article><div className="agent-limit-note">Điểm dự đoán thuộc nhãn dataset nghiên cứu; không thay thế đánh giá của nhân viên y tế.</div></aside>
    </div>
    {localError && <div className="inline-error agent-error"><CircleHelp size={16} />{localError}</div>}
  </>;
}

function WorkflowPage({ setError }) {
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState(null);
  const [busy, setBusy] = useState(true);
  const load = useCallback(async () => { setBusy(true); try { const next = await api('/api/history'); setRuns(next); setSelected((old) => next.find((run) => run.id === old)?.id || next[0]?.id || null); } catch (e) { setError(e.message); } finally { setBusy(false); } }, [setError]);
  useEffect(() => { load(); }, [load]);
  const clear = async () => { if (!window.confirm('Xóa toàn bộ lịch sử demo trên máy này?')) return; try { await api('/api/history', { method: 'DELETE' }); setSelected(null); await load(); } catch (e) { setError(e.message); } };
  const current = runs.find((run) => run.id === selected);
  return <>
    <PageHeading eyebrow="OBSERVABILITY" title="Workflow & trace" description="Theo dõi dữ liệu đầu vào, thứ tự gọi agent và output thực tế được lưu trên máy." action={<button className="button button-outline" onClick={load}><RefreshCw size={15} /> Làm mới</button>} />
    <div className="workflow-layout"><aside className="panel run-panel"><div className="panel-heading"><div><div className="eyebrow">LOCAL HISTORY</div><h2>Lịch sử gần đây</h2></div><span className="count-badge">{runs.length}</span></div>{busy ? <div className="loading-inline"><LoaderCircle className="spin" size={16} /> Đang tải…</div> : runs.length ? <div className="run-list">{runs.map((run) => <button key={run.id} className={cx('run-item', selected === run.id && 'selected')} onClick={() => setSelected(run.id)}><span className="run-icon">{run.kind === 'prediction' ? <BrainCircuit size={16} /> : <MessageCircle size={16} />}</span><span className="run-main"><b>{run.kind === 'prediction' ? 'Phân loại hồ sơ' : run.payload?.question || 'Hỏi đáp tri thức'}</b><small><Clock3 size={11} /> {dateTime(run.created_at)}</small></span><ChevronRight size={15} /></button>)}</div> : <div className="small-empty"><History size={19} /><p>Chưa có workflow nào.<br />Hãy chạy dự đoán hoặc hỏi đáp.</p></div>}<button className="clear-history" onClick={clear} disabled={!runs.length}><Trash2 size={14} /> Xóa lịch sử local</button></aside>
      <article className="panel trace-panel">{current ? <><div className="panel-heading"><div><div className="eyebrow">RUN {current.id.slice(0, 8).toUpperCase()}</div><h2>{current.kind === 'prediction' ? 'Workflow phân loại' : 'Workflow hỏi đáp'}</h2></div><span className="soft-tag"><Check size={13} /> {current.payload?.trace?.length || 0} agent calls</span></div><div className="trace-timeline">{(current.payload?.trace || []).map((step, index) => <div className="trace-step" key={`${step.agent}-${index}`}><div className="trace-step-mark">{step.order ?? index + 1}</div><div className="trace-step-body"><div className="trace-step-head"><div><b>{step.agent}</b><span className={cx('step-state', step.status === 'error' && 'step-error')}>{step.status}</span></div><small>{step.duration_ms ?? 0} ms</small></div><div className="trace-details"><details><summary>Input <ChevronDown size={13} /></summary><pre>{JSON.stringify(step.input, null, 2)}</pre></details><details><summary>Output <ChevronDown size={13} /></summary><pre>{JSON.stringify(step.output, null, 2)}</pre></details></div></div></div>)}</div></> : <div className="trace-empty"><div className="empty-orbit"><Workflow size={25} /></div><h2>Chọn một lần chạy</h2><p>Trace của từng agent sẽ hiển thị ở đây.</p></div>}</article></div>
  </>;
}

function ExperimentsPage({ experiments, metrics, refreshExperiments }) {
  const ml = experiments?.ml;
  const metadata = ml?.metadata;
  const rows = Array.isArray(ml?.models) ? ml.models : Object.entries(ml?.metrics || {}).map(([model, metric]) => ({ model, ...metric }));
  const formatMetric = (row, key) => row?.[key] == null ? '—' : pct(row[key]);
  const artifact = (category, file) => `/api/artifact/${category}/${file}`;
  return <>
    <PageHeading eyebrow="REPRODUCIBLE EVALUATION" title="Thực nghiệm & kết quả" description="Các chỉ số, biểu đồ và artifact được đọc trực tiếp từ những lần chạy thực nghiệm của dự án." action={<button className="button button-outline" onClick={refreshExperiments}><RefreshCw size={15} /> Tải lại dữ liệu</button>} />
    {!experiments && <div className="panel loading-panel"><LoaderCircle className="spin" size={18} /> Đang đọc artifact…</div>}
    {ml ? <article className="panel experiment-panel"><div className="panel-heading"><div><div className="eyebrow">01 · DEEP LEARNING + BASELINE</div><h2>Phân loại trên tập test giữ riêng</h2></div><span className="soft-tag">Seed {metadata?.seed ?? '—'}</span></div><div className="experiment-summary"><div><b>{pretty(metadata?.dataset?.raw_rows)}</b><small>Dòng Kaggle gốc</small></div><div><b>{pretty(metadata?.dataset?.unique_rows)}</b><small>Dòng sau xử lý</small></div><div><b>{metadata?.splits ? `${metadata.splits.train.rows} / ${metadata.splits.validation.rows} / ${metadata.splits.test.rows}` : '—'}</b><small>Train / validation / test</small></div></div>
      {rows.length > 0 && <div className="table-wrap"><table><thead><tr><th>Mô hình</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr></thead><tbody>{rows.map((row, i) => <tr key={row.model || i}><td><b>{row.model || 'Model'}</b></td>{['accuracy', 'precision', 'recall', 'f1', 'roc_auc'].map((key) => <td key={key}>{formatMetric(row.metrics || row, key)}</td>)}</tr>)}</tbody></table></div>}
      <div className="chart-grid"><Figure src={artifact('ml', 'training_loss.png')} title="Training & validation loss" /><Figure src={artifact('ml', 'roc_curves.png')} title="ROC curves · test set" /><Figure src={artifact('ml', 'confusion_matrices.png')} title="Confusion matrices" wide /></div><div className="artifact-links"><a href={artifact('ml', 'evaluation.json')}>evaluation.json <ExternalLink size={13} /></a><a href={artifact('ml', 'model_comparison.csv')}>model_comparison.csv <ExternalLink size={13} /></a><a href={artifact('ml', 'test_predictions.csv')}>test_predictions.csv <ExternalLink size={13} /></a></div><p className="clinical-caveat">Dataset chưa mô tả đầy đủ quy trình thu thập và tạo nhãn. Accuracy cao không tự chứng minh năng lực sàng lọc; cần đọc Recall, F1 và giới hạn nghiên cứu cùng nhau.</p></article> : <article className="panel small-empty"><FlaskConical size={22} /><p>Chưa có artifact đánh giá mô hình. Chạy <code>python -m diabetes_demo.train</code> để tạo kết quả.</p></article>}
    {experiments?.rl_manifest && <article className="panel experiment-panel"><div className="panel-heading"><div><div className="eyebrow">02 · Q-LEARNING · MODULE RIÊNG</div><h2>Mô phỏng cạnh tranh thương mại điện tử</h2></div><a className="quiet-button" href="/legacy-demo" target="_blank" rel="noreferrer">Mở demo cũ <ExternalLink size={14} /></a></div><p className="body-copy">Q-learning được đánh giá trong môi trường thương mại điện tử độc lập; module này không tham gia workflow tiểu đường.</p><div className="artifact-links"><a href={artifact('rl', 'baseline_evaluation.csv')}>baseline_evaluation.csv <ExternalLink size={13} /></a><a href={artifact('rl', 'multi_seed_summary.csv')}>multi_seed_summary.csv <ExternalLink size={13} /></a></div></article>}
    {experiments?.retrieval && <article className="panel experiment-panel"><div className="eyebrow">03 · KNOWLEDGE RETRIEVAL</div><h2>Đánh giá truy xuất</h2><div className="retrieval-metrics"><div><b>{pct(experiments.retrieval.hit_at_1)}</b><small>Hit@1</small></div><div><b>{pct(experiments.retrieval.hit_at_3)}</b><small>Hit@3</small></div><div><b>{pct(experiments.retrieval.rejection_accuracy)}</b><small>Rejection accuracy</small></div></div></article>}
  </>;
}

function Figure({ src, title, wide }) {
  const [failed, setFailed] = useState(false);
  return <figure className={cx('figure-card', wide && 'figure-wide')}>{failed ? <div className="figure-missing">Chưa có biểu đồ này trong artifacts.</div> : <img src={src} alt={title} onError={() => setFailed(true)} loading="lazy" />}<figcaption>{title}</figcaption></figure>;
}

export default App;
