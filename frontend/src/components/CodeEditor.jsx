import { useMemo, useRef, useState } from 'react';

const KEYWORDS = new Set([
  'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
  'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally',
  'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal',
  'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield',
]);

const BUILTINS = new Set([
  'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter', 'float', 'int',
  'isinstance', 'len', 'list', 'map', 'max', 'min', 'pow', 'print', 'range',
  'reversed', 'round', 'set', 'slice', 'sorted', 'str', 'sum', 'tuple', 'zip',
  'pd', 'np', 'math', 'statistics', 'ParamSpec', 'indicator', 'strategy',
]);

const TOKEN_RE = /(#.*$)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(@[A-Za-z_][A-Za-z0-9_.]*)|(\b\d+(?:\.\d+)?\b)|(\b[A-Za-z_][A-Za-z0-9_]*\b)/gm;

function tokenizeLine(line) {
  const parts = [];
  let last = 0;
  let expectFn = false;
  TOKEN_RE.lastIndex = 0;
  let m;
  while ((m = TOKEN_RE.exec(line)) !== null) {
    if (m.index > last) {
      parts.push({ text: line.slice(last, m.index), cls: null });
    }
    const [full, comment, str, dec, num, ident] = m;
    if (comment) parts.push({ text: full, cls: 'tok-com' });
    else if (str) parts.push({ text: full, cls: 'tok-str' });
    else if (dec) parts.push({ text: full, cls: 'tok-dec' });
    else if (num) parts.push({ text: full, cls: 'tok-num' });
    else if (ident) {
      if (expectFn) {
        parts.push({ text: full, cls: 'tok-fn' });
        expectFn = false;
      } else if (KEYWORDS.has(ident)) {
        parts.push({ text: full, cls: 'tok-kw' });
        if (ident === 'def' || ident === 'class') expectFn = true;
      } else if (BUILTINS.has(ident)) {
        parts.push({ text: full, cls: 'tok-bi' });
      } else {
        parts.push({ text: full, cls: null });
      }
    }
    last = m.index + full.length;
  }
  if (last < line.length) {
    parts.push({ text: line.slice(last), cls: null });
  }
  return parts;
}

export default function CodeEditor({
  value,
  onChange,
  filename = 'custom_module.py',
  errorLine = null,
  dirty = false,
  height = 380,
}) {
  const gutterRef = useRef(null);
  const highlightRef = useRef(null);
  const textareaRef = useRef(null);
  const [cursor, setCursor] = useState({ line: 1, col: 1 });

  const lines = useMemo(() => String(value || '').split('\n'), [value]);
  const highlighted = useMemo(() => lines.map((line) => tokenizeLine(line)), [lines]);

  const syncScroll = (e) => {
    const { scrollTop, scrollLeft } = e.target;
    if (gutterRef.current) gutterRef.current.scrollTop = scrollTop;
    if (highlightRef.current) {
      highlightRef.current.style.transform = `translate(${-scrollLeft}px, ${-scrollTop}px)`;
    }
  };

  const updateCursor = (el) => {
    const pos = el.selectionStart || 0;
    const before = el.value.slice(0, pos);
    const split = before.split('\n');
    setCursor({ line: split.length, col: split[split.length - 1].length + 1 });
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const el = e.target;
      const start = el.selectionStart;
      const end = el.selectionEnd;
      const next = value.slice(0, start) + '    ' + value.slice(end);
      onChange(next);
      requestAnimationFrame(() => {
        if (textareaRef.current) {
          textareaRef.current.selectionStart = textareaRef.current.selectionEnd = start + 4;
          updateCursor(textareaRef.current);
        }
      });
    }
  };

  return (
    <div className="py-editor-wrap">
      <div className="py-editor-head">
        <span className="mono">
          {filename} {dirty ? <span style={{ color: 'var(--warn)' }}>● modified</span> : ''}
        </span>
        <span>Python 3.11 · Isolated Sandbox</span>
      </div>
      <div className="py-editor-body" style={{ height }}>
        <div className="py-gutter" ref={gutterRef}>
          {lines.map((_, idx) => (
            <div key={idx} className={`py-gutter-line ${errorLine === idx + 1 ? 'err' : ''}`}>
              {idx + 1}
            </div>
          ))}
        </div>
        <div className="py-stage">
          <div className="py-highlight" ref={highlightRef} aria-hidden="true">
            {highlighted.map((tokens, idx) => (
              <div key={idx} className={`py-line ${errorLine === idx + 1 ? 'err' : ''}`}>
                {tokens.length === 0 ? '\u00A0' : tokens.map((tok, j) => (
                  tok.cls ? <span key={j} className={tok.cls}>{tok.text}</span> : <span key={j}>{tok.text}</span>
                ))}
              </div>
            ))}
          </div>
          <textarea
            ref={textareaRef}
            className="py-textarea"
            spellCheck={false}
            autoCapitalize="off"
            autoComplete="off"
            autoCorrect="off"
            value={value}
            onChange={(e) => { onChange(e.target.value); updateCursor(e.target); }}
            onScroll={syncScroll}
            onClick={(e) => updateCursor(e.target)}
            onKeyUp={(e) => updateCursor(e.target)}
            onKeyDown={handleKeyDown}
          />
        </div>
      </div>
      <div className="py-editor-foot">
        <span>Tab inserts 4 spaces · {lines.length} lines</span>
        <span className="mono">Ln {cursor.line}, Col {cursor.col}</span>
      </div>
    </div>
  );
}
