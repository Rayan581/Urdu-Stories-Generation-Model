"use client";

import { useState, useRef, useCallback, useEffect } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

interface GenerationState {
  isGenerating: boolean;
  generatedText: string;
  error: string | null;
}

export default function Home() {
  const [prefix, setPrefix] = useState("");
  const [maxLength, setMaxLength] = useState(100);
  const [temperature, setTemperature] = useState(0.7);
  const [mounted, setMounted] = useState(false);
  const [state, setState] = useState<GenerationState>({
    isGenerating: false,
    generatedText: "",
    error: null,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => setMounted(true), []);

  const generateStory = useCallback(async () => {
    if (!prefix.trim()) {
      setState(s => ({ ...s, error: "براہ کرم ایک شروعاتی جملہ درج کریں" }));
      return;
    }
    if (abortControllerRef.current) abortControllerRef.current.abort();
    abortControllerRef.current = new AbortController();
    setState({ isGenerating: true, generatedText: "", error: null });

    try {
      const response = await fetch(`${BACKEND_URL}/generate/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prefix, max_length: maxLength, temperature, top_k: 40, top_p: 0.92 }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let fullText = "";
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.token) { fullText += data.token; setState(s => ({ ...s, generatedText: fullText })); }
              if (data.done) { setState(s => ({ ...s, isGenerating: false })); return; }
            } catch { /* skip */ }
          }
        }
      }
      setState(s => ({ ...s, isGenerating: false }));
    } catch (err) {
      if ((err as Error).name === "AbortError") { setState(s => ({ ...s, isGenerating: false })); return; }
      setState(s => ({ ...s, isGenerating: false, error: `خرابی: ${(err as Error).message}` }));
    }
  }, [prefix, maxLength, temperature]);

  const stopGeneration = useCallback(() => {
    abortControllerRef.current?.abort();
    setState(s => ({ ...s, isGenerating: false }));
  }, []);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');

        :root {
          --ink: #1a1025;
          --parchment: #f5efe0;
          --gold: #c9973a;
          --gold-light: #e8c577;
          --deep: #2d1b4e;
          --crimson: #8b1a2e;
          --mist: #ede8f5;
          --border: rgba(201,151,58,0.3);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
          background: var(--ink);
          font-family: 'Cormorant Garamond', Georgia, serif;
          min-height: 100vh;
          overflow-x: hidden;
        }

        .page-bg {
          position: fixed; inset: 0; z-index: 0;
          background: radial-gradient(ellipse at 20% 20%, #2d1b4e 0%, #1a1025 40%, #0d0818 100%);
        }
        .page-bg::before {
          content: '';
          position: absolute; inset: 0;
          background-image: 
            radial-gradient(circle at 15% 85%, rgba(201,151,58,0.08) 0%, transparent 40%),
            radial-gradient(circle at 85% 10%, rgba(139,26,46,0.06) 0%, transparent 40%);
        }
        .grain {
          position: fixed; inset: 0; z-index: 1; pointer-events: none; opacity: 0.03;
          background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
        }

        .container {
          position: relative; z-index: 2;
          max-width: 860px; margin: 0 auto;
          padding: 40px 20px 60px;
        }

        /* ORNAMENTAL DIVIDER */
        .ornament {
          display: flex; align-items: center; gap: 16px;
          color: var(--gold); font-size: 18px; letter-spacing: 4px;
          margin: 0 auto 8px;
          width: fit-content;
        }
        .ornament-line {
          width: 60px; height: 1px;
          background: linear-gradient(90deg, transparent, var(--gold), transparent);
        }

        /* HEADER */
        header {
          text-align: center; margin-bottom: 48px;
          animation: fadeUp 0.8s ease both;
        }

        .header-badge {
          display: inline-block;
          background: linear-gradient(135deg, rgba(201,151,58,0.15), rgba(201,151,58,0.05));
          border: 1px solid var(--border);
          border-radius: 100px; padding: 6px 20px;
          font-size: 11px; letter-spacing: 3px; text-transform: uppercase;
          color: var(--gold-light); margin-bottom: 20px;
        }

        .main-title {
          font-family: 'Noto Nastaliq Urdu', serif;
          font-size: clamp(2.4rem, 6vw, 3.8rem);
          font-weight: 700;
          color: var(--parchment);
          line-height: 1.3;
          text-shadow: 0 0 60px rgba(201,151,58,0.3);
          margin-bottom: 6px;
        }

        .sub-title {
          font-family: 'Cormorant Garamond', serif;
          font-size: 1rem; font-style: italic; font-weight: 300;
          color: rgba(201,151,58,0.7); letter-spacing: 1px;
          margin-bottom: 20px;
        }

        .caption {
          font-family: 'Noto Nastaliq Urdu', serif;
          font-size: 1rem; color: rgba(245,239,224,0.5);
          line-height: 1.8;
        }

        /* CARD */
        .card {
          background: linear-gradient(145deg, rgba(45,27,78,0.6) 0%, rgba(26,16,37,0.8) 100%);
          border: 1px solid var(--border);
          border-radius: 2px; padding: 40px;
          backdrop-filter: blur(20px);
          position: relative; overflow: hidden;
          animation: fadeUp 0.9s ease 0.15s both;
        }

        .card::before {
          content: '';
          position: absolute; top: 0; left: 0; right: 0; height: 1px;
          background: linear-gradient(90deg, transparent, var(--gold-light), transparent);
        }
        .card::after {
          content: '';
          position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
          background: linear-gradient(90deg, transparent, var(--gold), transparent);
        }

        /* Corner decorations */
        .card-corner {
          position: absolute; width: 30px; height: 30px;
          border-color: var(--gold);
        }
        .card-corner.tl { top: 10px; left: 10px; border-top: 1px solid; border-left: 1px solid; }
        .card-corner.tr { top: 10px; right: 10px; border-top: 1px solid; border-right: 1px solid; }
        .card-corner.bl { bottom: 10px; left: 10px; border-bottom: 1px solid; border-left: 1px solid; }
        .card-corner.br { bottom: 10px; right: 10px; border-bottom: 1px solid; border-right: 1px solid; }

        /* LABEL */
        .field-label {
          display: block;
          font-family: 'Noto Nastaliq Urdu', serif;
          font-size: 1rem; color: var(--gold-light);
          margin-bottom: 12px; text-align: right;
        }

        /* TEXTAREA */
        .urdu-input {
          width: 100%; min-height: 100px;
          background: rgba(10,6,20,0.6);
          border: 1px solid rgba(201,151,58,0.25);
          border-radius: 2px;
          padding: 16px 20px;
          font-family: 'Noto Nastaliq Urdu', serif;
          font-size: 1.25rem; line-height: 1.9;
          color: var(--parchment);
          resize: none; outline: none;
          transition: border-color 0.3s, box-shadow 0.3s;
          text-align: right; direction: rtl;
        }
        .urdu-input::placeholder { color: rgba(245,239,224,0.25); }
        .urdu-input:focus {
          border-color: rgba(201,151,58,0.6);
          box-shadow: 0 0 0 3px rgba(201,151,58,0.08), inset 0 0 20px rgba(201,151,58,0.03);
        }
        .urdu-input:disabled { opacity: 0.5; cursor: not-allowed; }

        /* SLIDERS */
        .sliders-grid {
          display: grid; grid-template-columns: 1fr 1fr; gap: 28px;
          margin: 28px 0;
        }
        @media (max-width: 560px) { .sliders-grid { grid-template-columns: 1fr; } }

        .slider-group {}
        .slider-meta {
          display: flex; justify-content: space-between; align-items: baseline;
          margin-bottom: 10px;
        }
        .slider-label {
          font-family: 'Noto Nastaliq Urdu', serif;
          font-size: 0.85rem; color: rgba(201,151,58,0.7);
        }
        .slider-val {
          font-family: 'Cormorant Garamond', serif;
          font-size: 1.1rem; font-weight: 600; color: var(--gold-light);
        }

        input[type='range'] {
          -webkit-appearance: none; width: 100%; height: 2px;
          background: rgba(201,151,58,0.2);
          border-radius: 2px; outline: none; cursor: pointer;
          transition: background 0.3s;
        }
        input[type='range']:hover { background: rgba(201,151,58,0.35); }
        input[type='range']::-webkit-slider-thumb {
          -webkit-appearance: none; width: 16px; height: 16px;
          background: var(--gold); border-radius: 50%;
          box-shadow: 0 0 12px rgba(201,151,58,0.6);
          transition: transform 0.2s, box-shadow 0.2s;
        }
        input[type='range']::-webkit-slider-thumb:hover {
          transform: scale(1.3);
          box-shadow: 0 0 20px rgba(201,151,58,0.8);
        }
        input[type='range']:disabled { opacity: 0.4; cursor: not-allowed; }

        /* DIVIDER */
        .section-divider {
          display: flex; align-items: center; gap: 12px;
          margin: 28px 0; color: rgba(201,151,58,0.25);
        }
        .section-divider::before, .section-divider::after {
          content: ''; flex: 1; height: 1px;
          background: linear-gradient(90deg, transparent, rgba(201,151,58,0.2), transparent);
        }
        .section-divider span { font-size: 14px; white-space: nowrap; }

        /* BUTTON */
        .btn-generate {
          width: 100%; padding: 18px 28px;
          font-family: 'Noto Nastaliq Urdu', serif;
          font-size: 1.2rem; font-weight: 700;
          border: none; cursor: pointer;
          position: relative; overflow: hidden;
          border-radius: 2px;
          transition: transform 0.2s, box-shadow 0.3s;
        }
        .btn-generate.generate {
          background: linear-gradient(135deg, #c9973a 0%, #e8c577 50%, #c9973a 100%);
          background-size: 200% 100%;
          color: var(--ink);
          box-shadow: 0 4px 30px rgba(201,151,58,0.35);
          animation: shimmer 3s ease infinite;
        }
        .btn-generate.stop {
          background: linear-gradient(135deg, #8b1a2e, #c0293e);
          color: white;
          box-shadow: 0 4px 20px rgba(139,26,46,0.35);
        }
        .btn-generate:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 8px 40px rgba(201,151,58,0.5);
        }
        .btn-generate:active { transform: translateY(0); }
        .btn-inner { display: flex; align-items: center; justify-content: center; gap: 12px; }

        @keyframes shimmer {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }

        /* SPINNER */
        .spinner {
          width: 20px; height: 20px; border: 2px solid rgba(255,255,255,0.3);
          border-top-color: white; border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* ERROR */
        .error-box {
          margin: 20px 0; padding: 14px 18px;
          background: rgba(139,26,46,0.15); border: 1px solid rgba(139,26,46,0.4);
          border-radius: 2px; text-align: right; direction: rtl;
          font-family: 'Noto Nastaliq Urdu', serif;
          font-size: 0.95rem; color: #f87171;
        }

        /* OUTPUT */
        .output-section {
          margin-top: 36px;
          animation: fadeUp 0.9s ease 0.3s both;
        }
        .output-header {
          display: flex; align-items: center; justify-content: space-between;
          margin-bottom: 20px;
        }
        .output-label {
          font-family: 'Cormorant Garamond', serif;
          font-style: italic; font-size: 0.85rem; letter-spacing: 2px;
          text-transform: uppercase; color: rgba(201,151,58,0.5);
        }
        .output-label-urdu {
          font-family: 'Noto Nastaliq Urdu', serif;
          font-size: 1rem; color: var(--gold-light);
        }

        .output-box {
          background: rgba(10,6,20,0.7);
          border: 1px solid rgba(201,151,58,0.2);
          border-radius: 2px; padding: 30px;
          min-height: 200px; position: relative;
          transition: border-color 0.3s;
        }
        .output-box.active { border-color: rgba(201,151,58,0.4); }
        .output-box::before {
          content: '❝';
          position: absolute; top: 12px; left: 16px;
          font-size: 2rem; color: rgba(201,151,58,0.12);
          line-height: 1; pointer-events: none;
        }

        .story-text {
          font-family: 'Noto Nastaliq Urdu', serif;
          font-size: 1.3rem; line-height: 2.2;
          color: var(--parchment);
          text-align: right; direction: rtl;
          white-space: pre-wrap;
        }

        .placeholder-text {
          font-family: 'Noto Nastaliq Urdu', serif;
          font-size: 1.1rem; color: rgba(245,239,224,0.2);
          text-align: center; direction: rtl;
          padding: 40px 0;
        }

        .cursor-blink {
          display: inline-block; width: 2px; height: 1.2em;
          background: var(--gold); vertical-align: text-bottom; margin-right: 4px;
          animation: blink 1s ease-in-out infinite;
        }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

        /* FOOTER */
        footer {
          text-align: center; margin-top: 40px;
          color: rgba(201,151,58,0.3);
          font-family: 'Cormorant Garamond', serif;
          font-size: 0.85rem; letter-spacing: 1px;
          animation: fadeUp 1s ease 0.45s both;
        }

        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }

        /* WORD COUNT */
        .word-count {
          font-family: 'Cormorant Garamond', serif;
          font-size: 0.8rem; color: rgba(201,151,58,0.4);
          text-align: left; margin-top: 8px;
        }
      `}</style>

      <div className="page-bg" />
      <div className="grain" />

      {mounted && (
        <div className="container">
          {/* HEADER */}
          <header>
            <div className="header-badge">Trigram Language Model · BPE Tokenizer</div>
            <h1 className="main-title" dir="rtl">اردو کہانی جنریٹر</h1>
            <p className="sub-title">Urdu Story Generator</p>
            <div className="ornament">
              <span className="ornament-line" />
              <span>✦</span>
              <span className="ornament-line" />
            </div>
            <p className="caption" dir="rtl">
              لفظوں کی دنیا میں خوش آمدید — یہاں ہر جملہ ایک نئی کہانی کا آغاز ہے
            </p>
          </header>

          {/* MAIN CARD */}
          <div className="card">
            <div className="card-corner tl" />
            <div className="card-corner tr" />
            <div className="card-corner bl" />
            <div className="card-corner br" />

            {/* Input */}
            <div>
              <label className="field-label">شروعاتی جملہ درج کریں</label>
              <textarea
                className="urdu-input"
                dir="rtl"
                placeholder="مثال: ایک دفعہ کا ذکر ہے کہ..."
                value={prefix}
                onChange={e => setPrefix(e.target.value)}
                disabled={state.isGenerating}
              />
              {prefix && (
                <div className="word-count">{prefix.trim().split(/\s+/).length} words</div>
              )}
            </div>

            {/* Sliders */}
            <div className="sliders-grid">
              <div className="slider-group">
                <div className="slider-meta">
                  <span className="slider-val">{maxLength}</span>
                  <span className="slider-label">زیادہ سے زیادہ ٹوکنز</span>
                </div>
                <input
                  type="range" min="50" max="300" value={maxLength}
                  onChange={e => setMaxLength(Number(e.target.value))}
                  disabled={state.isGenerating}
                />
              </div>
              <div className="slider-group">
                <div className="slider-meta">
                  <span className="slider-val">{temperature.toFixed(1)}</span>
                  <span className="slider-label">تخلیقیت</span>
                </div>
                <input
                  type="range" min="0.1" max="1.5" step="0.1" value={temperature}
                  onChange={e => setTemperature(Number(e.target.value))}
                  disabled={state.isGenerating}
                />
              </div>
            </div>

            <div className="section-divider">
              <span>✦</span>
            </div>

            {/* Button */}
            <button
              className={`btn-generate ${state.isGenerating ? "stop" : "generate"}`}
              onClick={state.isGenerating ? stopGeneration : generateStory}
            >
              <span className="btn-inner">
                {state.isGenerating ? (
                  <>
                    <div className="spinner" />
                    <span>روکیں</span>
                  </>
                ) : (
                  <span>کہانی تخلیق کریں</span>
                )}
              </span>
            </button>

            {/* Error */}
            {state.error && (
              <div className="error-box" dir="rtl">⚠ {state.error}</div>
            )}
          </div>

          {/* OUTPUT */}
          <div className="output-section">
            <div className="output-header">
              <span className="output-label">Generated Story</span>
              <span className="output-label-urdu" dir="rtl">تخلیق شدہ کہانی</span>
            </div>

            <div className={`output-box ${state.isGenerating || state.generatedText ? "active" : ""}`}>
              {state.generatedText ? (
                <div className="story-text" dir="rtl">
                  {state.generatedText}
                  {state.isGenerating && <span className="cursor-blink" />}
                </div>
              ) : (
                <div className="placeholder-text" dir="rtl">
                  {state.isGenerating
                    ? "تخلیق جاری ہے..."
                    : "یہاں آپ کی کہانی ظاہر ہوگی"}
                </div>
              )}
            </div>
          </div>

          {/* FOOTER */}
          <footer className="mt-8 text-center text-gray-500 dark:text-gray-400 text-sm">
            <p>NLP Assignment - Trigram Language Model with BPE Tokenizer</p>
            <p className="mt-1">FastAPI Backend (REST API)</p>
            <div className="mt-2">
              <strong>Group Members:</strong><br />
              Rayan Ahmed (23i-0018) &nbsp;|&nbsp; Awwab Ahmad (23i-0079) &nbsp;|&nbsp; Uwaid Munir (23i-2574)
            </div>
          </footer>
        </div>
      )}
    </>
  );
}