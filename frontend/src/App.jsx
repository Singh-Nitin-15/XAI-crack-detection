import { useState, useEffect, useCallback, useRef } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const ALLOWED_EXT = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
const MAX_MB = 20

const isNetworkError = (e) => e.name === 'TypeError' || e.message.includes('fetch')

const resolveUrl = (url) => {
  if (!url) return ''
  if (url.startsWith('http')) return url
  return `${API_URL}${url}`
}

const formatDate = (iso) => {
  if (!iso) return '—'
  return new Date(iso).toLocaleString(undefined, {
    day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

const formatSize = (bytes) =>
  bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`

const confidenceLevel = (c) => (c >= 0.9 ? 'high' : c >= 0.7 ? 'moderate' : 'low')

const padId = (id) => `#${String(id).padStart(4, '0')}`

function Mark() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="2.75" y="2.75" width="18.5" height="18.5" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M9 2.75 11.2 8.4 8.6 12.1 13 15.6 11.6 21.25" fill="none" stroke="var(--accent)"
        strokeWidth="1.75" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.5"
      strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 15V4M7.5 8.5 12 4l4.5 4.5" />
      <path d="M4 14.5V19a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4.5" />
    </svg>
  )
}

function Result({ result }) {
  const [mode, setMode] = useState('side')
  const [blend, setBlend] = useState(60)
  const [ratio, setRatio] = useState(1)

  const crack = result.label === 'positive'
  const pct = result.confidence * 100
  const onOriginalLoad = (e) => {
    const { naturalWidth: w, naturalHeight: h } = e.currentTarget
    if (w && h) setRatio(w / h)
  }
  // Frames take the original photo's aspect ratio. The heatmap is a square
  // 224px render, so stretching it into the same frame lines it up exactly.
  const frameStyle = (maxHeight) => ({
    aspectRatio: ratio,
    width: `min(100%, ${Math.round(maxHeight * ratio)}px)`,
  })

  return (
    <div className={`result ${crack ? 'is-crack' : 'is-clear'}`}>
      <div className="summary">
        <div className="verdict">
          <span className="label">Classification</span>
          <strong>{crack ? 'Crack detected' : 'No crack detected'}</strong>
        </div>
        <div className="gauge">
          <div className="gauge-row">
            <span className="label">Confidence</span>
            <span className="gauge-value">
              {pct.toFixed(1)}% <span className="muted">· {confidenceLevel(result.confidence)}</span>
            </span>
          </div>
          <div className="gauge-track" role="meter" aria-valuemin={0} aria-valuemax={100}
            aria-valuenow={Number(pct.toFixed(1))} aria-label="Confidence">
            <div className="gauge-fill" style={{ width: `${pct}%` }} />
          </div>
          <div className="gauge-ticks"><span>0</span><span>50</span><span>100</span></div>
        </div>
      </div>

      <div className="viewer">
        <div className="viewer-head">
          <span className="label">Explanation</span>
          <div className="seg" role="tablist" aria-label="Image view">
            <button role="tab" aria-selected={mode === 'side'} onClick={() => setMode('side')}>Side by side</button>
            <button role="tab" aria-selected={mode === 'overlay'} onClick={() => setMode('overlay')}>Overlay</button>
          </div>
        </div>

        {mode === 'side' ? (
          <div className="figs">
            <figure>
              <div className="frame" style={frameStyle(320)}>
                <img src={resolveUrl(result.image_url)} alt="Uploaded surface" onLoad={onOriginalLoad} />
              </div>
              <figcaption>Fig. A — Input image</figcaption>
            </figure>
            <figure>
              <div className="frame" style={frameStyle(320)}>
                <img src={resolveUrl(result.heatmap_url)} alt="Grad-CAM heatmap" />
              </div>
              <figcaption>Fig. B — Grad-CAM attention</figcaption>
            </figure>
          </div>
        ) : (
          <figure>
            <div className="frame" style={frameStyle(420)}>
              <img src={resolveUrl(result.image_url)} alt="Uploaded surface" onLoad={onOriginalLoad} />
              <img src={resolveUrl(result.heatmap_url)} alt="Grad-CAM heatmap overlay" style={{ opacity: blend / 100 }} />
            </div>
            <label className="slider">
              <span>Input</span>
              <input type="range" min="0" max="100" value={blend}
                onChange={(e) => setBlend(Number(e.target.value))} aria-label="Heatmap opacity" />
              <span>Heatmap</span>
            </label>
          </figure>
        )}

        <div className="legend">
          <span>Low attention</span>
          <div className="legend-bar" aria-hidden="true" />
          <span>High</span>
        </div>
      </div>

      <div className="result-meta">
        <span>{padId(result.id)}</span>
        <span>{result.filename}</span>
        <span>{formatDate(result.created_at)}</span>
      </div>
    </div>
  )
}

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [error, setError] = useState(null)
  const [apiStatus, setApiStatus] = useState('checking')
  const inputRef = useRef(null)
  const resultRef = useRef(null)

  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/history`)
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      setHistory(await res.json())
      setApiStatus('online')
    } catch (e) {
      if (isNetworkError(e)) setApiStatus('offline')
      console.error('Failed to fetch history:', e)
    }
  }, [])

  useEffect(() => { fetchHistory() }, [fetchHistory])

  // Release the previous preview's object URL when it changes or on unmount.
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  const handleFile = (f) => {
    if (!f) return
    const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase()
    if (!f.type.startsWith('image/') || !ALLOWED_EXT.includes(ext)) {
      setError('Please choose a JPG, PNG, WebP or BMP image.')
      return
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setError(`This file is ${formatSize(f.size)}. The limit is ${MAX_MB} MB.`)
      return
    }
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setError(null)
  }

  const clearFile = () => {
    setFile(null)
    setPreview(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragActive(false)
    handleFile(e.dataTransfer.files[0])
  }

  const openPicker = () => inputRef.current?.click()

  const handleSubmit = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData })
      if (!res.ok) {
        let detail = `Server error: ${res.status}`
        try {
          const body = await res.json()
          if (typeof body.detail === 'string') detail = body.detail
        } catch { /* non-JSON error body: keep the status message */ }
        throw new Error(detail)
      }
      setResult(await res.json())
      setApiStatus('online')
      fetchHistory()
    } catch (e) {
      if (isNetworkError(e)) {
        setError('Cannot reach the API. Make sure the backend is running.')
        setApiStatus('offline')
      } else {
        setError(e.message)
      }
    } finally {
      setLoading(false)
    }
  }

  const openRecord = (row) => {
    setResult(row)
    resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }

  const statusText = { checking: 'Connecting', online: 'API online', offline: 'API offline' }[apiStatus]

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <Mark />
            <div>
              <span className="brand-name">Crack Detection</span>
              <span className="brand-sub">Structural surface inspection</span>
            </div>
          </div>
          <div className={`status status-${apiStatus}`} role="status">
            <span className="status-dot" />
            {statusText}
          </div>
        </div>
      </header>

      <main className="main">
        {apiStatus === 'offline' && (
          <div className="notice" role="alert">
            Can’t reach the API at <code>{API_URL}</code>. Start the backend, then refresh this page.
          </div>
        )}

        <section className="intro">
          <p className="eyebrow">Concrete crack inspection</p>
          <h1>Check a surface photo for cracks.</h1>
          <p>
            Upload an image of concrete. A ResNet-50 classifier labels it cracked or intact,
            and a Grad-CAM heatmap shows which parts of the image drove that decision.
          </p>
        </section>

        <div className="workspace">
          <section className="panel">
            <header className="panel-head">
              <span className="step">01</span>
              <h2>Specimen</h2>
            </header>

            <div
              className={`dropzone${dragActive ? ' is-drag' : ''}${preview ? ' has-file' : ''}`}
              role="button"
              tabIndex={0}
              aria-label={preview ? 'Change image' : 'Choose an image to upload'}
              onClick={openPicker}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openPicker() } }}
              onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
            >
              <input
                ref={inputRef}
                type="file"
                accept={ALLOWED_EXT.join(',')}
                onChange={(e) => handleFile(e.target.files[0])}
                hidden
              />
              {preview ? (
                <img src={preview} alt="Selected image preview" className="dz-preview" />
              ) : (
                <div className="dz-empty">
                  <UploadIcon />
                  <p className="dz-title">Drop an image here or <u>browse</u></p>
                  <p className="dz-hint">JPG, PNG, WebP, BMP · up to {MAX_MB} MB</p>
                </div>
              )}
            </div>

            {file && (
              <div className="file-meta">
                <span className="name" title={file.name}>{file.name}</span>
                <span>{formatSize(file.size)}</span>
                <button className="link-btn" onClick={clearFile} disabled={loading}>Remove</button>
              </div>
            )}

            <button className="btn-primary" onClick={handleSubmit} disabled={!file || loading}>
              {loading ? 'Analyzing…' : 'Run analysis'}
            </button>
            {loading && <div className="progress" aria-hidden="true"><span /></div>}
            {error && <p className="error" role="alert">{error}</p>}
          </section>

          <section className="panel" ref={resultRef} aria-live="polite">
            <header className="panel-head">
              <span className="step">02</span>
              <h2>Result</h2>
            </header>
            {result ? (
              <Result key={result.id} result={result} />
            ) : (
              <div className="placeholder">
                <strong>No result yet</strong>
                <span>Run an analysis, or select a past inspection from the log below.</span>
              </div>
            )}
          </section>
        </div>

        <section className="log">
          <header className="log-head">
            <h2>Inspection log</h2>
            <span className="muted">
              {history.length} {history.length === 1 ? 'record' : 'records'}
            </span>
          </header>

          {history.length === 0 ? (
            <div className="empty">No inspections yet. Results you run will be listed here.</div>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th className="col-id">ID</th>
                    <th>Specimen</th>
                    <th>Result</th>
                    <th>Confidence</th>
                    <th className="col-date">Recorded</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((row) => (
                    <tr
                      key={row.id}
                      className={result?.id === row.id ? 'is-selected' : undefined}
                      tabIndex={0}
                      onClick={() => openRecord(row)}
                      onKeyDown={(e) => { if (e.key === 'Enter') openRecord(row) }}
                      aria-label={`Open inspection ${padId(row.id)}`}
                    >
                      <td className="col-id">{padId(row.id)}</td>
                      <td>
                        <div className="specimen">
                          <div className="thumbs">
                            <img src={resolveUrl(row.image_url)} alt="" loading="lazy" />
                            <img src={resolveUrl(row.heatmap_url)} alt="" loading="lazy" />
                          </div>
                          <span className="fname" title={row.filename}>{row.filename}</span>
                        </div>
                      </td>
                      <td>
                        <span className={`tag ${row.label === 'positive' ? 'crack' : 'clear'}`}>
                          {row.label === 'positive' ? 'Crack' : 'No crack'}
                        </span>
                      </td>
                      <td>
                        <div className="conf">
                          <div className="conf-track"><span style={{ width: `${row.confidence * 100}%` }} /></div>
                          {(row.confidence * 100).toFixed(1)}%
                        </div>
                      </td>
                      <td className="col-date">{formatDate(row.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        <div className="footer-inner">
          <span>ResNet-50 classifier · Grad-CAM on layer4</span>
          <span>PyTorch · FastAPI · React</span>
        </div>
      </footer>
    </div>
  )
}

export default App
