import { useState, useEffect, useCallback } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [error, setError] = useState(null)

  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/history`)
      const data = await res.json()
      setHistory(data)
    } catch (e) {
      console.error('Failed to fetch history:', e)
    }
  }, [])

  useEffect(() => { fetchHistory() }, [fetchHistory])

  const handleFile = (f) => {
    if (f && f.type.startsWith('image/')) {
      setFile(f)
      setPreview(URL.createObjectURL(f))
      setResult(null)
      setError(null)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragActive(false)
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0])
  }

  const handleSubmit = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data = await res.json()
      setResult(data)
      fetchHistory()
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const resolveUrl = (url) => {
    if (!url) return ''
    if (url.startsWith('http')) return url
    return `${API_URL}${url}`
  }

  const getConfidenceColor = (conf) => {
    if (conf >= 0.9) return 'var(--accent-green)'
    if (conf >= 0.7) return 'var(--accent-amber)'
    return 'var(--accent-red)'
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">🔍</div>
            <div>
              <h1>XAI Crack Detection</h1>
              <p className="subtitle">Explainable AI for Structural Analysis</p>
            </div>
          </div>
          <div className="header-badge">
            <span className="pulse"></span>
            AI-Powered
          </div>
        </div>
      </header>

      <main className="main">
        <section className="upload-section">
          <div
            className={`drop-zone ${dragActive ? 'active' : ''} ${preview ? 'has-file' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-input').click()}
          >
            <input
              id="file-input"
              type="file"
              accept="image/*"
              onChange={(e) => handleFile(e.target.files[0])}
              hidden
            />
            {preview ? (
              <div className="preview-container">
                <img src={preview} alt="Preview" className="preview-img" />
                <div className="preview-overlay">
                  <span>Click or drop to change</span>
                </div>
              </div>
            ) : (
              <div className="drop-content">
                <div className="drop-icon">📷</div>
                <p className="drop-title">Drop your image here</p>
                <p className="drop-sub">or click to browse • JPG, PNG supported</p>
              </div>
            )}
          </div>

          <button
            className={`analyze-btn ${loading ? 'loading' : ''}`}
            onClick={handleSubmit}
            disabled={!file || loading}
          >
            {loading ? (
              <><span className="spinner"></span> Analyzing...</>
            ) : (
              <><span className="btn-icon">⚡</span> Analyze Image</>
            )}
          </button>

          {error && <div className="error-msg">⚠️ {error}</div>}
        </section>

        {result && (
          <section className="result-section">
            <h2 className="section-title">Analysis Result</h2>
            <div className="result-card">
              <div className="result-header">
                <div className={`label-badge ${result.label === 'positive' ? 'crack' : 'safe'}`}>
                  {result.label === 'positive' ? '🔴 Crack Detected' : '🟢 No Crack'}
                </div>
                <div className="confidence-display">
                  <div className="confidence-value" style={{ color: getConfidenceColor(result.confidence) }}>
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="confidence-label">Confidence</div>
                </div>
              </div>

              <div className="images-comparison">
                <div className="image-panel">
                  <h3>Original Image</h3>
                  <img src={resolveUrl(result.image_url)} alt="Original" />
                </div>
                <div className="image-divider">
                  <span>→</span>
                </div>
                <div className="image-panel heatmap-panel">
                  <h3>Grad-CAM Heatmap</h3>
                  <img src={resolveUrl(result.heatmap_url)} alt="Heatmap" />
                </div>
              </div>

              <div className="result-meta">
                <span>📁 {result.filename}</span>
                <span>🕐 {new Date(result.created_at).toLocaleString()}</span>
              </div>
            </div>
          </section>
        )}

        <section className="history-section">
          <h2 className="section-title">
            Prediction History
            <span className="history-count">{history.length}</span>
          </h2>
          {history.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">📋</div>
              <p>No predictions yet. Upload an image to get started.</p>
            </div>
          ) : (
            <div className="table-container">
              <table className="history-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Filename</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                    <th>Images</th>
                    <th>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map(row => (
                    <tr key={row.id}>
                      <td className="id-cell">#{row.id}</td>
                      <td className="filename-cell" title={row.filename}>{row.filename}</td>
                      <td>
                        <span className={`label-pill ${row.label === 'positive' ? 'crack' : 'safe'}`}>
                          {row.label === 'positive' ? 'Crack' : 'No Crack'}
                        </span>
                      </td>
                      <td>
                        <div className="conf-bar-wrapper">
                          <div
                            className="conf-bar"
                            style={{
                              width: `${row.confidence * 100}%`,
                              background: getConfidenceColor(row.confidence)
                            }}
                          />
                          <span>{(row.confidence * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="thumb-cell">
                        <img src={resolveUrl(row.image_url)} alt="thumb" className="thumb" />
                        <img src={resolveUrl(row.heatmap_url)} alt="heatmap" className="thumb heatmap-thumb" />
                      </td>
                      <td className="date-cell">
                        {row.created_at ? new Date(row.created_at).toLocaleString() : '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        <p>XAI Crack Detection Service • ResNet18 + Grad-CAM • Built with PyTorch & FastAPI</p>
      </footer>
    </div>
  )
}

export default App
