import { useState, useRef } from 'react'
import { uploadCsv } from '../api/glucoseApi'
import type { UploadResponse } from '../types/glucose'

interface UploadPanelProps {
  onSuccess: (data: UploadResponse) => void
}

function UploadPanel({ onSuccess }: UploadPanelProps) {
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState<UploadResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = (f: File | null | undefined) => {
    if (!f) return
    setFile(f)
    setResult(null)
    setError(null)
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }

  const handleUpload = async () => {
    if (!file) return
    setUploading(true)
    setError(null)
    setResult(null)
    try {
      const data = await uploadCsv(file)
      setResult(data)
      onSuccess(data)
    } catch {
      setError('PARSE_ERROR: invalid CareLink CSV format')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="upload-panel">
      <div
        className={`upload-zone ${dragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        <div className="upload-zone-inner">
          <span className="upload-zone-icon">{file ? '◉' : '○'}</span>
          <span className="upload-zone-text">
            {file ? file.name : 'DROP CARELINK CSV'}
          </span>
          <span className="upload-zone-sub">
            {file ? `${(file.size / 1024).toFixed(1)} KB · READY` : 'or click to browse'}
          </span>
        </div>
      </div>

      <button
        className={`upload-btn ${uploading ? 'uploading' : ''}`}
        onClick={handleUpload}
        disabled={!file || uploading}
      >
        {uploading ? (
          <><span className="upload-spinner">◌</span> PROCESSING...</>
        ) : (
          '▶ UPLOAD & PARSE'
        )}
      </button>

      {result && (
        <div className="upload-result">
          <div className="upload-result-header">
            <span className="upload-ok">✓</span> IMPORT COMPLETE
          </div>
          <div className="upload-result-grid">
            <span className="res-key">READINGS</span>
            <span className="res-val">{result.readings_saved}</span>
            {result.avg_glucose != null && (
              <>
                <span className="res-key">AVG</span>
                <span className="res-val">{Math.round(result.avg_glucose)} mg/dL</span>
              </>
            )}
            {result.time_in_range_percent != null && (
              <>
                <span className="res-key">TIR</span>
                <span className="res-val" style={{
                  color: result.time_in_range_percent >= 70 ? '#00ff88' : '#ffaa00'
                }}>{result.time_in_range_percent}%</span>
              </>
            )}
            {result.gmi != null && (
              <>
                <span className="res-key">GMI</span>
                <span className="res-val">{result.gmi.toFixed(2)}%</span>
              </>
            )}
            {result.cv_percent != null && (
              <>
                <span className="res-key">CV</span>
                <span className="res-val" style={{
                  color: result.cv_is_stable ? '#00ff88' : '#ffaa00'
                }}>
                  {result.cv_percent}% {result.cv_is_stable ? '✓' : '⚠'}
                </span>
              </>
            )}
          </div>
        </div>
      )}

      {error && (
        <div className="upload-error">
          <span className="error-prefix">!</span> {error}
        </div>
      )}
    </div>
  )
}

export default UploadPanel