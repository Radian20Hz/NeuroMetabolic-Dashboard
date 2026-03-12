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
      setError('Upload failed. Check if the file is a valid CareLink CSV.')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="upload-panel">
      <div className="section-label">Import CareLink CSV</div>

      <div
        className={`upload-dropzone ${dragOver ? 'drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <div className="upload-icon">📂</div>
        <div className="upload-text">
          {file ? file.name : 'Drop CSV file here'}
        </div>
        <div className="upload-hint">
          {file ? `${(file.size / 1024).toFixed(1)} KB` : 'or click to browse'}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>

      <button
        className="upload-btn"
        onClick={handleUpload}
        disabled={!file || uploading}
      >
        {uploading ? 'Uploading...' : 'Upload & Analyze'}
      </button>

      {result && (
        <div className="upload-result">
          <div className="upload-result-row">
            <span className="upload-result-key">Readings saved</span>
            <span>{result.readings_saved}</span>
          </div>
          {result.avg_glucose != null && (
            <div className="upload-result-row">
              <span className="upload-result-key">Average glucose</span>
              <span>{Math.round(result.avg_glucose)} mg/dL</span>
            </div>
          )}
          {result.time_in_range_percent != null && (
            <div className="upload-result-row">
              <span className="upload-result-key">Time in Range</span>
              <span>{result.time_in_range_percent}%</span>
            </div>
          )}
          {result.gmi != null && (
            <div className="upload-result-row">
              <span className="upload-result-key">GMI</span>
              <span>{result.gmi.toFixed(2)}%</span>
            </div>
          )}
          {result.cv_percent != null && (
            <div className="upload-result-row">
              <span className="upload-result-key">CV</span>
              <span
                style={{ color: result.cv_is_stable ? '#86efac' : '#fca5a5' }}
              >
                {result.cv_percent}%
                {result.cv_is_stable != null && (
                  <span style={{ marginLeft: 6, fontSize: 11 }}>
                    {result.cv_is_stable ? '✓ Stable' : '⚠ Unstable'}
                  </span>
                )}
              </span>
            </div>
          )}
        </div>
      )}

      {error && <div className="upload-error">{error}</div>}
    </div>
  )
}

export default UploadPanel
