import { useState, useCallback, useEffect } from 'react'
import GlucoseChart from './components/GlucoseChart'
import StatsCards from './components/StatsCards'
import UploadPanel from './components/UploadPanel'
import ClarkeErrorGrid from './components/ClarkeErrorGrid'
import { fetchLatestReadings, fetchPrediction, fetchClarke } from './api/glucoseApi'
import type { GlucoseReading, UploadResponse } from './types/glucose'
import type { ClarkeResponseAPI } from './api/glucoseApi'

const REFRESH_INTERVAL = 5 * 60 * 1000
const HOUR_OPTIONS = [24, 72, 168, 720] as const

function App() {
  const [readings, setReadings] = useState<GlucoseReading[]>([])
  const [uploadStats, setUploadStats] = useState<UploadResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [hours, setHours] = useState<number>(720)
  const [clarkeResult, setClarkeResult] = useState<ClarkeResponseAPI | null>(null)
  const [clarkeLoading, setClarkeLoading] = useState(false)
  const [clarkeError, setClarkeError] = useState<string | null>(null)

  const loadData = useCallback(async () => {
    try {
      setError(null)
      const data = await fetchLatestReadings(hours)
      setReadings(data.readings ?? [])
      setLastUpdated(new Date())
    } catch {
      setError('Cannot reach backend')
    } finally {
      setLoading(false)
    }
  }, [hours])

  useEffect(() => {
    loadData()
    const interval = setInterval(loadData, REFRESH_INTERVAL)
    return () => clearInterval(interval)
  }, [loadData])

  const handleUploadSuccess = (data: UploadResponse) => {
    setUploadStats(data)
    loadData()
  }

  const handleRunClarke = async () => {
    if (readings.length < 24) {
      setClarkeError('Need at least 24 readings to run Clarke EGA.')
      return
    }
    setClarkeLoading(true)
    setClarkeError(null)
    try {
      const recent = readings.slice(-48)
      const referenceValues = recent.map((r) => r.glucose_mg_dl)
      const predResult = await fetchPrediction(referenceValues.slice(-24))
      // Use last 12 reference values paired with 12 predictions
      const ref12 = referenceValues.slice(-12)
      const pred12 = predResult.predictions.map((p) => p.glucose_mg_dl)
      const result = await fetchClarke({
        reference_values: ref12,
        predicted_values: pred12,
      })
      setClarkeResult(result)
    } catch {
      setClarkeError('Clarke EGA failed — check backend.')
    } finally {
      setClarkeLoading(false)
    }
  }

  const now = lastUpdated
    ? lastUpdated.toLocaleTimeString('pl-PL', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : '--:--:--'

  const latestGlucose = readings.length > 0 ? readings[readings.length - 1].glucose_mg_dl : null

  const glucoseZone = latestGlucose
    ? latestGlucose < 54 ? 'CRITICAL LOW'
    : latestGlucose < 70 ? 'LOW'
    : latestGlucose <= 180 ? 'IN RANGE'
    : latestGlucose <= 250 ? 'HIGH'
    : 'CRITICAL HIGH'
    : null

  const zoneColor: Record<string, string> = {
    'CRITICAL LOW': 'var(--red)',
    'LOW': 'var(--amber)',
    'IN RANGE': 'var(--green)',
    'HIGH': 'var(--amber)',
    'CRITICAL HIGH': 'var(--red)',
  }

  const zoneBg: Record<string, string> = {
    'CRITICAL LOW': 'var(--red-bg)',
    'LOW': 'var(--amber-bg)',
    'IN RANGE': 'var(--green-bg)',
    'HIGH': 'var(--amber-bg)',
    'CRITICAL HIGH': 'var(--red-bg)',
  }

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="header-left">
          <div className="logo-mark">
            <span className="logo-cross">✚</span>
          </div>
          <div>
            <h1 className="app-title">NeuroMetabolic Dashboard</h1>
            <p className="app-subtitle">CGM · AI Prediction · Medtronic 780G</p>
          </div>
        </div>

        <div className="header-right">
          {latestGlucose && glucoseZone && (
            <div className="live-glucose" style={{ color: zoneColor[glucoseZone] }}>
              <span className="live-dot" style={{ background: zoneColor[glucoseZone] }} />
              <span className="live-value">{Math.round(latestGlucose)}</span>
              <span className="live-unit">mg/dL</span>
              <span className="live-badge" style={{
                color: zoneColor[glucoseZone],
                background: zoneBg[glucoseZone],
              }}>{glucoseZone}</span>
            </div>
          )}
          <div className="header-meta">
            <span className="meta-time">Updated {now}</span>
            <div className="hours-selector">
              {HOUR_OPTIONS.map((h) => (
                <button
                  key={h}
                  className={`hours-btn ${hours === h ? 'active' : ''}`}
                  onClick={() => setHours(h)}
                >
                  {h >= 168 ? `${h / 24}d` : `${h}h`}
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>

      {error && (
        <div className="error-strip">
          ⚠ {error} — check if uvicorn is running on :8000
        </div>
      )}

      <main className="app-main">
        {/* Chart card */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Glycemic Trace</span>
            <span style={{ fontSize: 12, color: 'var(--text-3)' }}>{readings.length} readings</span>
          </div>
          <div className="card-body">
            {loading ? (
              <div className="loading-state">
                <div className="loading-bar" />
                <span>Loading glucose data…</span>
              </div>
            ) : (
              <GlucoseChart readings={readings} />
            )}
          </div>
        </div>

        {/* Bottom grid */}
        <div className="bottom-grid">
          <div className="card">
            <div className="card-header">
              <span className="card-title">Glycemic Metrics</span>
            </div>
            <div className="card-body">
              <StatsCards readings={readings} uploadStats={uploadStats} />
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <span className="card-title">Import Data</span>
            </div>
            <div className="card-body">
              <UploadPanel onSuccess={handleUploadSuccess} />
            </div>
          </div>
        </div>

        {/* Clarke Error Grid card */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Clarke Error Grid Analysis</span>
            <button
              className="predict-btn"
              onClick={handleRunClarke}
              disabled={clarkeLoading || readings.length < 24}
              style={{ fontSize: 12, padding: '6px 14px' }}
            >
              {clarkeLoading ? '⟳ Running…' : clarkeResult ? '↺ Refresh EGA' : '✦ Run Clarke EGA'}
            </button>
          </div>
          <div className="card-body">
            {clarkeError && <div className="pred-error" style={{ marginBottom: 16 }}>{clarkeError}</div>}
            {clarkeResult ? (
              <ClarkeErrorGrid result={clarkeResult} />
            ) : (
              <div className="chart-empty" style={{ height: 200 }}>
                <span className="chart-empty-icon">◎</span>
                <span>Run Clarke EGA to evaluate TFT forecast accuracy against recent CGM readings</span>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
