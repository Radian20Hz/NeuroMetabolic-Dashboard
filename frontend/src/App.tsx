import { useState, useCallback, useEffect } from 'react'
import GlucoseChart from './components/GlucoseChart'
import StatsCards from './components/StatsCards'
import UploadPanel from './components/UploadPanel'
import { fetchLatestReadings } from './api/glucoseApi'
import type { GlucoseReading, UploadResponse } from './types/glucose'

const REFRESH_INTERVAL = 5 * 60 * 1000 // 5 minutes
const HOUR_OPTIONS = [24, 72, 168] as const

function App() {
  const [readings, setReadings] = useState<GlucoseReading[]>([])
  const [uploadStats, setUploadStats] = useState<UploadResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [hours, setHours] = useState<number>(72)

  const loadData = useCallback(async () => {
    try {
      setError(null)
      const data = await fetchLatestReadings(hours)
      setReadings(data.readings ?? [])
      setLastUpdated(new Date())
    } catch {
      setError('Unable to reach backend. Is uvicorn running?')
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

  const formatLastUpdated = () => {
    if (!lastUpdated) return ''
    return lastUpdated.toLocaleTimeString('pl-PL', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  }

  return (
    <div className="app-root">
      <div className="bg-mesh" />

      <div className="app-container">
        {/* Header */}
        <header className="app-header">
          <div className="header-left">
            <div className="logo-badge">NMD</div>
            <div>
              <h1 className="app-title">NeuroMetabolic Dashboard</h1>
              <p className="app-subtitle">AI-driven glycemic decision support · Medtronic 780G</p>
            </div>
          </div>
          <div className="header-right">
            {lastUpdated && (
              <span className="last-updated">↻ {formatLastUpdated()}</span>
            )}
            <div className="hours-selector">
              {HOUR_OPTIONS.map((h) => (
                <button
                  key={h}
                  className={`hours-btn ${hours === h ? 'active' : ''}`}
                  onClick={() => setHours(h)}
                >
                  {h}h
                </button>
              ))}
            </div>
          </div>
        </header>

        {/* Error */}
        {error && (
          <div className="error-banner">
            <span>⚠</span> {error}
          </div>
        )}

        {/* Chart */}
        <div className="glass-card chart-card">
          {loading ? (
            <div className="loading-state">
              <div className="spinner" />
              <span>Loading glucose data...</span>
            </div>
          ) : (
            <GlucoseChart readings={readings} />
          )}
        </div>

        {/* Bottom grid */}
        <div className="bottom-grid">
          <div className="glass-card">
            <StatsCards readings={readings} uploadStats={uploadStats} />
          </div>
          <div className="glass-card">
            <UploadPanel onSuccess={handleUploadSuccess} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
