import type { GlucoseReading, UploadResponse, ComputedStats } from '../types/glucose'

interface StatsCardsProps {
  readings: GlucoseReading[]
  uploadStats?: UploadResponse | null
}

function computeStatsFromReadings(data: GlucoseReading[]): ComputedStats | null {
  if (!data || data.length === 0) return null
  const values = data.map((r) => r.glucose_mg_dl)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const avg = values.reduce((a, b) => a + b, 0) / values.length
  const variance = values.reduce((a, b) => a + (b - avg) ** 2, 0) / values.length
  const std_dev = Math.sqrt(variance)
  const inRange = values.filter((v) => v >= 70 && v <= 180).length
  const tir = (inRange / values.length) * 100
  const gmi = 3.31 + 0.02392 * avg
  const cv = (std_dev / avg) * 100

  return {
    min_glucose: Math.round(min),
    max_glucose: Math.round(max),
    avg_glucose: Math.round(avg),
    std_dev: Math.round(std_dev * 10) / 10,
    time_in_range_percent: Math.round(tir * 10) / 10,
    gmi: Math.round(gmi * 100) / 100,
    cv_percent: Math.round(cv * 10) / 10,
    cv_is_stable: cv < 36,
    count: data.length,
  }
}

function tirColor(tir: number): string {
  if (tir >= 70) return '#86efac'
  if (tir >= 50) return '#ffb085'
  return '#fca5a5'
}

function cvColor(cv: number | null, stable: boolean | null): string {
  if (cv === null || stable === null) return '#5a5480'
  return stable ? '#86efac' : '#fca5a5'
}

function StatItem({
  label,
  value,
  unit,
  color,
}: {
  label: string
  value: string | number | null
  unit?: string
  color?: string
}) {
  return (
    <div className="stat-item">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={color ? { color } : undefined}>
        {value ?? '—'}
        {unit && value != null && <span className="stat-unit">{unit}</span>}
      </div>
    </div>
  )
}

function StatsCards({ readings, uploadStats }: StatsCardsProps) {
  const stats: ComputedStats | null = uploadStats
    ? {
        min_glucose: uploadStats.min_glucose ?? 0,
        max_glucose: uploadStats.max_glucose ?? 0,
        avg_glucose: uploadStats.avg_glucose ? Math.round(uploadStats.avg_glucose) : 0,
        std_dev: uploadStats.std_dev,
        time_in_range_percent: uploadStats.time_in_range_percent ?? 0,
        gmi: uploadStats.gmi,
        cv_percent: uploadStats.cv_percent,
        cv_is_stable: uploadStats.cv_is_stable,
        count: uploadStats.readings_saved,
      }
    : computeStatsFromReadings(readings)

  if (!stats) {
    return (
      <div
        style={{
          padding: 24,
          color: '#5a5480',
          fontSize: 14,
          textAlign: 'center',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        Upload a CSV to see statistics
      </div>
    )
  }

  const tir = stats.time_in_range_percent

  return (
    <div className="stats-grid">
      <div className="section-label" style={{ gridColumn: 'span 2' }}>
        Glycemic Statistics
      </div>

      <StatItem label="Minimum" value={stats.min_glucose} unit="mg/dL" />
      <StatItem label="Maximum" value={stats.max_glucose} unit="mg/dL" />
      <StatItem label="Average" value={stats.avg_glucose} unit="mg/dL" />
      <StatItem label="Readings" value={stats.count} />

      {/* Std Dev */}
      <StatItem label="Std Dev" value={stats.std_dev} unit="mg/dL" />

      {/* GMI */}
      <StatItem
        label="GMI (est. HbA1c)"
        value={stats.gmi !== null ? stats.gmi?.toFixed(2) : null}
        unit="%"
      />

      {/* CV */}
      <div className="stat-item" style={{ gridColumn: 'span 2' }}>
        <div className="stat-label">
          Coefficient of Variation
          <span
            style={{
              marginLeft: 8,
              fontSize: 10,
              fontWeight: 700,
              color: cvColor(stats.cv_percent, stats.cv_is_stable),
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
            }}
          >
            {stats.cv_is_stable === true
              ? '✓ Stable'
              : stats.cv_is_stable === false
              ? '⚠ Unstable'
              : ''}
          </span>
        </div>
        <div className="stat-value" style={{ color: cvColor(stats.cv_percent, stats.cv_is_stable) }}>
          {stats.cv_percent !== null ? `${stats.cv_percent}` : '—'}
          {stats.cv_percent !== null && <span className="stat-unit">%</span>}
        </div>
        <div style={{ fontSize: 11, color: '#5a5480', marginTop: 4 }}>
          Target: &lt;36% · ADA 2024
        </div>
      </div>

      {/* TIR bar */}
      <div className="tir-bar-wrap">
        <div className="tir-header">
          <span className="tir-label">Time in Range</span>
          <span className="tir-value" style={{ color: tirColor(tir) }}>
            {tir}%
          </span>
        </div>
        <div className="tir-bar-bg">
          <div
            className="tir-bar-fill"
            style={{
              width: `${tir}%`,
              background: `linear-gradient(90deg, ${tirColor(tir)}88, ${tirColor(tir)})`,
            }}
          />
        </div>
        <div style={{ fontSize: 11, color: '#5a5480', marginTop: 8 }}>
          Target: ≥70% · ADA 2024
        </div>
      </div>
    </div>
  )
}

export default StatsCards
