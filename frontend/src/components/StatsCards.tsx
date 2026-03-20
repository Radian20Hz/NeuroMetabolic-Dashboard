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
  if (tir >= 70) return '#00ff88'
  if (tir >= 50) return '#ffaa00'
  return '#ff4444'
}

interface MetricRowProps {
  label: string
  value: string | number | null
  unit?: string
  color?: string
  target?: string
  status?: 'ok' | 'warn' | 'crit' | null
}

function MetricRow({ label, value, unit, color, target, status }: MetricRowProps) {
  const statusColors = { ok: '#00ff88', warn: '#ffaa00', crit: '#ff4444' }
  const sc = status ? statusColors[status] : undefined

  return (
    <div className="metric-row">
      <span className="metric-label">{label}</span>
      <span className="metric-value" style={{ color: color ?? sc ?? '#e0e0e0' }}>
        {value ?? '—'}
        {unit && value != null && <span className="metric-unit"> {unit}</span>}
      </span>
      {target && <span className="metric-target">{target}</span>}
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
      <div className="stats-empty">
        <span className="stats-empty-icon">▣</span>
        <span>NO DATA — upload CSV to compute metrics</span>
      </div>
    )
  }

  const tir = stats.time_in_range_percent
  const tirStatus = tir >= 70 ? 'ok' : tir >= 50 ? 'warn' : 'crit'
  const cvStatus = stats.cv_is_stable ? 'ok' : 'warn'

  return (
    <div className="stats-panel">
      <div className="stats-group">
        <div className="stats-group-label">GLUCOSE RANGE</div>
        <MetricRow label="MIN" value={stats.min_glucose} unit="mg/dL"
          color={stats.min_glucose < 70 ? '#ff4444' : '#888'} />
        <MetricRow label="MAX" value={stats.max_glucose} unit="mg/dL"
          color={stats.max_glucose > 180 ? '#ffaa00' : '#888'} />
        <MetricRow label="AVG" value={stats.avg_glucose} unit="mg/dL" />
        <MetricRow label="SD" value={stats.std_dev} unit="mg/dL" />
        <MetricRow label="N" value={stats.count} />
      </div>

      <div className="stats-divider" />

      <div className="stats-group">
        <div className="stats-group-label">CLINICAL INDICES</div>
        <MetricRow
          label="GMI"
          value={stats.gmi?.toFixed(2) ?? null}
          unit="%"
          target="est. HbA1c"
        />
        <MetricRow
          label="CV"
          value={stats.cv_percent}
          unit="%"
          status={cvStatus}
          target={stats.cv_is_stable ? '✓ STABLE' : '⚠ UNSTABLE'}
        />
      </div>

      <div className="stats-divider" />

      <div className="stats-group">
        <div className="stats-group-label">TIME IN RANGE · ADA 2024</div>
        <div className="tir-display">
          <span className="tir-number" style={{ color: tirColor(tir) }}>
            {tir}
          </span>
          <span className="tir-pct" style={{ color: tirColor(tir) }}>%</span>
          <span className="tir-status" style={{ color: tirColor(tir) }}>
            {tirStatus === 'ok' ? '↑ TARGET MET' : tirStatus === 'warn' ? '→ SUBOPTIMAL' : '↓ BELOW TARGET'}
          </span>
        </div>
        <div className="tir-bar-track">
          <div
            className="tir-bar-fill"
            style={{
              width: `${tir}%`,
              background: tirColor(tir),
            }}
          />
          <div className="tir-target-line" style={{ left: '70%' }} />
        </div>
        <div className="tir-footnote">Target ≥70% · {stats.count} readings</div>
      </div>
    </div>
  )
}

export default StatsCards