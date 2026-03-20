import { useState } from 'react'
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import type { TooltipProps } from 'recharts'
import type { GlucoseReading } from '../types/glucose'
import { fetchPrediction } from '../api/glucoseApi'
import type { PredictionPoint } from '../types/glucose'

interface ChartPoint {
  time: string
  timestamp: number
  glucose?: number
  pred?: number
  lower?: number
  upper?: number
}

const formatTime = (timestamp: string): string => {
  const date = new Date(timestamp)
  return date.toLocaleTimeString('pl-PL', { hour: '2-digit', minute: '2-digit' })
}

interface TooltipEntry {
  dataKey?: string | number
  value?: number | string | Array<number | string>
  payload?: ChartPoint
}

interface CustomTooltipProps { active?: boolean; payload?: TooltipEntry[] }
const CustomTooltip = ({ active, payload }: CustomTooltipProps) => {
  if (!active || !payload?.length) return null
  const entries = payload as TooltipEntry[]
  const glucose = entries.find((p) => p.dataKey === 'glucose')?.value as number | undefined
  const pred = entries.find((p) => p.dataKey === 'pred')?.value as number | undefined
  const val = glucose ?? pred
  if (val == null) return null

  const color = val < 70 ? 'var(--red)' : val > 180 ? 'var(--amber)' : 'var(--green)'
  const time = entries[0]?.payload?.time ?? ''

  return (
    <div className="chart-tooltip">
      <div className="tooltip-time">{time}</div>
      <div className="tooltip-value" style={{ color }}>
        {Math.round(val)}
        <span className="tooltip-unit"> mg/dL</span>
      </div>
      {pred != null && <div className="tooltip-tag">Predicted</div>}
    </div>
  )
}

interface GlucoseChartProps {
  readings: GlucoseReading[]
}

function GlucoseChart({ readings }: GlucoseChartProps) {
  const [predicting, setPredicting] = useState(false)
  const [predictions, setPredictions] = useState<PredictionPoint[] | null>(null)
  const [predError, setPredError] = useState<string | null>(null)

  const handlePredict = async () => {
    if (readings.length < 24) {
      setPredError('Need at least 24 readings (2h of data) to generate a forecast.')
      return
    }
    setPredicting(true)
    setPredError(null)
    try {
      const recent = readings.slice(-48).map((r) => r.glucose_mg_dl)
      const result = await fetchPrediction(recent)
      setPredictions(result.predictions)
    } catch {
      setPredError('Forecast failed — check backend.')
    } finally {
      setPredicting(false)
    }
  }

  if (!readings || readings.length === 0) {
    return (
      <div className="chart-empty">
        <span className="chart-empty-icon">📈</span>
        <span>No data — upload a CareLink CSV to visualize glucose</span>
      </div>
    )
  }

  const histData: ChartPoint[] = readings.slice(-288).map((r) => ({
    time: formatTime(r.timestamp),
    timestamp: new Date(r.timestamp).getTime(),
    glucose: Math.round(r.glucose_mg_dl),
  }))

  let allData = [...histData]
  if (predictions) {
    const lastTs = histData[histData.length - 1].timestamp
    allData[allData.length - 1] = {
      ...allData[allData.length - 1],
      pred: allData[allData.length - 1].glucose,
      lower: allData[allData.length - 1].glucose,
      upper: allData[allData.length - 1].glucose,
    }
    const predPoints: ChartPoint[] = predictions.map((p) => ({
      time: formatTime(new Date(lastTs + p.minutes_ahead * 60000).toISOString()),
      timestamp: lastTs + p.minutes_ahead * 60000,
      pred: Math.round(p.glucose_mg_dl),
      lower: Math.round(p.lower_mg_dl),
      upper: Math.round(p.upper_mg_dl),
    }))
    allData = [...allData, ...predPoints]
  }

  const tickInterval = Math.max(1, Math.floor(allData.length / 8))

  return (
    <div className="chart-wrap">
      <div className="chart-controls">
        <div className="chart-legend">
          <div className="legend-item">
            <span className="legend-line" style={{ background: 'var(--green)' }} />
            <span>CGM reading</span>
          </div>
          {predictions && (
            <div className="legend-item">
              <span className="legend-line" style={{ background: 'var(--blue)', opacity: 0.7 }} />
              <span>TFT forecast · 60 min</span>
            </div>
          )}
        </div>
        <button
          className="predict-btn"
          onClick={handlePredict}
          disabled={predicting || readings.length < 24}
        >
          {predicting ? '⟳ Forecasting…' : predictions ? '↺ Refresh Forecast' : '✦ Run AI Forecast'}
        </button>
      </div>

      {predError && <div className="pred-error">{predError}</div>}

      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={allData} margin={{ top: 8, right: 4, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="glucoseGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--green)" stopOpacity={0.15} />
              <stop offset="100%" stopColor="var(--green)" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="ciGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--blue)" stopOpacity={0.12} />
              <stop offset="100%" stopColor="var(--blue)" stopOpacity={0.03} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="4 4" stroke="rgba(0,0,0,0.05)" vertical={false} />

          <XAxis
            dataKey="time"
            tick={{ fill: '#aeaeb2', fontSize: 11, fontFamily: 'DM Sans, sans-serif' }}
            axisLine={false}
            tickLine={false}
            interval={tickInterval}
          />
          <YAxis
            domain={[40, 400]}
            tick={{ fill: '#aeaeb2', fontSize: 11, fontFamily: 'DM Sans, sans-serif' }}
            axisLine={false}
            tickLine={false}
            width={32}
          />

          <Tooltip content={<CustomTooltip />} />

          <ReferenceLine y={70} stroke="var(--red)" strokeDasharray="4 4" strokeOpacity={0.4} strokeWidth={1} />
          <ReferenceLine y={180} stroke="var(--amber)" strokeDasharray="4 4" strokeOpacity={0.4} strokeWidth={1} />

          {predictions && (
            <Area dataKey="upper" stroke="none" fill="url(#ciGrad)" legendType="none" connectNulls />
          )}
          {predictions && (
            <Area dataKey="lower" stroke="none" fill="transparent" legendType="none" connectNulls />
          )}

          <Line
            type="monotone"
            dataKey="glucose"
            stroke="var(--green)"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: 'var(--green)', stroke: 'white', strokeWidth: 2 }}
            connectNulls
          />

          {predictions && (
            <Line
              type="monotone"
              dataKey="pred"
              stroke="var(--blue)"
              strokeWidth={2}
              strokeDasharray="6 4"
              dot={false}
              activeDot={{ r: 4, fill: 'var(--blue)', stroke: 'white', strokeWidth: 2 }}
              connectNulls
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

export default GlucoseChart