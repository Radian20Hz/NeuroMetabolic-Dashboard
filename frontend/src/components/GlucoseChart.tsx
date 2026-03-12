import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import type { TooltipProps } from 'recharts'
import type { GlucoseReading } from '../types/glucose'

interface ChartPoint {
  time: string
  glucose: number
}

const formatTime = (timestamp: string): string => {
  const date = new Date(timestamp)
  return date.toLocaleTimeString('pl-PL', { hour: '2-digit', minute: '2-digit' })
}

const CustomTooltip = ({ active, payload }: TooltipProps<number, string>) => {
  if (!active || !payload?.length) return null
  const val = payload[0].value as number
  const color = val < 70 ? '#fca5a5' : val > 180 ? '#ffb085' : '#86efac'

  return (
    <div style={{
      background: 'rgba(20,18,40,0.97)',
      border: '1px solid rgba(180,150,255,0.25)',
      borderRadius: 12,
      padding: '10px 14px',
      fontFamily: 'Outfit, sans-serif',
    }}>
      <div style={{ fontSize: 11, color: '#5a5480', marginBottom: 4 }}>
        {(payload[0].payload as ChartPoint).time}
      </div>
      <div style={{ fontSize: 20, fontWeight: 600, color, fontFamily: 'DM Mono, monospace' }}>
        {val} <span style={{ fontSize: 12, color: '#5a5480' }}>mg/dL</span>
      </div>
    </div>
  )
}

interface GlucoseChartProps {
  readings: GlucoseReading[]
}

function GlucoseChart({ readings }: GlucoseChartProps) {
  if (!readings || readings.length === 0) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: 320,
        color: '#5a5480',
        fontSize: 14,
        fontFamily: 'Outfit, sans-serif',
      }}>
        No glucose data available
      </div>
    )
  }

  const data: ChartPoint[] = readings.map((r) => ({
    time: formatTime(r.timestamp),
    glucose: Math.round(r.glucose_mg_dl),
  }))

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <div className="section-label">CGM Glucose Trace</div>
          <div style={{ fontSize: 13, color: '#5a5480' }}>{readings.length} readings</div>
        </div>
        <div style={{ display: 'flex', gap: 16, fontSize: 12, color: '#5a5480' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 20, height: 2, background: '#fca5a5', display: 'inline-block', borderRadius: 2 }} />
            Hypo &lt;70
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 20, height: 2, background: '#ffb085', display: 'inline-block', borderRadius: 2 }} />
            Hyper &gt;180
          </span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <defs>
            <linearGradient id="glucoseLine" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#b49dff" />
              <stop offset="50%" stopColor="#7dd3fc" />
              <stop offset="100%" stopColor="#86efac" />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
          <XAxis
            dataKey="time"
            tick={{ fill: '#5a5480', fontSize: 11, fontFamily: 'DM Mono' }}
            axisLine={false}
            tickLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[40, 400]}
            tick={{ fill: '#5a5480', fontSize: 11, fontFamily: 'DM Mono' }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => `${v}`}
            width={36}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={70} stroke="#fca5a5" strokeDasharray="4 4" strokeOpacity={0.6} />
          <ReferenceLine y={180} stroke="#ffb085" strokeDasharray="4 4" strokeOpacity={0.6} />
          <Line
            type="monotone"
            dataKey="glucose"
            stroke="url(#glucoseLine)"
            strokeWidth={2.5}
            dot={false}
            activeDot={{ r: 5, fill: '#b49dff', stroke: '#0d0d1a', strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default GlucoseChart
