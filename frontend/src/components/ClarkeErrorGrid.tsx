import { useMemo } from 'react'

interface ClarkePoint {
  reference: number
  predicted: number
  zone: string
}

interface ClarkeResult {
  total: number
  zone_counts: Record<string, number>
  zone_percents: Record<string, number>
  clinically_acceptable_percent: number
  points: ClarkePoint[]
}

interface ClarkeErrorGridProps {
  result: ClarkeResult
}

const ZONE_COLORS: Record<string, string> = {
  A: '#30d158',
  B: '#0071e3',
  C: '#ff9f0a',
  D: '#ff6930',
  E: '#ff3b30',
}

const ZONE_BG: Record<string, string> = {
  A: '#f0fdf4',
  B: '#f0f5ff',
  C: '#fffbeb',
  D: '#fff4f0',
  E: '#fff1f0',
}

// SVG dimensions
const W = 480
const H = 480
const PAD = 48
const PLOT_W = W - PAD * 2
const PLOT_H = H - PAD * 2
const MAX_VAL = 400

function toX(ref: number) {
  return PAD + (ref / MAX_VAL) * PLOT_W
}

function toY(pred: number) {
  return PAD + PLOT_H - (pred / MAX_VAL) * PLOT_H
}

// Clarke zone boundary lines (simplified key lines)
const ZONE_LINES = [
  // Zone A upper boundary
  { x1: 0, y1: 0, x2: 70, y2: 56, stroke: '#30d158', opacity: 0.25 },
  { x1: 70, y1: 56, x2: 400, y2: 320, stroke: '#30d158', opacity: 0.25 },
  // Zone A lower boundary
  { x1: 0, y1: 0, x2: 70, y2: 84, stroke: '#30d158', opacity: 0.25 },
  { x1: 70, y1: 84, x2: 400, y2: 480, stroke: '#30d158', opacity: 0.25 },
  // Hypo/Hyper reference lines
  { x1: 70, y1: 0, x2: 70, y2: 400, stroke: '#ff3b30', opacity: 0.12, dash: '4 4' },
  { x1: 180, y1: 0, x2: 180, y2: 400, stroke: '#ff9f0a', opacity: 0.12, dash: '4 4' },
  { x1: 0, y1: 70, x2: 400, y2: 70, stroke: '#ff3b30', opacity: 0.12, dash: '4 4' },
  { x1: 0, y1: 180, x2: 400, y2: 180, stroke: '#ff9f0a', opacity: 0.12, dash: '4 4' },
]

// Zone label positions [ref, pred, label]
const ZONE_LABELS: [number, number, string][] = [
  [100, 100, 'A'],
  [60, 180, 'B'],
  [280, 380, 'B'],
  [200, 380, 'C'],
  [350, 120, 'D'],
  [50, 320, 'E'],
  [320, 50, 'E'],
]

function ClarkeErrorGrid({ result }: ClarkeErrorGridProps) {
  const ticks = [0, 70, 100, 180, 250, 300, 400]

  const zoneSummary = useMemo(() => {
    return Object.entries(result.zone_counts)
      .map(([zone, count]) => ({
        zone,
        count,
        pct: result.zone_percents[zone],
      }))
      .filter((z) => z.count > 0)
      .sort((a, b) => a.zone.localeCompare(b.zone))
  }, [result])

  return (
    <div className="clarke-wrap">
      {/* Summary bar */}
      <div className="clarke-summary">
        <div className="clarke-acceptable">
          <span className="clarke-acceptable-value" style={{
            color: result.clinically_acceptable_percent >= 95 ? 'var(--green)'
              : result.clinically_acceptable_percent >= 85 ? 'var(--amber)'
              : 'var(--red)'
          }}>
            {result.clinically_acceptable_percent}%
          </span>
          <span className="clarke-acceptable-label">A+B (clinically acceptable)</span>
        </div>
        <div className="clarke-zone-pills">
          {zoneSummary.map(({ zone, count, pct }) => (
            <div key={zone} className="clarke-pill" style={{
              background: ZONE_BG[zone],
              color: ZONE_COLORS[zone],
              border: `1px solid ${ZONE_COLORS[zone]}33`,
            }}>
              <span className="pill-zone">Zone {zone}</span>
              <span className="pill-count">{count}</span>
              <span className="pill-pct">{pct}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* SVG plot */}
      <div className="clarke-plot-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 520 }}>
          {/* Background */}
          <rect x={PAD} y={PAD} width={PLOT_W} height={PLOT_H} fill="#fafafa" rx={4} />

          {/* Grid lines */}
          {ticks.map((t) => (
            <g key={t}>
              <line
                x1={toX(t)} y1={PAD}
                x2={toX(t)} y2={PAD + PLOT_H}
                stroke="rgba(0,0,0,0.06)" strokeWidth={1}
              />
              <line
                x1={PAD} y1={toY(t)}
                x2={PAD + PLOT_W} y2={toY(t)}
                stroke="rgba(0,0,0,0.06)" strokeWidth={1}
              />
              {t > 0 && (
                <>
                  <text x={toX(t)} y={PAD + PLOT_H + 16} textAnchor="middle"
                    fontSize={10} fill="#aeaeb2" fontFamily="DM Mono, monospace">
                    {t}
                  </text>
                  <text x={PAD - 8} y={toY(t) + 4} textAnchor="end"
                    fontSize={10} fill="#aeaeb2" fontFamily="DM Mono, monospace">
                    {t}
                  </text>
                </>
              )}
            </g>
          ))}

          {/* Zone boundary lines */}
          {ZONE_LINES.map((l, i) => (
            <line
              key={i}
              x1={toX(l.x1)} y1={toY(l.y1)}
              x2={toX(l.x2)} y2={toY(l.y2)}
              stroke={l.stroke}
              strokeOpacity={l.opacity}
              strokeWidth={1.5}
              strokeDasharray={l.dash}
            />
          ))}

          {/* Diagonal reference line */}
          <line
            x1={toX(0)} y1={toY(0)}
            x2={toX(400)} y2={toY(400)}
            stroke="rgba(0,0,0,0.1)"
            strokeWidth={1}
            strokeDasharray="4 4"
          />

          {/* Zone labels */}
          {ZONE_LABELS.map(([ref, pred, label], i) => (
            <text
              key={i}
              x={toX(ref)} y={toY(pred)}
              textAnchor="middle" dominantBaseline="middle"
              fontSize={18} fontWeight={600}
              fill={ZONE_COLORS[label]}
              opacity={0.18}
              fontFamily="DM Sans, sans-serif"
            >
              {label}
            </text>
          ))}

          {/* Data points */}
          {result.points.map((p, i) => (
            <circle
              key={i}
              cx={toX(p.reference)}
              cy={toY(p.predicted)}
              r={4}
              fill={ZONE_COLORS[p.zone]}
              fillOpacity={0.75}
              stroke="white"
              strokeWidth={1}
            />
          ))}

          {/* Axis labels */}
          <text
            x={PAD + PLOT_W / 2} y={H - 4}
            textAnchor="middle" fontSize={11}
            fill="#6e6e73" fontFamily="DM Sans, sans-serif"
          >
            Reference glucose (mg/dL)
          </text>
          <text
            x={12} y={PAD + PLOT_H / 2}
            textAnchor="middle" fontSize={11}
            fill="#6e6e73" fontFamily="DM Sans, sans-serif"
            transform={`rotate(-90, 12, ${PAD + PLOT_H / 2})`}
          >
            Predicted glucose (mg/dL)
          </text>

          {/* n label */}
          <text
            x={PAD + PLOT_W - 4} y={PAD + 16}
            textAnchor="end" fontSize={10}
            fill="#aeaeb2" fontFamily="DM Mono, monospace"
          >
            n={result.total}
          </text>
        </svg>
      </div>
    </div>
  )
}

export default ClarkeErrorGrid