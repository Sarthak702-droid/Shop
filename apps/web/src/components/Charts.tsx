import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { formatMediaTime, reasonLabel } from '../lib/format'
import type { ActionLabel, ActionTimeline } from '../lib/types'

/*
 * Small SVG chart kit for the research views.
 *
 * Colour: categorical slots 1-5 of the validated reference palette, assigned to the five MERL
 * classes in fixed order (never by rank). Light-mode slots 3-5 are under 3:1 contrast on white,
 * so every chart here carries direct labels and a table view (relief rule). Text always uses ink
 * tokens, never series colours.
 */
export const ACTION_COLORS: Record<ActionLabel, string> = {
  reach_to_shelf: '#2a78d6',
  retract_from_shelf: '#eb6834',
  hand_in_shelf: '#1baf7a',
  inspect_product: '#eda100',
  inspect_shelf: '#e87ba4',
}
const SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
const INK = '#142b3b'
const INK_2 = '#536572'
const GRID = '#e4eaed'
const AXIS = '#c3d0d6'

export function useWidth<T extends HTMLElement>(fallback = 720): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(fallback)
  useEffect(() => {
    const node = ref.current
    if (!node) return
    const measure = () => setWidth(Math.max(280, Math.floor(node.getBoundingClientRect().width) || fallback))
    measure()
    if (typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(measure)
    observer.observe(node)
    return () => observer.disconnect()
  }, [fallback])
  return [ref, width]
}

export function Legend({ items }: { items: Array<{ key: string; label: string; color: string; dashed?: boolean }> }) {
  return (
    <ul className="chart-legend" aria-label="Legend">
      {items.map((item) => (
        <li key={item.key}>
          <span
            className={`chart-legend__swatch${item.dashed ? ' chart-legend__swatch--dashed' : ''}`}
            style={item.dashed ? { borderColor: item.color } : { background: item.color }}
            aria-hidden="true"
          />
          {item.label}
        </li>
      ))}
    </ul>
  )
}

function Tooltip({ x, y, width, children }: { x: number; y: number; width: number; children: ReactNode }) {
  const left = Math.min(Math.max(8, x + 14), width - 220)
  return (
    <div className="chart-tooltip" style={{ left, top: Math.max(0, y - 8) }} role="status">
      {children}
    </div>
  )
}

export function ChartFrame({ title, detail, legend, children, table }: { title?: string; detail?: string; legend?: ReactNode; children: ReactNode; table?: ReactNode }) {
  const [showTable, setShowTable] = useState(false)
  return (
    <figure className="chart-frame">
      {(title || table) && (
        <figcaption className="chart-frame__head">
          <div>
            {title && <strong>{title}</strong>}
            {detail && <p>{detail}</p>}
          </div>
          {table && (
            <button type="button" className="button button--ghost button--small" onClick={() => setShowTable((v) => !v)} aria-pressed={showTable}>
              {showTable ? 'Show chart' : 'Show table'}
            </button>
          )}
        </figcaption>
      )}
      {legend}
      {showTable && table ? <div className="table-scroll">{table}</div> : children}
    </figure>
  )
}

/* ---------------------------------------------------------------- action timeline (swimlanes) */

export function ActionTimelineChart({
  timeline,
  onSelectTime,
  highlight,
}: {
  timeline: ActionTimeline
  onSelectTime?: (ms: number) => void
  highlight?: { start_ms: number; end_ms: number } | null
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<number | null>(null)
  const labelW = 132
  const laneH = 38
  const visH = 14
  const top = 8
  const axisH = 24
  const plotW = Math.max(120, width - labelW - 12)
  const n = timeline.pts_ms.length
  const endMs = n ? timeline.pts_ms[n - 1] + timeline.step_ms : 1
  const x = (ms: number) => labelW + (ms / endMs) * plotW
  const lanes = timeline.classes
  const height = top + visH + 10 + lanes.length * laneH + axisH

  const paths = useMemo(() => {
    // Downsample to ~2 points per pixel so long videos stay light.
    const stride = Math.max(1, Math.floor(n / (plotW * 2)))
    return lanes.map((_, k) => {
      const y0 = top + visH + 10 + k * laneH + laneH - 6
      const h = laneH - 12
      let d = `M${x(timeline.pts_ms[0] ?? 0).toFixed(1)},${y0}`
      for (let i = 0; i < n; i += stride) {
        d += `L${x(timeline.pts_ms[i]).toFixed(1)},${(y0 - timeline.scores[k][i] * h).toFixed(1)}`
      }
      d += `L${x(endMs).toFixed(1)},${y0}Z`
      return d
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timeline, plotW])

  const segments = useMemo(() => {
    const out: Array<{ k: number; start: number; end: number }> = []
    for (const ep of timeline.episodes) {
      for (const a of ep.actions) out.push({ k: lanes.indexOf(a.label), start: a.start_ms, end: a.end_ms })
    }
    return out
  }, [timeline, lanes])

  const ticks = useMemo(() => {
    const target = Math.max(2, Math.floor(plotW / 90))
    const raw = endMs / target
    const steps = [1000, 2000, 5000, 10000, 15000, 30000, 60000, 120000, 300000, 600000]
    const step = steps.find((s) => s >= raw) ?? 600000
    const out = []
    for (let t = 0; t <= endMs; t += step) out.push(t)
    return out
  }, [endMs, plotW])

  const hoverIndex = hover === null ? null : Math.min(n - 1, Math.max(0, Math.round(((hover - labelW) / plotW) * endMs / timeline.step_ms)))

  const onMove = (event: React.MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const px = event.clientX - rect.left
    setHover(px >= labelW && px <= labelW + plotW ? px : null)
  }

  return (
    <div className="chart-canvas" ref={ref}>
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`Shelf-action score timeline over ${formatMediaTime(endMs)} with ${timeline.episodes.length} interaction episodes`}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        onClick={() => {
          if (hoverIndex !== null && onSelectTime) onSelectTime(timeline.pts_ms[hoverIndex])
        }}
        style={{ cursor: onSelectTime ? 'pointer' : 'default' }}
      >
        {/* person visibility strip */}
        <text x={labelW - 8} y={top + visH - 3} textAnchor="end" fontSize={11} fill={INK_2}>Person visible</text>
        <rect x={labelW} y={top} width={plotW} height={visH} fill="#f1f4f6" rx={3} />
        {timeline.person_present.map((v, i) =>
          v ? null : (
            <rect key={i} x={x(timeline.pts_ms[i])} y={top} width={Math.max(0.6, (timeline.step_ms / endMs) * plotW)} height={visH} fill="#c3d0d6" />
          ),
        )}
        {highlight && (
          <rect x={x(highlight.start_ms)} y={top} width={Math.max(2, x(highlight.end_ms) - x(highlight.start_ms))} height={height - top - axisH} fill="#087f78" opacity={0.08} />
        )}
        {lanes.map((label, k) => {
          const laneTop = top + visH + 10 + k * laneH
          const y0 = laneTop + laneH - 6
          const h = laneH - 12
          const thr = timeline.thresholds[k]
          return (
            <g key={label}>
              <text x={labelW - 8} y={laneTop + laneH / 2 + 4} textAnchor="end" fontSize={12} fill={INK}>
                {timeline.display[label] ?? reasonLabel(label)}
              </text>
              <line x1={labelW} x2={labelW + plotW} y1={y0} y2={y0} stroke={GRID} />
              <path d={paths[k]} fill={ACTION_COLORS[label]} fillOpacity={0.28} stroke={ACTION_COLORS[label]} strokeWidth={1.5} />
              <line x1={labelW} x2={labelW + plotW} y1={y0 - thr * h} y2={y0 - thr * h} stroke={INK_2} strokeDasharray="3 3" strokeWidth={1} />
              {segments.filter((s) => s.k === k).map((s, i) => (
                <rect key={i} x={x(s.start)} y={laneTop + 1} width={Math.max(2, x(s.end) - x(s.start))} height={4} rx={2} fill={ACTION_COLORS[label]} />
              ))}
            </g>
          )
        })}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={height - axisH} y2={height - axisH + 4} stroke={AXIS} />
            <text x={x(t)} y={height - 6} textAnchor="middle" fontSize={11} fill={INK_2}>{formatMediaTime(t)}</text>
          </g>
        ))}
        {hover !== null && <line x1={hover} x2={hover} y1={top} y2={height - axisH} stroke={INK} strokeWidth={1} opacity={0.5} />}
      </svg>
      {hoverIndex !== null && hover !== null && (
        <Tooltip x={hover} y={top + visH + 10} width={width}>
          <strong>{formatMediaTime(timeline.pts_ms[hoverIndex])}</strong>
          <span className="chart-tooltip__muted">{timeline.person_present[hoverIndex] ? 'Person detected' : 'No person detected'}</span>
          {lanes.map((label, k) => (
            <span key={label} className="chart-tooltip__row">
              <span className="chart-legend__swatch" style={{ background: ACTION_COLORS[label] }} aria-hidden="true" />
              {timeline.display[label] ?? label}
              <b>{timeline.scores[k][hoverIndex].toFixed(2)}</b>
              {timeline.scores[k][hoverIndex] >= timeline.thresholds[k] && <em>above threshold</em>}
            </span>
          ))}
          {onSelectTime && <span className="chart-tooltip__muted">Click to open the nearest episode</span>}
        </Tooltip>
      )}
    </div>
  )
}

/* ---------------------------------------------------------------- line chart (single scale) */

export function LineChart({
  series,
  xLabel,
  yLabel,
  yDomain,
  height = 220,
  marker,
}: {
  series: Array<{ key: string; label: string; color: string; points: Array<[number, number]>; dashed?: boolean }>
  xLabel: string
  yLabel: string
  yDomain?: [number, number]
  height?: number
  marker?: { x: number; label: string }
}) {
  const [ref, width] = useWidth<HTMLDivElement>(480)
  const [hover, setHover] = useState<number | null>(null)
  const m = { l: 48, r: 96, t: 12, b: 36 }
  const all = series.flatMap((s) => s.points)
  if (!all.length) return <p className="muted-copy">No data.</p>
  const xs = all.map((p) => p[0])
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)]
  const ys = all.map((p) => p[1])
  const [y0, y1] = yDomain ?? [Math.min(0, ...ys), Math.max(...ys) * 1.05]
  const pw = width - m.l - m.r
  const ph = height - m.t - m.b
  const sx = (v: number) => m.l + ((v - x0) / Math.max(1e-9, x1 - x0)) * pw
  const sy = (v: number) => m.t + ph - ((v - y0) / Math.max(1e-9, y1 - y0)) * ph
  const yTicks = Array.from({ length: 5 }, (_, i) => y0 + ((y1 - y0) * i) / 4)
  const xStep = Math.max(1, Math.ceil((x1 - x0) / Math.max(2, Math.floor(pw / 60))))
  const xTicks: number[] = []
  for (let v = Math.ceil(x0); v <= x1; v += xStep) xTicks.push(v)
  const hoverX = hover === null ? null : Math.round(x0 + ((hover - m.l) / pw) * (x1 - x0))
  return (
    <div className="chart-canvas" ref={ref}>
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`${yLabel} by ${xLabel}`}
        onMouseMove={(e) => {
          const px = e.clientX - e.currentTarget.getBoundingClientRect().left
          setHover(px >= m.l && px <= m.l + pw ? px : null)
        }}
        onMouseLeave={() => setHover(null)}
      >
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={m.l + pw} y1={sy(t)} y2={sy(t)} stroke={GRID} />
            <text x={m.l - 6} y={sy(t) + 4} textAnchor="end" fontSize={11} fill={INK_2}>{t.toFixed(2)}</text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} x={sx(t)} y={height - m.b + 16} textAnchor="middle" fontSize={11} fill={INK_2}>{t}</text>
        ))}
        <text x={m.l + pw / 2} y={height - 4} textAnchor="middle" fontSize={11} fill={INK_2}>{xLabel}</text>
        {marker && (
          <g>
            <line x1={sx(marker.x)} x2={sx(marker.x)} y1={m.t} y2={m.t + ph} stroke={INK_2} strokeDasharray="4 3" />
            <text x={sx(marker.x) + 4} y={m.t + 10} fontSize={11} fill={INK_2}>{marker.label}</text>
          </g>
        )}
        {series.map((s) => {
          const d = s.points.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join('')
          const last = s.points[s.points.length - 1]
          return (
            <g key={s.key}>
              <path d={d} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? '5 4' : undefined} />
              {series.length <= 6 && last && (
                <text x={sx(last[0]) + 6} y={sy(last[1]) + 4} fontSize={11} fill={INK}>{s.label}</text>
              )}
            </g>
          )
        })}
        {hover !== null && <line x1={hover} x2={hover} y1={m.t} y2={m.t + ph} stroke={INK} opacity={0.4} />}
      </svg>
      {hover !== null && hoverX !== null && (
        <Tooltip x={hover} y={m.t} width={width}>
          <strong>{xLabel} {hoverX}</strong>
          {series.map((s) => {
            const p = s.points.reduce((best, q) => (Math.abs(q[0] - hoverX) < Math.abs(best[0] - hoverX) ? q : best), s.points[0])
            return (
              <span key={s.key} className="chart-tooltip__row">
                <span className="chart-legend__swatch" style={{ background: s.color }} aria-hidden="true" />
                {s.label}<b>{p[1].toFixed(3)}</b>
              </span>
            )
          })}
        </Tooltip>
      )}
    </div>
  )
}

/* ---------------------------------------------------------------- per-class dot plot with reference ticks */

export function ClassDotPlot({
  rows,
  series,
  domain = [0, 1],
  format = (v: number) => v.toFixed(2),
}: {
  rows: Array<{ key: string; label: string; values: Record<string, number | null>; interval?: [number, number] | null }>
  series: Array<{ key: string; label: string; color: string; shape?: 'dot' | 'tick' }>
  domain?: [number, number]
  format?: (v: number) => string
}) {
  const [ref, width] = useWidth<HTMLDivElement>(480)
  const [hover, setHover] = useState<{ row: number; px: number; py: number } | null>(null)
  const labelW = 136
  const rowH = 34
  const m = { t: 8, b: 26, r: 16 }
  const pw = width - labelW - m.r
  const height = m.t + rows.length * rowH + m.b
  const sx = (v: number) => labelW + ((v - domain[0]) / (domain[1] - domain[0])) * pw
  const ticks = Array.from({ length: 6 }, (_, i) => domain[0] + ((domain[1] - domain[0]) * i) / 5)
  return (
    <div className="chart-canvas" ref={ref}>
      <svg width={width} height={height} role="img" aria-label="Per-class comparison" onMouseLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={m.t} y2={height - m.b} stroke={GRID} />
            <text x={sx(t)} y={height - 8} textAnchor="middle" fontSize={11} fill={INK_2}>{format(t)}</text>
          </g>
        ))}
        {rows.map((row, i) => {
          const cy = m.t + i * rowH + rowH / 2
          return (
            <g key={row.key} onMouseMove={(e) => {
              const rect = e.currentTarget.ownerSVGElement?.getBoundingClientRect()
              setHover({ row: i, px: e.clientX - (rect?.left ?? 0), py: cy })
            }}>
              <rect x={0} y={cy - rowH / 2} width={width} height={rowH} fill={hover?.row === i ? '#f3f6f8' : 'transparent'} />
              <text x={labelW - 8} y={cy + 4} textAnchor="end" fontSize={12} fill={INK}>{row.label}</text>
              {row.interval && (
                <line x1={sx(row.interval[0])} x2={sx(row.interval[1])} y1={cy} y2={cy} stroke={INK_2} strokeWidth={2} strokeLinecap="round" />
              )}
              {series.map((s) => {
                const v = row.values[s.key]
                if (v === null || v === undefined || !Number.isFinite(v)) return null
                return s.shape === 'tick' ? (
                  <line key={s.key} x1={sx(v)} x2={sx(v)} y1={cy - 9} y2={cy + 9} stroke={s.color} strokeWidth={2} />
                ) : (
                  <circle key={s.key} cx={sx(v)} cy={cy} r={5.5} fill={s.color} stroke="#fff" strokeWidth={2} />
                )
              })}
            </g>
          )
        })}
      </svg>
      {hover && (
        <Tooltip x={hover.px} y={hover.py} width={width}>
          <strong>{rows[hover.row].label}</strong>
          {series.map((s) => {
            const v = rows[hover.row].values[s.key]
            return (
              <span key={s.key} className="chart-tooltip__row">
                <span className="chart-legend__swatch" style={{ background: s.color }} aria-hidden="true" />
                {s.label}<b>{v === null || v === undefined ? '—' : format(v)}</b>
              </span>
            )
          })}
          {rows[hover.row].interval && (
            <span className="chart-tooltip__muted">95% interval {format(rows[hover.row].interval![0])}–{format(rows[hover.row].interval![1])}</span>
          )}
        </Tooltip>
      )}
    </div>
  )
}

/* ---------------------------------------------------------------- heatmap (confusion matrix) */

export function Heatmap({ rows, cols, values, counts }: { rows: string[]; cols: string[]; values: number[][]; counts?: number[][] }) {
  const [hover, setHover] = useState<[number, number] | null>(null)
  const color = (v: number) => SEQ[Math.min(SEQ.length - 1, Math.max(0, Math.round(v * (SEQ.length - 1))))]
  return (
    <div className="heatmap" role="table" aria-label="Confusion matrix">
      <div className="heatmap__row heatmap__row--head" role="row">
        <span className="heatmap__corner" role="columnheader">Annotated ↓ / Predicted →</span>
        {cols.map((c) => <span key={c} role="columnheader" className="heatmap__col">{reasonLabel(c)}</span>)}
      </div>
      {rows.map((r, i) => (
        <div className="heatmap__row" role="row" key={r}>
          <span className="heatmap__rowhead" role="rowheader">{reasonLabel(r)}</span>
          {values[i].map((v, j) => (
            <span
              key={j}
              role="cell"
              className={`heatmap__cell${i === j ? ' heatmap__cell--diag' : ''}${hover && hover[0] === i && hover[1] === j ? ' heatmap__cell--hover' : ''}`}
              style={{ background: color(v), color: v > 0.45 ? '#ffffff' : INK }}
              onMouseEnter={() => setHover([i, j])}
              onMouseLeave={() => setHover(null)}
              title={`${reasonLabel(r)} → ${reasonLabel(cols[j])}: ${(v * 100).toFixed(0)}%${counts ? ` (${counts[i][j].toLocaleString()} steps)` : ''}`}
            >
              {(v * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      ))}
    </div>
  )
}

/* ---------------------------------------------------------------- simple vertical bars (histogram) */

export function Bars({ items, color = '#2a78d6', format = (v: number) => String(v) }: { items: Array<{ label: string; value: number }>; color?: string; format?: (v: number) => string }) {
  const [ref, width] = useWidth<HTMLDivElement>(360)
  const [hover, setHover] = useState<number | null>(null)
  const height = 160
  const m = { l: 8, r: 8, t: 18, b: 28 }
  const max = Math.max(1, ...items.map((i) => i.value))
  const bw = (width - m.l - m.r) / Math.max(1, items.length)
  return (
    <div className="chart-canvas" ref={ref}>
      <svg width={width} height={height} role="img" aria-label="Bar chart" onMouseLeave={() => setHover(null)}>
        <line x1={m.l} x2={width - m.r} y1={height - m.b} y2={height - m.b} stroke={AXIS} />
        {items.map((item, i) => {
          const h = ((height - m.t - m.b) * item.value) / max
          const x = m.l + i * bw + 2
          const w = Math.max(2, bw - 4)
          const y = height - m.b - h
          return (
            <g key={item.label} onMouseEnter={() => setHover(i)}>
              <rect x={m.l + i * bw} y={m.t} width={bw} height={height - m.t - m.b} fill="transparent" />
              {h > 0 && <path d={`M${x},${height - m.b}V${y + Math.min(4, h)}Q${x},${y} ${x + 4},${y}H${x + w - 4}Q${x + w},${y} ${x + w},${y + Math.min(4, h)}V${height - m.b}Z`} fill={color} opacity={hover === null || hover === i ? 1 : 0.55} />}
              <text x={x + w / 2} y={y - 4} textAnchor="middle" fontSize={11} fill={INK}>{format(item.value)}</text>
              <text x={x + w / 2} y={height - 10} textAnchor="middle" fontSize={11} fill={INK_2}>{item.label}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
