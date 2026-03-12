// Shared TypeScript interfaces for NMD frontend
// Mirrors backend Pydantic models in backend/app/models/glucose.py

export interface GlucoseReading {
  timestamp: string
  glucose_mg_dl: number
  units: string
}

export interface UploadResponse {
  status: string
  readings_saved: number
  min_glucose: number | null
  max_glucose: number | null
  avg_glucose: number | null
  std_dev: number | null
  time_in_range_percent: number | null
  gmi: number | null
  cv_percent: number | null
  cv_is_stable: boolean | null
}

export interface LatestReadingsResponse {
  status: string
  hours_requested: number
  count: number
  readings: GlucoseReading[]
}

export interface GlucoseStatisticsResponse {
  count: number
  min_glucose: number | null
  max_glucose: number | null
  avg_glucose: number | null
  std_dev: number | null
  time_in_range_percent: number
  gmi: number | null
  cv_percent: number | null
  cv_is_stable: boolean | null
}

export interface ClassifyResponse {
  glucose_mg_dl: number
  zone: string
  is_critical: boolean
  message: string
}

// Derived stats shape used internally by StatsCards
export interface ComputedStats {
  min_glucose: number
  max_glucose: number
  avg_glucose: number
  std_dev: number | null
  time_in_range_percent: number
  gmi: number | null
  cv_percent: number | null
  cv_is_stable: boolean | null
  count: number
}
