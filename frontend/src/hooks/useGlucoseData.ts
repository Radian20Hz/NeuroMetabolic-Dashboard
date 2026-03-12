import { useState, useEffect } from 'react'
import { fetchLatestReadings } from '../api/glucoseApi'
import type { GlucoseReading } from '../types/glucose'

interface UseGlucoseDataResult {
  readings: GlucoseReading[]
  loading: boolean
  error: string | null
  refetch: () => Promise<void>
}

export const useGlucoseData = (hours = 24): UseGlucoseDataResult => {
  const [readings, setReadings] = useState<GlucoseReading[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadData = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await fetchLatestReadings(hours)
      setReadings(data.readings)
    } catch {
      setError('Failed to fetch glucose data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [hours])

  return { readings, loading, error, refetch: loadData }
}
