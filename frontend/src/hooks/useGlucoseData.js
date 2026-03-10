import { useState, useEffect } from 'react'
import { fetchLatestReadings } from '../api/glucoseApi'

export const useGlucoseData = (hours = 24) => {
  const [readings, setReadings] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true)
        const data = await fetchLatestReadings(hours)
        setReadings(data.readings)
      } catch (err) {
        setError('Failed to fetch glucose data')
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [hours])

  return { readings, loading, error }
}