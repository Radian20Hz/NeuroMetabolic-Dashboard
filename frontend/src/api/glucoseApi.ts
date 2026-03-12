import axios from 'axios'
import type {
  LatestReadingsResponse,
  UploadResponse,
  GlucoseStatisticsResponse,
  ClassifyResponse,
} from '../types/glucose'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1',
})

export const fetchLatestReadings = async (hours = 24): Promise<LatestReadingsResponse> => {
  const response = await api.get<LatestReadingsResponse>(`/glucose/latest?hours=${hours}`)
  return response.data
}

export const uploadCsv = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData()
  formData.append('file', file)
  const response = await api.post<UploadResponse>('/glucose/upload', formData)
  return response.data
}

export const getStatistics = async (file: File): Promise<GlucoseStatisticsResponse> => {
  const formData = new FormData()
  formData.append('file', file)
  const response = await api.post<GlucoseStatisticsResponse>('/glucose/statistics', formData)
  return response.data
}

export const classifyReading = async (glucose_mg_dl: number): Promise<ClassifyResponse> => {
  const response = await api.post<ClassifyResponse>('/glucose/classify', { glucose_mg_dl })
  return response.data
}

export const triggerScrape = async (): Promise<UploadResponse> => {
  const response = await api.post<UploadResponse>('/glucose/scrape')
  return response.data
}
