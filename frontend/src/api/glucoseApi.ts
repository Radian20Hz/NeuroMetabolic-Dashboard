import axios from 'axios'
import type {
  LatestReadingsResponse,
  UploadResponse,
  GlucoseStatisticsResponse,
  ClassifyResponse,
  PredictResponse,
} from '../types/glucose'

const api = axios.create({
  baseURL: ( import.meta as unknown as { env: Record<string, string> }).env.VITE_API_URL || 'http://localhost:8000/api/v1',
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

export const fetchPrediction = async (
  glucose_mg_dl: number[],
  subject_id = '559'
): Promise<PredictResponse> => {
  const response = await api.post<PredictResponse>('/predict', {
    glucose_mg_dl,
    subject_id,
  })
  return response.data
}

export interface ClarkeRequest {
  reference_values: number[]
  predicted_values: number[]
}

export interface ClarkePointAPI {
  reference: number
  predicted: number
  zone: string
}

export interface ClarkeResponseAPI {
  total: number
  zone_counts: Record<string, number>
  zone_percents: Record<string, number>
  clinically_acceptable_percent: number
  points: ClarkePointAPI[]
}

export const fetchClarke = async (req: ClarkeRequest): Promise<ClarkeResponseAPI> => {
  const response = await api.post<ClarkeResponseAPI>('/clarke', req)
  return response.data
}