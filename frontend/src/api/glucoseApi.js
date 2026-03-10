import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
})

export const fetchLatestReadings = async (hours = 24) => {
  const response = await api.get(`/glucose/latest?hours=${hours}`)
  return response.data
}

export const uploadCsv = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  const response = await api.post('/glucose/upload', formData)
  return response.data
}