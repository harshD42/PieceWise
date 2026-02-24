// Copyright (c) 2026 Harsh Dwivedi
// Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0

/**
 * PieceWise — Typed API Client
 * All HTTP calls to the FastAPI backend go through this module.
 */

import axios from 'axios'
import type {
  CorrectionRequest,
  CorrectionResponse,
  JobStatusResponse,
  SolutionManifest,
  SolveResponse,
} from '@/types/solver'

const api = axios.create({
  baseURL: '/',
  timeout: 30_000,
})

// ─── Submit a solve job ───────────────────────────────────────────────────────

export async function submitSolve(
  referenceImage: File,
  piecesImage: File,
): Promise<SolveResponse> {
  const form = new FormData()
  form.append('reference_image', referenceImage)
  form.append('pieces_image', piecesImage)

  const { data } = await api.post<SolveResponse>('/solve', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

// ─── Poll job status ──────────────────────────────────────────────────────────

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const { data } = await api.get<JobStatusResponse>(`/status/${jobId}`)
  return data
}

// ─── Fetch solution manifest ──────────────────────────────────────────────────

export async function fetchManifest(
  manifestUrl: string,
): Promise<SolutionManifest> {
  const { data } = await api.get<SolutionManifest>(manifestUrl)
  return data
}

// ─── Apply human-in-the-loop correction ──────────────────────────────────────

export async function submitCorrection(
  jobId: string,
  correction: CorrectionRequest,
): Promise<CorrectionResponse> {
  const { data } = await api.patch<CorrectionResponse>(
    `/solve/${jobId}/correct`,
    correction,
  )
  return data
}

// ─── Asset URL helper ─────────────────────────────────────────────────────────

export function assetUrl(path: string): string {
  // Relative URLs from the manifest are served directly by the Vite proxy
  return path.startsWith('/') ? path : `/${path}`
}