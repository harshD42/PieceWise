// Copyright (c) 2026 Harsh Dwivedi
// Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0

/**
 * PieceWise — TypeScript types mirroring backend Pydantic models.
 * Keep in sync with:
 *   backend/app/models/piece.py
 *   backend/app/models/job.py
 *   backend/app/models/output.py
 */

// ─── Enums ───────────────────────────────────────────────────────────────────

export type PieceType = 'corner' | 'edge' | 'interior' | 'unknown'

export type JobStatus = 'pending' | 'running' | 'done' | 'failed'

export type JobStage =
  | 'queued'
  | 'preprocessing'
  | 'segmentation'
  | 'feature_extraction'
  | 'matching'
  | 'adjacency_refinement'
  | 'sequencing'
  | 'rendering'
  | 'done'
  | 'failed'

// ─── Matching ────────────────────────────────────────────────────────────────

export interface CandidateMatch {
  grid_pos: [number, number]
  rotation_deg: number
  spatial_score: number
  flat_side_score: number
  composite_score: number
}

// ─── Solution Manifest ───────────────────────────────────────────────────────

export interface StepManifestEntry {
  step_num: number
  piece_id: number
  grid_pos: [number, number]
  rotation_deg: number
  piece_type: PieceType
  composite_confidence: number
  adjacency_score: number
  curvature_complement_score: number
  flagged: boolean
  piece_crop_url: string
  step_card_url: string
  top3_candidates: CandidateMatch[]
}

export interface SolutionManifest {
  job_id: string
  grid_shape: [number, number]
  total_pieces: number
  flagged_count: number
  mean_confidence: number
  min_confidence: number
  max_confidence: number
  steps: StepManifestEntry[]
  asset_urls: {
    overlay_reference: string
    overlay_pieces: string
    solution_manifest: string
    step_cards: string[]
  }
  corner_count: number
  edge_count: number
  interior_count: number
}

// ─── Job / API ────────────────────────────────────────────────────────────────

export interface OutputBundle {
  overlay_reference_url: string
  overlay_pieces_url: string
  solution_manifest_url: string
  step_card_urls: string[]
  total_pieces: number
  flagged_count: number
  mean_confidence: number
}

export interface JobStatusResponse {
  job_id: string
  status: JobStatus
  stage: JobStage
  progress: number
  error: string | null
  result?: OutputBundle
}

export interface SolveResponse {
  job_id: string
  status: JobStatus
  message: string
}

export interface CorrectionRequest {
  piece_id: number
  corrected_grid_pos: [number, number]
}

export interface CorrectionResponse {
  job_id: string
  piece_id: number
  corrected_grid_pos: [number, number]
  message: string
}

// ─── UI State ─────────────────────────────────────────────────────────────────

export interface UploadedImage {
  file: File
  previewUrl: string
}

export type AppPhase =
  | 'upload'      // waiting for both images
  | 'solving'     // job running, polling status
  | 'done'        // solution ready to view
  | 'error'       // unrecoverable error

export interface SolverState {
  phase: AppPhase
  jobId: string | null
  progress: number
  stage: JobStage | null
  error: string | null
  manifest: SolutionManifest | null
  activeStepIndex: number          // 0-based index into manifest.steps
  selectedPieceId: number | null   // for PieceInspector panel
}