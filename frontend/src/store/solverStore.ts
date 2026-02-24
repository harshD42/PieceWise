// Copyright (c) 2026 Harsh Dwivedi
// Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0

/**
 * PieceWise — Zustand Global State
 * Single source of truth for the entire solve lifecycle.
 */

import { create } from 'zustand'
import type { AppPhase, JobStage, SolutionManifest, SolverState } from '@/types/solver'

interface SolverActions {
  setPhase: (phase: AppPhase) => void
  setJobId: (id: string) => void
  setProgress: (progress: number, stage: JobStage) => void
  setManifest: (manifest: SolutionManifest) => void
  setError: (error: string) => void
  setActiveStep: (index: number) => void
  setSelectedPiece: (pieceId: number | null) => void
  reset: () => void
}

const initialState: SolverState = {
  phase: 'upload',
  jobId: null,
  progress: 0,
  stage: null,
  error: null,
  manifest: null,
  activeStepIndex: 0,
  selectedPieceId: null,
}

export const useSolverStore = create<SolverState & SolverActions>((set) => ({
  ...initialState,

  setPhase: (phase) => set({ phase }),

  setJobId: (jobId) => set({ jobId }),

  setProgress: (progress, stage) => set({ progress, stage }),

  setManifest: (manifest) => set({ manifest, phase: 'done' }),

  setError: (error) => set({ error, phase: 'error' }),

  setActiveStep: (activeStepIndex) => set({ activeStepIndex }),

  setSelectedPiece: (selectedPieceId) => set({ selectedPieceId }),

  reset: () => set(initialState),
}))