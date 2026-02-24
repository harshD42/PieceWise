// Copyright (c) 2026 Harsh Dwivedi
// Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0

/**
 * PieceWise — Solver Job Hook
 * Handles the full job lifecycle:
 *   1. Submit solve request (POST /solve)
 *   2. Poll status every 1.5s (GET /status/{job_id})
 *   3. Fetch solution manifest on completion
 *   4. Submit human-in-the-loop corrections (PATCH /solve/{job_id}/correct)
 */

import { useCallback, useEffect, useRef } from 'react'
import { fetchManifest, getJobStatus, submitCorrection, submitSolve } from '@/api/solverApi'
import { useSolverStore } from '@/store/solverStore'
import type { CorrectionRequest } from '@/types/solver'

const POLL_INTERVAL_MS = 1500

export function useSolverJob() {
  const store = useSolverStore()
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  // Clean up on unmount
  useEffect(() => () => stopPolling(), [stopPolling])

  const startPolling = useCallback((jobId: string) => {
    stopPolling()

    pollRef.current = setInterval(async () => {
      try {
        const status = await getJobStatus(jobId)
        store.setProgress(status.progress, status.stage)

        if (status.status === 'done' && status.result) {
          stopPolling()
          // Fetch the full manifest
          try {
            const manifest = await fetchManifest(status.result.solution_manifest_url)
            store.setManifest(manifest)
          } catch {
            store.setError('Failed to load solution manifest. Please try again.')
          }
        } else if (status.status === 'failed') {
          stopPolling()
          store.setError(status.error ?? 'Solve job failed. Please try again.')
        }
      } catch {
        stopPolling()
        store.setError('Lost connection to server. Please refresh and try again.')
      }
    }, POLL_INTERVAL_MS)
  }, [store, stopPolling])

  const solve = useCallback(async (referenceFile: File, piecesFile: File) => {
    store.setPhase('solving')
    try {
      const response = await submitSolve(referenceFile, piecesFile)
      store.setJobId(response.job_id)
      startPolling(response.job_id)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to submit solve request.'
      store.setError(msg)
    }
  }, [store, startPolling])

  const correct = useCallback(async (correction: CorrectionRequest) => {
    const { jobId } = store
    if (!jobId) return

    store.setPhase('solving')
    store.setProgress(0, 'sequencing')

    try {
      await submitCorrection(jobId, correction)
      startPolling(jobId)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to apply correction.'
      store.setError(msg)
    }
  }, [store, startPolling])

  return { solve, correct, stopPolling }
}