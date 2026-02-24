// Copyright (c) 2026 Harsh Dwivedi
// Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0

import { useCallback, useEffect } from 'react'
import { useSolverStore } from '@/store/solverStore'
import { assetUrl } from '@/api/solverApi'
import { StepCard } from './StepCard'
import { GridOverlay } from './GridOverlay'
import { PieceInspector } from '@/components/PieceInspector/PieceInspector'
import styles from './SolutionViewer.module.css'

export function SolutionViewer() {
  const {
    manifest,
    activeStepIndex,
    selectedPieceId,
    setActiveStep,
    setSelectedPiece,
    reset,
  } = useSolverStore()

  if (!manifest) return null

  const steps = manifest.steps
  const currentStep = steps[activeStepIndex]
  const totalSteps = steps.length

  // Keyboard navigation
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        setActiveStep(Math.min(activeStepIndex + 1, totalSteps - 1))
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        setActiveStep(Math.max(activeStepIndex - 1, 0))
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [activeStepIndex, totalSteps, setActiveStep])

  const handleGridCellClick = useCallback((row: number, col: number) => {
    const idx = steps.findIndex((s) => s.grid_pos[0] === row && s.grid_pos[1] === col)
    if (idx >= 0) {
      setActiveStep(idx)
      setSelectedPiece(steps[idx].piece_id)
    }
  }, [steps, setActiveStep, setSelectedPiece])

  const selectedStep = selectedPieceId !== null
    ? steps.find((s) => s.piece_id === selectedPieceId) ?? null
    : null

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div>
          <h2 className={styles.title}>🧩 Solution Ready</h2>
          <p className={styles.meta}>
            {manifest.total_pieces} pieces · {manifest.corner_count} corners ·{' '}
            {manifest.edge_count} edges · {manifest.flagged_count} uncertain
          </p>
        </div>
        <div className={styles.headerActions}>
          <a
            href={assetUrl(manifest.asset_urls.overlay_reference)}
            download="overlay_reference.jpg"
            className={styles.downloadBtn}
          >
            ⬇ Reference
          </a>
          <a
            href={assetUrl(manifest.asset_urls.overlay_pieces)}
            download="overlay_pieces.jpg"
            className={styles.downloadBtn}
          >
            ⬇ Pieces
          </a>
          <button className={styles.resetBtn} onClick={reset} type="button">
            New Puzzle
          </button>
        </div>
      </div>

      <div className={styles.body}>
        {/* Left: grid overlay */}
        <div className={styles.gridSection}>
          <p className={styles.sectionLabel}>Reference — click a cell to jump to step</p>
          <GridOverlay
            overlayUrl={assetUrl(manifest.asset_urls.overlay_reference)}
            steps={steps}
            activeStepIndex={activeStepIndex}
            onCellClick={handleGridCellClick}
          />
        </div>

        {/* Centre: step card + navigation */}
        <div className={styles.cardSection}>
          <p className={styles.sectionLabel}>
            Step {activeStepIndex + 1} of {totalSteps}
            <span className={styles.keyHint}> ← → to navigate</span>
          </p>
          <StepCard step={currentStep} />

          <div className={styles.nav}>
            <button
              className={styles.navBtn}
              onClick={() => setActiveStep(Math.max(0, activeStepIndex - 1))}
              disabled={activeStepIndex === 0}
              type="button"
            >
              ← Prev
            </button>
            <input
              type="number"
              className={styles.jumpInput}
              value={activeStepIndex + 1}
              min={1}
              max={totalSteps}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10)
                if (!isNaN(v) && v >= 1 && v <= totalSteps) {
                  setActiveStep(v - 1)
                }
              }}
            />
            <button
              className={styles.navBtn}
              onClick={() => setActiveStep(Math.min(totalSteps - 1, activeStepIndex + 1))}
              disabled={activeStepIndex === totalSteps - 1}
              type="button"
            >
              Next →
            </button>
          </div>
        </div>

        {/* Right: piece inspector */}
        {selectedStep && (
          <div className={styles.inspectorSection}>
            <p className={styles.sectionLabel}>Piece Inspector</p>
            <PieceInspector
              step={selectedStep}
              onClose={() => setSelectedPiece(null)}
            />
          </div>
        )}
      </div>
    </div>
  )
}