// Copyright (c) 2026 Harsh Dwivedi
// Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0

import { assetUrl } from '@/api/solverApi'
import { useSolverStore } from '@/store/solverStore'
import type { StepManifestEntry } from '@/types/solver'
import styles from './SolutionViewer.module.css'

function confidenceColour(conf: number): string {
  if (conf >= 0.75) return '#4ade80'
  if (conf >= 0.55) return '#fbbf24'
  return '#f87171'
}

export function StepCard({ step }: { step: StepManifestEntry }) {
  const setSelectedPiece = useSolverStore((s) => s.setSelectedPiece)

  return (
    <div
      className={`${styles.card} ${step.flagged ? styles.cardFlagged : ''}`}
      onClick={() => setSelectedPiece(step.piece_id)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && setSelectedPiece(step.piece_id)}
    >
      <img
        src={assetUrl(step.step_card_url)}
        alt={`Step ${step.step_num}`}
        className={styles.cardImg}
      />
      <div className={styles.cardMeta}>
        <div className={styles.cardRow}>
          <span className={styles.cardStepNum}>Step {step.step_num}</span>
          <span className={styles.cardType}>{step.piece_type.toUpperCase()}</span>
          {step.flagged && <span className={styles.flagBadge}>⚠ Uncertain</span>}
        </div>
        <div className={styles.cardRow}>
          <span>Row {step.grid_pos[0]}, Col {step.grid_pos[1]}</span>
          <span>Rotate {step.rotation_deg}°</span>
        </div>
        <div className={styles.cardRow}>
          <span style={{ color: confidenceColour(step.composite_confidence) }}>
            Confidence: {(step.composite_confidence * 100).toFixed(0)}%
          </span>
          <span style={{ color: '#64748b' }}>
            Adjacency: {(step.adjacency_score * 100).toFixed(0)}%
          </span>
        </div>
      </div>
    </div>
  )
}