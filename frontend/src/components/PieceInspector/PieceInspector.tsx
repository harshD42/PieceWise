// Copyright (c) 2026 Harsh Dwivedi
// Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0

import { useCallback } from 'react'
import { assetUrl } from '@/api/solverApi'
import { useSolverJob } from '@/hooks/useSolverJob'
import { useSolverStore } from '@/store/solverStore'
import type { CandidateMatch, StepManifestEntry } from '@/types/solver'
import styles from './PieceInspector.module.css'

interface Props {
  step: StepManifestEntry
  onClose: () => void
}

function ScoreBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100)
  const colour = value >= 0.75 ? '#4ade80' : value >= 0.55 ? '#fbbf24' : '#f87171'
  return (
    <div className={styles.scoreRow}>
      <span className={styles.scoreLabel}>{label}</span>
      <div className={styles.scoreBar}>
        <div className={styles.scoreBarFill} style={{ width: `${pct}%`, background: colour }} />
      </div>
      <span className={styles.scoreVal} style={{ color: colour }}>{pct}%</span>
    </div>
  )
}

function CandidateCard({
  candidate,
  rank,
  jobId,
  pieceId,
}: {
  candidate: CandidateMatch
  rank: number
  jobId: string
  pieceId: number
}) {
  const { correct } = useSolverJob()

  const handleCorrect = useCallback(() => {
    correct({
      piece_id: pieceId,
      corrected_grid_pos: candidate.grid_pos,
    })
  }, [correct, pieceId, candidate.grid_pos])

  return (
    <div className={styles.candidate}>
      <div className={styles.candidateHeader}>
        <span className={styles.candidateRank}>#{rank}</span>
        <span className={styles.candidatePos}>
          Row {candidate.grid_pos[0]}, Col {candidate.grid_pos[1]}
        </span>
        <span className={styles.candidateScore}>
          {(candidate.composite_score * 100).toFixed(0)}%
        </span>
      </div>
      <button
        className={styles.correctBtn}
        onClick={handleCorrect}
        type="button"
      >
        ✓ Place here
      </button>
    </div>
  )
}

export function PieceInspector({ step, onClose }: Props) {
  const { jobId } = useSolverStore()

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <span className={styles.panelTitle}>Piece #{step.piece_id}</span>
        <button className={styles.closeBtn} onClick={onClose} type="button">✕</button>
      </div>

      {/* Piece crop thumbnail */}
      <img
        src={assetUrl(step.piece_crop_url)}
        alt={`Piece ${step.piece_id}`}
        className={styles.cropImg}
      />

      {/* Metadata */}
      <div className={styles.meta}>
        <div className={styles.metaRow}>
          <span className={styles.metaLabel}>Type</span>
          <span className={styles.metaVal}>{step.piece_type.toUpperCase()}</span>
        </div>
        <div className={styles.metaRow}>
          <span className={styles.metaLabel}>Position</span>
          <span className={styles.metaVal}>Row {step.grid_pos[0]}, Col {step.grid_pos[1]}</span>
        </div>
        <div className={styles.metaRow}>
          <span className={styles.metaLabel}>Rotation</span>
          <span className={styles.metaVal}>{step.rotation_deg}°</span>
        </div>
        <div className={styles.metaRow}>
          <span className={styles.metaLabel}>Step</span>
          <span className={styles.metaVal}>{step.step_num}</span>
        </div>
      </div>

      {/* Score bars */}
      <div className={styles.scores}>
        <ScoreBar label="Match confidence" value={step.composite_confidence} />
        <ScoreBar label="Edge compatibility" value={step.adjacency_score} />
        <ScoreBar label="Shape complement" value={step.curvature_complement_score} />
      </div>

      {/* Human-in-the-loop: top-3 candidates for flagged pieces */}
      {step.flagged && step.top3_candidates.length > 0 && jobId && (
        <div className={styles.candidates}>
          <p className={styles.candidatesTitle}>⚠ Uncertain placement — select correct cell:</p>
          {step.top3_candidates.map((c, i) => (
            <CandidateCard
              key={`${c.grid_pos[0]}-${c.grid_pos[1]}`}
              candidate={c}
              rank={i + 1}
              jobId={jobId}
              pieceId={step.piece_id}
            />
          ))}
        </div>
      )}
    </div>
  )
}