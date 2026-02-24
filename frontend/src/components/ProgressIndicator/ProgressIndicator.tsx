// Copyright (c) 2026 Harsh Dwivedi
// Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0

import { useSolverStore } from '@/store/solverStore'
import type { JobStage } from '@/types/solver'
import styles from './ProgressIndicator.module.css'

const STAGES: { key: JobStage; label: string; icon: string }[] = [
  { key: 'preprocessing',       label: 'Preprocessing',      icon: '🔧' },
  { key: 'segmentation',        label: 'Segmenting Pieces',  icon: '✂️' },
  { key: 'feature_extraction',  label: 'Extracting Features',icon: '🧠' },
  { key: 'matching',            label: 'Matching Pieces',    icon: '🎯' },
  { key: 'adjacency_refinement',label: 'Refining Placement', icon: '🔗' },
  { key: 'sequencing',          label: 'Sequencing Steps',   icon: '📋' },
  { key: 'rendering',           label: 'Rendering Output',   icon: '🖼️' },
]

const STAGE_ORDER = STAGES.map((s) => s.key)

function stageStatus(
  stageKey: JobStage,
  currentStage: JobStage | null,
): 'done' | 'active' | 'pending' {
  if (!currentStage) return 'pending'
  if (currentStage === 'done') return 'done'
  const cur = STAGE_ORDER.indexOf(currentStage)
  const idx = STAGE_ORDER.indexOf(stageKey)
  if (idx < cur) return 'done'
  if (idx === cur) return 'active'
  return 'pending'
}

export function ProgressIndicator() {
  const { progress, stage } = useSolverStore()

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h2 className={styles.title}>Solving your puzzle…</h2>
        <span className={styles.pct}>{progress}%</span>
      </div>

      <div className={styles.bar}>
        <div className={styles.fill} style={{ width: `${progress}%` }} />
      </div>

      <div className={styles.stages}>
        {STAGES.map((s) => {
          const status = stageStatus(s.key, stage)
          return (
            <div key={s.key} className={`${styles.stage} ${styles[status]}`}>
              <span className={styles.stageIcon}>
                {status === 'done' ? '✅' : s.icon}
              </span>
              <span className={styles.stageLabel}>{s.label}</span>
              {status === 'active' && (
                <span className={styles.spinner} />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}