// Copyright (c) 2026 Harsh Dwivedi
// Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0

import { useSolverStore } from '@/store/solverStore'
import { UploadPanel } from '@/components/UploadPanel/UploadPanel'
import { ProgressIndicator } from '@/components/ProgressIndicator/ProgressIndicator'
import { SolutionViewer } from '@/components/SolutionViewer/SolutionViewer'
import styles from './App.module.css'

export default function App() {
  const { phase, error, reset } = useSolverStore()

  return (
    <div className={styles.app}>
      {phase === 'upload'  && <UploadPanel />}
      {phase === 'solving' && <ProgressIndicator />}
      {phase === 'done'    && <SolutionViewer />}
      {phase === 'error'   && (
        <div className={styles.errorScreen}>
          <p className={styles.errorIcon}>⚠️</p>
          <h2>Something went wrong</h2>
          <p className={styles.errorMsg}>{error}</p>
          <button className={styles.retryBtn} onClick={reset} type="button">
            Try Again
          </button>
        </div>
      )}
    </div>
  )
}