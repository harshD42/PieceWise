// Copyright (c) 2026 Harsh Dwivedi
// Licensed under the Harsh Non-Commercial Attribution License (HNCAL) v1.0

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)