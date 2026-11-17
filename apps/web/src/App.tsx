import { useEffect } from 'react'
import { Outlet, Route, Routes, useLocation } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { OverviewPage } from './pages/OverviewPage'
import { ReviewQueuePage } from './pages/ReviewQueuePage'
import { UploadPage } from './pages/UploadPage'
import { JobPage } from './pages/JobPage'
import { CamerasPage } from './pages/CamerasPage'
import { ModelsPage } from './pages/ModelsPage'
import { SettingsPage } from './pages/SettingsPage'
import { NotFoundPage } from './pages/NotFoundPage'

function Layout() {
  const location = useLocation()
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'auto' })
  }, [location.pathname])
  return (
    <AppShell>
      <Outlet />
    </AppShell>
  )
}

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<OverviewPage />} />
        <Route path="review" element={<ReviewQueuePage />} />
        <Route path="review/:eventId" element={<ReviewQueuePage />} />
        <Route path="upload" element={<UploadPage />} />
        <Route path="jobs/:jobId" element={<JobPage />} />
        <Route path="cameras" element={<CamerasPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  )
}
