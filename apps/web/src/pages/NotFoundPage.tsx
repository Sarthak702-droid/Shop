import { Link } from 'react-router-dom'
import { EmptyState, PageHeader } from '../components/Common'
import { IconAlert } from '../components/Icons'

export function NotFoundPage() {
  return (
    <div className="page-stack">
      <PageHeader eyebrow="Workspace" title="Page not found" />
      <section className="panel">
        <EmptyState icon={<IconAlert />} title="This local view does not exist">
          <p>
            The address did not match a page in this build. Nothing was changed in the local database.
          </p>
          <Link to="/" className="button button--secondary">Return to overview</Link>
        </EmptyState>
      </section>
    </div>
  )
}
