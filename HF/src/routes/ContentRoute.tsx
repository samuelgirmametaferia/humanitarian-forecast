import type { ReactNode } from 'react'

export function ContentRoute({ title, intro, children }: { title: string; intro?: string; children: ReactNode }) {
  return <main id="main-content" className="content-route"><header><h1>{title}</h1>{intro && <p>{intro}</p>}</header><div className="content-body">{children}</div></main>
}

export function Definition({ term, children }: { term: string; children: ReactNode }) {
  return <div className="definition"><dt>{term}</dt><dd>{children}</dd></div>
}
