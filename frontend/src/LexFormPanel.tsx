import { Alert, Badge, Button, Collapse, Loader, Select, Tabs, Textarea } from '@mantine/core'
import { IconChevronDown, IconCodeDots, IconRefresh } from '@tabler/icons-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import './lexform.css'

type Status = { checkpoints: { id: string; variant: string }[] }
type Clause = { text: string; quote?: string }
type Statute = {
  id: string; code: string; heading: string; text: string | null
  reference_elements: { text: string; note: string }[]
  induced_form: { groups: { alternatives: Clause[] }[]; reviewed: boolean } | null
  learned_form?: { groups: { alternatives: Clause[] }[]; reviewed: boolean } | null
}
type Prediction = {
  section: string; heading?: string; score: number; weakest_group: number; accepted: boolean
  groups: { score: number; alternatives: { predicate: number; initialization_label: string | null; membership: number }[] }[]
}

const percent = (value: number) => `${(value * 100).toFixed(1)}%`

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Could not load LexForm (${response.status}).`)
  return response.json() as Promise<T>
}

export function LexFormPanel({ english }: { english: string }) {
  const [opened, setOpened] = useState(false)
  const [activeTab, setActiveTab] = useState<string | null>('inspect')
  const [status, setStatus] = useState<Status | null>(null)
  const [statutes, setStatutes] = useState<Statute[]>([])
  const [selected, setSelected] = useState<string | null>('BNS:303')
  const [checkpoint, setCheckpoint] = useState<string | null>(null)
  const [code, setCode] = useState<string | null>('BNS')
  const [text, setText] = useState('')
  const [predictions, setPredictions] = useState<Prediction[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [inspecting, setInspecting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const request = useRef(0)
  const inspectRequest = useRef(0)

  async function refresh() {
    const id = ++request.current
    setLoading(true)
    setError(null)
    try {
      const [nextStatus, nextStatutes] = await Promise.all([
        getJson<Status>('/v1/research/lexform/status'), getJson<Statute[]>('/v1/research/lexform/statutes'),
      ])
      if (id !== request.current) return
      setStatus(nextStatus)
      setStatutes(nextStatutes)
      setCheckpoint(current => nextStatus.checkpoints.some(item => item.id === current) ? current : nextStatus.checkpoints[0]?.id ?? null)
    } catch (problem) {
      if (id === request.current) setError(problem instanceof Error ? problem.message : 'Could not load LexForm.')
    } finally {
      if (id === request.current) setLoading(false)
    }
  }

  useEffect(() => {
    if (!opened) return
    void refresh()
    return () => { request.current++ }
  }, [opened])

  useEffect(() => { inspectRequest.current++; setPredictions(null); setInspecting(false) }, [text, code, checkpoint])
  const statute = useMemo(() => statutes.find(item => `${item.code}:${item.id}` === selected), [statutes, selected])
  const program = statute?.induced_form ?? statute?.learned_form

  async function inspect() {
    if (!checkpoint || !text.trim()) return
    const id = ++inspectRequest.current
    setInspecting(true)
    setError(null)
    try {
      const response = await fetch('/v1/research/lexform/inspect', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ checkpoint_id: checkpoint, code, text: text.trim() }),
      })
      const body = await response.json() as { detail?: string; predictions?: Prediction[] }
      if (!response.ok) throw new Error(body.detail ?? 'Inspection could not be completed.')
      if (id === inspectRequest.current) setPredictions(body.predictions ?? [])
    } catch (problem) {
      if (id === inspectRequest.current) setError(problem instanceof Error ? problem.message : 'Inspection failed.')
    } finally {
      if (id === inspectRequest.current) setInspecting(false)
    }
  }

  return <section className="review-zone lex-zone" aria-labelledby="lexform-heading">
    <button className="review-heading" type="button" onClick={() => setOpened(value => !value)} aria-expanded={opened}>
      <span className="review-icon"><IconCodeDots size={20} /></span><h2 id="lexform-heading">LexForm</h2>
      <span className="lex-research-label">Element inspection</span><IconChevronDown className={opened ? 'chevron-up' : ''} size={20} />
    </button>
    <Collapse expanded={opened}><div className="lex-content">
      <Tabs value={activeTab} onChange={setActiveTab}>
        <div className="lex-toolbar"><Tabs.List aria-label="LexForm views"><Tabs.Tab value="inspect">Inspect facts</Tabs.Tab><Tabs.Tab value="statutes">Statutory forms</Tabs.Tab></Tabs.List>
          <Button size="compact-sm" variant="subtle" aria-label="Refresh LexForm" onClick={() => void refresh()} loading={loading} leftSection={<IconRefresh size={14} />}>Refresh</Button>
        </div>
        {error && <Alert color="red" variant="light" withCloseButton onClose={() => setError(null)}>{error}</Alert>}
        {!status && loading && <div className="lex-loading"><Loader size="sm" /></div>}
        {status && <Tabs.Panel value="inspect"><div className="lex-inspect">
          <div className="lex-inspect-controls"><Select label="Model" placeholder="No trained checkpoint" data={status.checkpoints.map(item => ({ value: item.id, label: `LexForm · ${item.variant === 'and_or' ? 'AND-of-ORs' : item.variant}` }))} value={checkpoint} onChange={setCheckpoint} disabled={!status.checkpoints.length} allowDeselect={false} /><Select label="Statutory code" data={['IPC', 'BNS']} value={code} onChange={setCode} allowDeselect={false} /></div>
          <Textarea label="Complaint facts" placeholder="Enter the English statement" value={text} onChange={event => setText(event.currentTarget.value)} autosize minRows={4} maxRows={10} maxLength={20000} />
          <div className="lex-inspect-actions"><Button variant="light" size="sm" disabled={!english.trim()} onClick={() => setText(english)}>Use English working text</Button><Button size="sm" loading={inspecting} disabled={!checkpoint || text.trim().length < 5} onClick={() => void inspect()}>Evaluate elements</Button></div>
          {!checkpoint && <div className="lex-no-form">No calibrated checkpoint is available.</div>}
          {predictions && <div className="lex-predictions"><div className="lex-subheading"><h3>Ranked provisions</h3><span>Suggestions only · not applied to the FIR</span></div>{predictions.map(prediction => <details key={prediction.section} className="lex-prediction"><summary><strong>{code} {prediction.section}</strong><span>Score {percent(prediction.score)}</span><span>Weakest {percent(prediction.weakest_group)}</span><Badge size="sm" color={prediction.accepted ? 'teal' : 'orange'} variant="light">{prediction.accepted ? 'Retained' : 'Abstained'}</Badge></summary>
            {prediction.heading && <h4>{prediction.heading}</h4>}
            {prediction.groups.map((group, index) => <div className="lex-predicate-evidence" key={index}><div className="lex-group-heading"><strong>Group {index + 1}</strong><span>{percent(group.score)}</span></div><div className="lex-group-meter"><span style={{ width: percent(group.score) }} /></div>{group.alternatives.map(element => <span key={element.predicate}>p{element.predicate} · {element.initialization_label ?? 'Learned prototype'}</span>)}</div>)}
            <div className="lex-prediction-footer"><small>Prototype descriptions are initialization labels, not verified explanations.</small><Button size="compact-xs" variant="subtle" onClick={() => { setSelected(`${code}:${prediction.section}`); setActiveTab('statutes') }}>View provision</Button></div>
          </details>)}</div>}
        </div></Tabs.Panel>}
        {status && <Tabs.Panel value="statutes"><div className="lex-statute-grid"><div className="lex-statute-picker"><Select label="Provision" searchable value={selected} onChange={setSelected} nothingFoundMessage="No matching section" data={statutes.map(item => ({ value: `${item.code}:${item.id}`, label: `${item.code} ${item.id}${item.heading ? ` · ${item.heading}` : ''}` }))} />
          <div className="lex-definition"><span className="lex-eyebrow">FORM TEMPLATE</span><code>AND (OR predicates)</code><p>4 groups · 256 shared predicates</p></div></div>
          {statute && <article className="lex-statute-detail"><div className="lex-subheading"><h3>{statute.code} {statute.id}</h3><Badge color={program ? 'teal' : 'gray'} variant="light">{statute.induced_form ? 'Source decomposition' : statute.learned_form ? 'Learned program' : statute.reference_elements.length ? 'Reference elements' : 'Statutory text'}</Badge></div>
            {statute.heading && <h4>{statute.heading}</h4>}
            {program ? <div className="lex-logical-form">{program.groups.map((group, index) => <div key={index}>{index > 0 && <div className="lex-operator">AND</div>}<div className="lex-clause-group">{group.alternatives.map((clause, clauseIndex) => <div key={clauseIndex}>{clauseIndex > 0 && <span className="lex-or">OR</span>}<strong>{clause.text}</strong>{clause.quote && <q>{clause.quote}</q>}</div>)}</div></div>)}<small>{statute.induced_form ? 'Source-quoted decomposition · awaiting legal review' : 'Learned prototype program · descriptions identify initialization, not verified legal ingredients'}</small></div>
              : statute.reference_elements.length > 0 && <div className="lex-logical-form">{statute.reference_elements.map((element, index) => <div key={index}>{index > 0 && <div className="lex-operator">AND</div>}<div className="lex-clause-group"><strong>{element.text}</strong>{element.note && <small>{element.note}</small>}</div></div>)}</div>}
            {statute.text && <details className="lex-source" open={!program}><summary>Statutory text</summary><p>{statute.text}</p></details>}
          </article>}
        </div></Tabs.Panel>}
      </Tabs>
    </div></Collapse>
  </section>
}
