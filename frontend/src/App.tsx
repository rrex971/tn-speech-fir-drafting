import {
  ActionIcon,
  Alert,
  Badge,
  Button,
  Checkbox,
  Collapse,
  Divider,
  FileButton,
  Loader,
  Paper,
  ScrollArea,
  Textarea,
  Tooltip,
} from '@mantine/core'
import { useDisclosure, useInterval } from '@mantine/hooks'
import {
  IconAlertCircle,
  IconCheck,
  IconChevronDown,
  IconClipboard,
  IconCopy,
  IconFileText,
  IconMicrophone,
  IconPlayerPauseFilled,
  IconPlayerStopFilled,
  IconPrinter,
  IconRefresh,
  IconUpload,
} from '@tabler/icons-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { FirFormPreview } from './FirFormPreview'
import { LexFormPanel } from './LexFormPanel'
import type { Capabilities, DraftResponse, RecordingDetail, RecordingStatus, RecordingSummary, SpeechProcessResponse, TextTranslationResponse } from './types'

const emptyCapabilities: Capabilities = {
  speech_ready: false,
  translation_ready: false,
  text_translation_ready: false,
  classification_ready: false,
  legal_data_ready: false,
  models: { asr: 'Unavailable', translation: 'Unavailable', classifier: null },
}

function apiError(message: string): string {
  return message || 'The request could not be completed. Please try again.'
}

async function responseError(response: Response): Promise<string> {
  try {
    const body = await response.json() as { detail?: string; message?: string }
    return body.detail || body.message || `${response.status} ${response.statusText}`
  } catch {
    return `${response.status} ${response.statusText}`.trim()
  }
}

function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
}

function extensionForMime(mime: string): string {
  if (mime.includes('mp4')) return 'm4a'
  if (mime.includes('ogg')) return 'ogg'
  return 'webm'
}

function Waveform({ active }: { active: boolean }) {
  const bars = useMemo(
    () => Array.from({ length: 42 }, (_, index) => 14 + ((index * 17 + 9) % 42)),
    [],
  )

  return (
    <div className={`waveform ${active ? 'waveform-active' : ''}`} aria-hidden="true">
      {bars.map((height, index) => <i key={index} style={{ height }} />)}
    </div>
  )
}

function CopyButton({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false)
    }
  }

  return (
    <Tooltip label={copied ? 'Copied' : `Copy ${label}`}>
      <ActionIcon variant="subtle" color="gray" onClick={copy} disabled={!value} aria-label={`Copy ${label}`}>
        <IconCopy size={18} />
      </ActionIcon>
    </Tooltip>
  )
}

export function App() {
  const [capabilities, setCapabilities] = useState<Capabilities>(emptyCapabilities)
  const [recording, setRecording] = useState<RecordingStatus>('idle')
  const [elapsed, setElapsed] = useState(0)
  const [tamil, setTamil] = useState('')
  const [english, setEnglish] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [recordings, setRecordings] = useState<RecordingSummary[]>([])
  const [activeRecording, setActiveRecording] = useState<RecordingSummary | null>(null)
  const [transcriptEdited, setTranscriptEdited] = useState(false)
  const [asrStatus, setAsrStatus] = useState<SpeechProcessResponse['asr_status'] | null>(null)
  const [asrLanguage, setAsrLanguage] = useState<string | null>(null)
  const [isRetryingTranscript, setIsRetryingTranscript] = useState(false)
  const [reviewAcknowledged, setReviewAcknowledged] = useState(false)
  const [translationStatus, setTranslationStatus] = useState<SpeechProcessResponse['translation_status'] | null>(null)
  const [translationError, setTranslationError] = useState<string | null>(null)
  const [isTranslating, setIsTranslating] = useState(false)
  const [draft, setDraft] = useState<DraftResponse | null>(null)
  const [isDrafting, setIsDrafting] = useState(false)
  const [draftView, setDraftView] = useState<'form' | 'rendered' | 'markdown'>('form')
  const [reviewOpen, reviewControls] = useDisclosure(false)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const mediaRecorder = useRef<MediaRecorder | null>(null)
  const audioParts = useRef<Blob[]>([])
  const translationRequest = useRef(0)
  const draftRequest = useRef(0)
  const timer = useInterval(() => setElapsed((value) => value + 1), 1000)

  const fetchCapabilities = async () => {
    try {
      const response = await fetch('/v1/capabilities')
      if (!response.ok) throw new Error('Capability check failed')
      setCapabilities(await response.json() as Capabilities)
    } catch {
      setCapabilities(emptyCapabilities)
    }
  }

  const fetchRecordings = async () => {
    try {
      const response = await fetch('/v1/recordings')
      if (response.ok) setRecordings(await response.json() as RecordingSummary[])
    } catch {
      setRecordings([])
    }
  }

  useEffect(() => {
    void fetchCapabilities()
    void fetchRecordings()
    const interval = window.setInterval(() => {
      void fetchCapabilities()
      void fetchRecordings()
    }, 10000)
    return () => window.clearInterval(interval)
  }, [])

  useEffect(() => {
    if (recording === 'recording') timer.start()
    else timer.stop()
    return timer.stop
  }, [recording])

  useEffect(() => () => stream?.getTracks().forEach((track) => track.stop()), [stream])

  const createDraft = async (narrative: string, translation: string) => {
    if (!narrative.trim() || !translation.trim() || !reviewAcknowledged) return
    const requestId = ++draftRequest.current
    setError(null)
    setIsDrafting(true)

    try {
      const response = await fetch('/v1/complaint/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ narrative, translation_en: translation.trim(), recording_id: activeRecording?.id ?? null, annotate: true }),
      })
      if (!response.ok) throw new Error(await responseError(response))
      const result = await response.json() as DraftResponse
      if (requestId !== draftRequest.current) return
      setDraft(result)
      if (result.decision?.errors?.length) setError(result.decision.errors.join(' '))
      void fetchRecordings()
      reviewControls.open()
    } catch (requestError) {
      if (requestId === draftRequest.current) {
        setError(apiError(requestError instanceof Error ? requestError.message : ''))
      }
    } finally {
      if (requestId === draftRequest.current) setIsDrafting(false)
    }
  }

  const translateText = async (text: string, requestId: number) => {
    setIsTranslating(true)
    setTranslationError(null)
    try {
      const response = await fetch('/v1/translate/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!response.ok) throw new Error(await responseError(response))
      const result = await response.json() as TextTranslationResponse
      if (requestId !== translationRequest.current) return
      setEnglish(result.translation_en ?? '')
      setTranslationStatus(result.translation_status)
      setTranslationError(result.translation_error)
    } catch (requestError) {
      if (requestId !== translationRequest.current) return
      setTranslationStatus('failed')
      setTranslationError(requestError instanceof Error ? requestError.message : null)
    } finally {
      if (requestId === translationRequest.current) setIsTranslating(false)
    }
  }

  useEffect(() => {
    if (!tamil.trim() || activeRecording) return
    const requestId = ++translationRequest.current
    const timeout = window.setTimeout(() => void translateText(tamil, requestId), 650)
    return () => {
      window.clearTimeout(timeout)
      translationRequest.current++
    }
  }, [tamil, activeRecording])

  const processAudio = async (file: Blob, filename: string) => {
    translationRequest.current++
    draftRequest.current++
    setRecording('processing')
    setError(null)
    setDraft(null)
    setActiveRecording(null)
    setTranscriptEdited(false)
    setAsrStatus(null)
    setAsrLanguage(null)
    setReviewAcknowledged(false)
    setTamil('')
    setEnglish('')
    setIsTranslating(false)
    setTranslationStatus(null)
    setTranslationError(null)
    const form = new FormData()
    form.append('file', file, filename)

    try {
      const response = await fetch('/v1/speech/process', { method: 'POST', body: form })
      if (!response.ok) throw new Error(await responseError(response))
      const result = await response.json() as SpeechProcessResponse
      if (result.recording_id && result.recording_sha256 && result.recording_url) {
        setActiveRecording({
          id: result.recording_id,
          created_at: new Date().toISOString(),
          original_filename: filename,
          content_type: file.type,
          size_bytes: file.size,
          sha256: result.recording_sha256,
          status: 'ready',
          duration_seconds: result.duration_seconds,
          audio_url: result.recording_url,
          drafts: [],
        })
      }
      setTamil(result.transcript_ta)
      setAsrStatus(result.asr_status)
      setAsrLanguage(result.language)
      setEnglish(result.translation_en ?? '')
      setTranslationStatus(result.translation_status)
      setTranslationError(result.translation_error)
      reviewControls.open()
      void fetchRecordings()
    } catch (requestError) {
      setError(apiError(requestError instanceof Error ? requestError.message : ''))
    } finally {
      setRecording('idle')
    }
  }

  const startRecording = async () => {
    if (recording === 'paused' && mediaRecorder.current) {
      mediaRecorder.current.resume()
      setRecording('recording')
      return
    }

    try {
      const nextStream = await navigator.mediaDevices.getUserMedia({ audio: true })
      setStream(nextStream)
      audioParts.current = []
      const recorder = new MediaRecorder(nextStream)
      mediaRecorder.current = recorder
      recorder.ondataavailable = (event) => {
        if (event.data.size) audioParts.current.push(event.data)
      }
      recorder.onstop = () => {
        const blob = new Blob(audioParts.current, { type: recorder.mimeType || 'audio/webm' })
        nextStream.getTracks().forEach((track) => track.stop())
        setStream(null)
        void processAudio(blob, `complaint-recording.${extensionForMime(blob.type)}`)
      }
      recorder.start()
      setElapsed(0)
      setRecording('recording')
    } catch {
      setError('Microphone access was not granted. Upload an audio file instead.')
    }
  }

  const pauseRecording = () => {
    if (!mediaRecorder.current) return
    if (recording === 'recording') {
      mediaRecorder.current.pause()
      setRecording('paused')
    } else if (recording === 'paused') {
      mediaRecorder.current.resume()
      setRecording('recording')
    }
  }

  const stopRecording = () => {
    if (mediaRecorder.current && (recording === 'recording' || recording === 'paused')) {
      mediaRecorder.current.stop()
    }
  }

  const handleUpload = (file: File | null) => {
    if (!file) return
    setElapsed(0)
    void processAudio(file, file.name)
  }

  const selectRecording = async (recording: RecordingSummary) => {
    try {
      const response = await fetch(`/v1/recordings/${recording.id}`)
      if (!response.ok) throw new Error(await responseError(response))
      const detail = await response.json() as RecordingDetail
      translationRequest.current++
      draftRequest.current++
      setActiveRecording(recording)
      setTamil(detail.transcript_ta ?? '')
      setAsrStatus(detail.asr_status)
      setAsrLanguage(detail.language)
      setEnglish(detail.translation_en ?? '')
      setTranscriptEdited(false)
      setReviewAcknowledged(false)
      setDraft(null)
      setTranslationStatus(detail.translation_en ? 'complete' : 'failed')
      setTranslationError(null)
      reviewControls.open()
    } catch (requestError) {
      setError(apiError(requestError instanceof Error ? requestError.message : ''))
    }
  }

  const retryTranscript = async () => {
    if (!activeRecording || isRetryingTranscript) return
    setIsRetryingTranscript(true)
    setError(null)
    try {
      const response = await fetch(`/v1/recordings/${activeRecording.id}/retry-transcript`, { method: 'POST' })
      if (!response.ok) throw new Error(await responseError(response))
      const detail = await response.json() as RecordingDetail
      draftRequest.current++
      setTamil(detail.transcript_ta ?? '')
      setAsrStatus(detail.asr_status)
      setAsrLanguage(detail.language)
      setTranscriptEdited(false)
      setReviewAcknowledged(false)
      setDraft(null)
    } catch (requestError) {
      setError(apiError(requestError instanceof Error ? requestError.message : ''))
    } finally {
      setIsRetryingTranscript(false)
    }
  }

  const canRecord = recording === 'idle' || recording === 'paused'
  const canUpload = recording === 'idle'
  const downstreamReady = capabilities.classification_ready && capabilities.legal_data_ready
  const sections = draft?.decision?.suggested_sections ?? []

  return (
    <div className="app-shell">
      <main className="workspace">
        {error && (
          <Alert
            className="error-alert"
            color="red"
            icon={<IconAlertCircle size={18} />}
            withCloseButton
            onClose={() => setError(null)}
          >
            {error}
          </Alert>
        )}

        <section className="capture-grid" aria-label="Narrative capture and translation">
          <Paper className="record-panel" radius="md" withBorder>
            <div className="panel-heading"><h2>Audio</h2></div>

            <div className={`microphone-wrap ${recording === 'recording' ? 'is-listening' : ''}`}>
              <button
                className="mic-button"
                type="button"
                onClick={() => void startRecording()}
                disabled={!canRecord}
                aria-label={recording === 'paused' ? 'Resume Tamil recording' : 'Start Tamil recording'}
              >
                {recording === 'processing'
                  ? <Loader color="white" size={29} />
                  : <IconMicrophone size={34} stroke={1.8} />}
              </button>
              <span>
                {recording === 'processing'
                  ? 'Processing'
                  : recording === 'recording'
                    ? `Recording ${formatTime(elapsed)}`
                    : recording === 'paused'
                      ? `Paused ${formatTime(elapsed)}`
                      : 'Record'}
              </span>
            </div>

            <Waveform active={recording === 'recording'} />

            <div className="record-actions">
              <Button
                variant="default"
                leftSection={<IconPlayerPauseFilled size={15} />}
                onClick={pauseRecording}
                disabled={recording !== 'recording' && recording !== 'paused'}
              >
                {recording === 'paused' ? 'Resume' : 'Pause'}
              </Button>
              <Button
                variant="default"
                leftSection={<IconPlayerStopFilled size={15} />}
                onClick={stopRecording}
                disabled={recording !== 'recording' && recording !== 'paused'}
              >
                Stop
              </Button>
            </div>

            <FileButton onChange={handleUpload} accept="audio/*">
              {(fileButtonProps) => (
                <Button
                  {...fileButtonProps}
                  className="upload-button"
                  variant="light"
                  fullWidth
                  leftSection={<IconUpload size={17} />}
                  disabled={!canUpload}
                >
                  Upload audio
                </Button>
              )}
            </FileButton>
            {activeRecording && (
              <div className="saved-audio">
                <audio controls preload="metadata" src={activeRecording.audio_url} aria-label="Saved original recording" />
                <div className="saved-audio-meta">
                  <span>Original saved · {new Date(activeRecording.created_at).toLocaleString()}</span>
                  <span title={activeRecording.sha256}>SHA-256 {activeRecording.sha256.slice(0, 16)}…</span>
                  <a href={activeRecording.audio_url} download={activeRecording.original_filename}>Download original</a>
                </div>
              </div>
            )}
            {recordings.length > 0 && (
              <details className="recording-history">
                <summary>Saved recordings ({recordings.length})</summary>
                <div className="recording-history-list">
                  {recordings.map((item) => (
                    <button key={item.id} type="button" onClick={() => void selectRecording(item)}>
                      <span>{new Date(item.created_at).toLocaleString()}</span>
                      <small>{item.original_filename} · {item.id.slice(0, 8)}</small>
                    </button>
                  ))}
                </div>
              </details>
            )}
          </Paper>

          <Paper className="narrative-panel" radius="md" withBorder>
            <div className="panel-heading">
              <h2>{asrLanguage === 'en' ? 'English transcript' : 'Tamil–English transcript'}</h2>
              <CopyButton value={tamil} label="transcript" />
            </div>
            {asrStatus && asrStatus !== 'complete' && !transcriptEdited && (
              <Alert color="orange" variant="light" icon={<IconAlertCircle size={16} />} className="asr-warning">
                The speech recognizer could not produce a reliable transcript. Listen to the original and enter the words here before drafting. The English text was decoded separately from the audio.
              </Alert>
            )}
            <Textarea
              aria-label={asrLanguage === 'en' ? 'English transcript' : 'Tamil–English transcript'}
              placeholder={asrLanguage === 'en' ? 'English transcript' : 'Tamil–English transcript'}
              value={tamil}
              onChange={(event) => {
                draftRequest.current++
                setTamil(event.currentTarget.value)
                setReviewAcknowledged(false)
                if (activeRecording) {
                  setTranscriptEdited(true)
                } else {
                  setEnglish('')
                  setTranslationStatus(null)
                  setTranslationError(null)
                }
                setDraft(null)
              }}
              autosize
              minRows={12}
              maxRows={16}
              className="narrative-input tamil-input"
            />
            <div className="pane-footer">
              <span>{tamil.length} characters</span>
              {activeRecording && (
                <Button variant="subtle" size="compact-xs" loading={isRetryingTranscript} onClick={() => void retryTranscript()}>
                  Try alternate transcript
                </Button>
              )}
            </div>
          </Paper>

          <Paper className="narrative-panel" radius="md" withBorder>
            <div className="panel-heading">
              <h2>English translation</h2>
              <CopyButton value={english} label="English translation" />
            </div>
            <Textarea
              aria-label="English translation"
              placeholder="English translation"
              value={english}
              onChange={(event) => {
                translationRequest.current++
                draftRequest.current++
                setEnglish(event.currentTarget.value)
                setIsTranslating(false)
                setReviewAcknowledged(false)
                setDraft(null)
              }}
              autosize
              minRows={12}
              maxRows={16}
              className="narrative-input"
            />
            {(isTranslating || translationStatus === 'model_missing' || translationStatus === 'failed') && (
              <div className="translation-feedback">
                <span>
                  {isTranslating
                    ? 'Translating…'
                    : translationStatus === 'model_missing'
                      ? 'Translation model unavailable'
                      : `Translation failed${translationError ? `: ${translationError}` : ''}`}
                </span>
                {!isTranslating && tamil.trim() && !activeRecording && (
                  <Button
                    variant="subtle"
                    size="xs"
                    onClick={() => void translateText(tamil, ++translationRequest.current)}
                  >
                    Retry
                  </Button>
                )}
              </div>
            )}
            <div className="pane-footer">
              <span>{english.length} characters</span>
              {activeRecording && <span>{transcriptEdited ? 'Transcript edited — verify this English text' : 'Translated directly from audio'}</span>}
            </div>
          </Paper>
        </section>

        <section className="review-zone" aria-labelledby="review-heading">
          <button
            className="review-heading"
            type="button"
            onClick={reviewControls.toggle}
            aria-expanded={reviewOpen}
          >
            <span className="review-icon"><IconClipboard size={20} /></span>
            <h2 id="review-heading">Officer review</h2>
            <IconChevronDown className={reviewOpen ? 'chevron-up' : ''} size={20} />
          </button>

          <Collapse expanded={reviewOpen}>
            <Divider />
            <div className="review-content">
              {!downstreamReady ? (
                <div className="review-unavailable">
                  <IconAlertCircle size={19} />
                  <strong>Legal analysis unavailable</strong>
                  <Button
                    variant="default"
                    size="xs"
                    leftSection={<IconRefresh size={14} />}
                    onClick={() => void fetchCapabilities()}
                  >
                    Retry
                  </Button>
                </div>
              ) : (
                <>
                  <div className="review-copy">
                    <h3>Suggested sections</h3>
                    {sections.length > 0 && (
                      <ul className="section-list">
                        {sections.map((item) => (
                          <li key={`${item.bns_section}-${item.ipc_section}`}>
                            <span className="check-square"><IconCheck size={13} /></span>
                            <span>
                              <strong>{item.bns_section}</strong>{item.offence && ` — ${item.offence}`}
                              <small>IPC {item.ipc_section} · {item.cognizable.replace('_', ' ')}</small>
                            </span>
                            {item.map_confidence && <Badge variant="light" color="teal">{item.map_confidence}</Badge>}
                          </li>
                        ))}
                      </ul>
                    )}
                    <Alert variant="light" color="blue" icon={<IconAlertCircle size={16} />}>
                      Compare the recording, transcript and English text before generating a draft.
                    </Alert>
                    <Checkbox
                      className="review-check"
                      checked={reviewAcknowledged}
                      onChange={(event) => setReviewAcknowledged(event.currentTarget.checked)}
                      label={activeRecording ? 'I reviewed the recording and both texts' : 'I reviewed both texts'}
                    />
                  </div>

                  <div className="if1-preview">
                    <div className="if1-title">
                      <h3>IF-1 draft</h3>
                      {draft && (
                        <div className="draft-tools">
                          <div className="draft-view-switch" aria-label="Draft view">
                            <button type="button" className={draftView === 'form' ? 'selected' : ''} onClick={() => setDraftView('form')}>Form</button>
                            <button type="button" className={draftView === 'rendered' ? 'selected' : ''} onClick={() => setDraftView('rendered')}>Document</button>
                            <button type="button" className={draftView === 'markdown' ? 'selected' : ''} onClick={() => setDraftView('markdown')}>Markdown</button>
                          </div>
                          {draftView === 'form' && (
                            <Tooltip label="Print or save as PDF">
                              <ActionIcon variant="subtle" color="gray" onClick={() => window.print()} aria-label="Print IF-1 draft">
                                <IconPrinter size={18} />
                              </ActionIcon>
                            </Tooltip>
                          )}
                          <CopyButton value={draft.if1_markdown ?? ''} label="IF-1 Markdown" />
                        </div>
                      )}
                    </div>
                    <ScrollArea h={draft ? 700 : 120} type="auto" className="if1-document">
                      {draft ? (
                        draftView === 'form' ? (
                          <FirFormPreview draft={draft} />
                        ) : draftView === 'rendered' ? (
                          <article className="markdown-document">
                            <Markdown remarkPlugins={[remarkGfm]} skipHtml>{draft.if1_markdown}</Markdown>
                          </article>
                        ) : (
                          <pre className="markdown-source">{draft.if1_markdown}</pre>
                        )
                      ) : (
                        <div className="draft-empty">No draft generated</div>
                      )}
                    </ScrollArea>
                    <Button
                      leftSection={<IconFileText size={16} />}
                      onClick={() => void createDraft(tamil, english)}
                      loading={isDrafting}
                      disabled={!tamil.trim() || !english.trim() || !reviewAcknowledged || isDrafting}
                    >
                      {draft ? 'Regenerate draft' : 'Generate draft'}
                    </Button>
                  </div>
                </>
              )}
            </div>
          </Collapse>
        </section>
        <LexFormPanel english={english} />
      </main>
    </div>
  )
}
