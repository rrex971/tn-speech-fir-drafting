from __future__ import annotations
from contextlib import asynccontextmanager
import importlib.util
import os
import threading
import time
from pathlib import Path
from typing import Annotated, Any, Literal
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fir.config import DATA_DIR, REPO_ROOT, pipeline
from fir.schema.if1 import FirRecord
from fir.schema.statute_decision import StatuteDecision
from fir.serving import recordings
from fir.statute.ipc_bns_map import IpcBnsMap
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
MAX_AUDIO_BYTES = 25 * 1024 * 1024
_AUDIO_SUFFIXES = {'.wav', '.mp3', '.m4a', '.mp4', '.webm', '.ogg', '.opus', '.flac'}
_AUDIO_MIME_SUFFIX = {'audio/webm': '.webm', 'video/webm': '.webm', 'audio/mp4': '.m4a', 'audio/x-m4a': '.m4a', 'audio/ogg': '.ogg', 'audio/opus': '.opus', 'audio/wav': '.wav', 'audio/x-wav': '.wav', 'audio/wave': '.wav', 'audio/mpeg': '.mp3', 'audio/flac': '.flac', 'audio/x-flac': '.flac'}
_FRONTEND_DIST = REPO_ROOT / 'frontend' / 'dist'

class CapabilityModels(BaseModel):
    asr: str
    translation: str
    classifier: str | None

class Capabilities(BaseModel):
    speech_ready: bool
    translation_ready: bool
    text_translation_ready: bool
    classification_ready: bool
    legal_data_ready: bool
    models: CapabilityModels

class Segment(BaseModel):
    start: float
    end: float
    text: str

class SpeechModels(BaseModel):
    asr: str
    translation: str | None

class SpeechTiming(BaseModel):
    transcription: int
    translation: int
    total: int

class SpeechResponse(BaseModel):
    transcript_ta: str
    transcript_raw_ta: str
    translation_en: str | None
    language: str
    language_probability: float
    asr_flags: list[str]
    asr_status: Literal['complete', 'needs_review', 'failed']
    asr_retried: bool = False
    duration_seconds: float
    segments: list[Segment]
    models: SpeechModels
    timing_ms: SpeechTiming
    translation_status: Literal['complete', 'not_needed', 'model_missing', 'failed']
    translation_error: str | None = None
    recording_id: str | None = None
    recording_sha256: str | None = None
    recording_url: str | None = None

class TextComplaint(BaseModel):
    narrative: str = Field(min_length=1)
    translation_en: str | None = None
    recording_id: str | None = None
    annotate: bool = True

class TextTranslation(BaseModel):
    text: str = Field(min_length=1)

class TranslationResponse(BaseModel):
    translation_en: str | None
    translation_status: Literal['complete', 'not_needed', 'model_missing', 'failed']
    translation_error: str | None = None
    model: str | None = None

class DraftResponse(BaseModel):
    decision: StatuteDecision
    record: FirRecord
    if1_text: str
    if1_markdown: str
    narrative_grounded: bool
    narrative_en: str | None = None
    translation: dict[str, Any] = Field(default_factory=dict)
    asr_flags: list[str] = Field(default_factory=list)
    fill_report: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: int
    recording_id: str | None = None
    disclaimer: str = 'Draft for officer review. The officer remains the author of record.'

def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

def _model_cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(model_id, 'config.json'), str)
    except Exception:
        return False

def _audio_suffix(filename: str | None, content_type: str | None) -> str:
    suffix = Path(filename or '').suffix.lower()
    if suffix in _AUDIO_SUFFIXES:
        return suffix
    mime = (content_type or '').split(';')[0].strip().lower()
    if mime in _AUDIO_MIME_SUFFIX:
        return _AUDIO_MIME_SUFFIX[mime]
    raise HTTPException(status_code=415, detail='Unsupported audio format')

class SpeechRuntime:

    def __init__(self) -> None:
        self._asr = None
        self._audio_translator = None
        self._translator = None
        self._lock = threading.Lock()

    def asr(self):
        if self._asr is None:
            from fir.asr.whisper import WhisperAsr
            self._asr = WhisperAsr()
        return self._asr

    def translator(self):
        if self._translator is None:
            from fir.translate.indictrans import IndicTranslator
            self._translator = IndicTranslator()
        return self._translator

    def audio_translator(self):
        if self._audio_translator is None:
            from fir.asr.whisper import WhisperAsr
            self._audio_translator = WhisperAsr()
        return self._audio_translator

    def _translate_text(self, text: str) -> TranslationResponse:
        translator = self.translator()
        if not translator.should_translate(text):
            return TranslationResponse(translation_en=text, translation_status='not_needed')
        if not translator.available():
            return TranslationResponse(translation_en=None, translation_status='model_missing')
        try:
            result = translator.translate(text, force=True)
            return TranslationResponse(
                translation_en=result.target if result else text,
                translation_status='complete',
                model=result.model if result else translator.model_id,
            )
        except Exception as exc:
            return TranslationResponse(
                translation_en=None,
                translation_status='failed',
                translation_error=f'{type(exc).__name__}: {exc}',
            )

    def translate_text(self, text: str) -> TranslationResponse:
        with self._lock:
            return self._translate_text(text.strip())

    def retry_transcript(self, path: Path):
        with self._lock:
            return self.asr().transcribe(path, force_rescue=True)

    def process(self, path: Path) -> SpeechResponse:
        from fir.asr.itn import normalise_numbers
        started = time.perf_counter()
        with self._lock:
            transcription_started = time.perf_counter()
            transcript = None
            asr_error = None
            try:
                transcript = self.asr().transcribe(path)
            except Exception as exc:
                asr_error = f'{type(exc).__name__}: {exc}'
            transcription_ms = int((time.perf_counter() - transcription_started) * 1000)
            raw_text = transcript.text.strip() if transcript else ''
            asr_flags = transcript.flags if transcript else ['failed']
            asr_status = 'failed' if transcript is None else 'needs_review' if asr_flags else 'complete'
            normalized_text = normalise_numbers(raw_text) if asr_status == 'complete' else ''
            translation_started = time.perf_counter()
            try:
                translated = self.audio_translator().translate_audio(path, language_hint=transcript.language if transcript else None)
                translation_en = translated.text.strip() or None
                translation_status = 'complete' if translation_en else 'failed'
                translation_error = None if translation_en else 'Whisper returned an empty English translation'
            except Exception as exc:
                translation_en = None
                translation_status = 'failed'
                translation_error = f'{type(exc).__name__}: {exc}'
            translation_ms = int((time.perf_counter() - translation_started) * 1000)
        total_ms = int((time.perf_counter() - started) * 1000)
        if not transcript and not translation_en:
            raise RuntimeError(f'Both speech passes failed: {asr_error}; {translation_error}')
        return SpeechResponse(transcript_ta=normalized_text, transcript_raw_ta=raw_text, translation_en=translation_en, language=transcript.language if transcript else '', language_probability=transcript.language_probability if transcript else 0.0, asr_flags=asr_flags, asr_status=asr_status, asr_retried=bool(transcript and transcript.retried), duration_seconds=transcript.duration if transcript else translated.duration, segments=[Segment(**segment) for segment in transcript.segments] if transcript else [], models=SpeechModels(asr=self.asr().model_name, translation=self.asr().model_name), timing_ms=SpeechTiming(transcription=transcription_ms, translation=translation_ms, total=total_ms), translation_status=translation_status, translation_error=translation_error)

class DraftRuntime:

    def __init__(self) -> None:
        self._graph = None
        self._lock = threading.Lock()

    def graph(self):
        if self._graph is None:
            from fir.orchestrator.graph import build_slice_graph
            from fir.statute import registry
            classifier = registry.load()
            self._graph = build_slice_graph(classifier=classifier, asr=None, ipc_map=IpcBnsMap.load(), translator=speech_runtime.translator())
        return self._graph

    def run(self, narrative: str, annotate: bool, translation_en: str | None = None, recording_id: str | None = None) -> DraftResponse:
        from fir.instantiate.markdown import render_if1_markdown
        from fir.instantiate.render import render_if1
        from fir.orchestrator.graph import to_decision
        if recording_id:
            recordings.get(recording_id)
        started = time.perf_counter()
        with self._lock:
            state = self.graph().invoke({'narrative': narrative, 'provided_translation_en': translation_en or '', 'errors': []})
        if state.get('errors') and (not state.get('fir_record')):
            raise HTTPException(status_code=500, detail='; '.join(state['errors']))
        record = FirRecord.model_validate(state['fir_record'])
        response = DraftResponse(decision=to_decision(state), record=record, if1_text=render_if1(record, annotate=annotate) if annotate else state['if1_text'], if1_markdown=render_if1_markdown(record, annotate=annotate, source_ta=narrative, source_en=translation_en or '', from_audio=bool(recording_id)), narrative_grounded=bool(state.get('narrative_grounded')), narrative_en=state.get('narrative_en') if state.get('translation') else None, translation=state.get('translation') or {}, asr_flags=list(state.get('asr_flags') or []), fill_report=state.get('fill_report') or {}, elapsed_ms=int((time.perf_counter() - started) * 1000), recording_id=recording_id)
        from fir.serving import audit
        audit.append(audit.entry_for(response))
        if recording_id:
            recordings.add_draft(recording_id, record.record_id, narrative, translation_en or '')
        return response
speech_runtime = SpeechRuntime()
draft_runtime = DraftRuntime()
warm_errors: dict[str, str] = {}

@asynccontextmanager
async def lifespan(_app: FastAPI):
    for name, load in (
        ('recordings', recordings._ensure_dir),
        ('asr', lambda: speech_runtime.asr().load()),
        ('audio_translation', lambda: speech_runtime.audio_translator().load()),
        ('text_translation', lambda: speech_runtime.translator().load() if speech_runtime.translator().available() else None),
        ('classifier', draft_runtime.graph),
        ('lexform', lambda: __import__('research.lexform.api', fromlist=['warm_checkpoint']).warm_checkpoint()),
    ):
        try:
            load()
        except Exception as exc:
            warm_errors[name] = f'{type(exc).__name__}: {exc}'
            print(f'==> startup {name} unavailable: {warm_errors[name]}', flush=True)
    yield

app = FastAPI(title='TN FIR Speech', version='0.1.0', description='Tamil speech, translation and officer-assisted IF-1 drafting prototype', lifespan=lifespan)

from research.lexform.api import router as lexform_router
app.include_router(lexform_router)

@app.get('/v1/capabilities', response_model=Capabilities)
def capabilities() -> Capabilities:
    from fir.statute import registry
    from fir.translate.indictrans import IndicTranslator
    cfg = pipeline()
    asr_model = str(cfg['asr']['model'])
    trained_classifier_ready = registry.available().get('tfidf', False) and all((_module_available(name) for name in ('joblib', 'sklearn', 'numpy')))
    classification_ready = registry.available().get('rule', False) and _module_available('langgraph')
    speech_ready = _module_available('faster_whisper') and _model_cached(asr_model) and 'asr' not in warm_errors
    text_translation_ready = IndicTranslator().available() and all((_module_available(name) for name in ('torch', 'transformers', 'sentencepiece'))) and 'text_translation' not in warm_errors
    legal_data_ready = all((path.exists() for path in (DATA_DIR / 'mapping' / 'ipc_bns_map.csv', DATA_DIR / 'statutes' / 'bns_2023.jsonl', DATA_DIR / 'statutes' / 'bnss_schedule1.csv')))
    classifier = 'tfidf+ovr-logistic' if trained_classifier_ready else 'rule-cue'
    return Capabilities(speech_ready=speech_ready, translation_ready=speech_ready and 'audio_translation' not in warm_errors, text_translation_ready=text_translation_ready, classification_ready=classification_ready, legal_data_ready=legal_data_ready, models=CapabilityModels(asr=asr_model, translation=asr_model, classifier=classifier if classification_ready else None))

@app.get('/v1/health')
def health() -> dict[str, Any]:
    caps = capabilities()
    return {'status': 'ok' if not warm_errors else 'degraded', 'capabilities': caps.model_dump(), 'local_only': True, 'stores_recordings': True, 'model_warm_errors': warm_errors, 'audit_entries': __import__('fir.serving.audit', fromlist=['count']).count()}

@app.post('/v1/speech/process', response_model=SpeechResponse)
def process_speech(file: Annotated[UploadFile, File(description='Tamil or Tamil-English audio')]) -> SpeechResponse:
    suffix = _audio_suffix(file.filename, file.content_type)
    stored = recordings.store_upload(file, suffix, MAX_AUDIO_BYTES)
    try:
        response = speech_runtime.process(recordings.audio_path(stored['id']))
        recordings.update(stored['id'], status='ready', duration_seconds=response.duration_seconds, transcript_ta=response.transcript_ta, transcript_raw_ta=response.transcript_raw_ta, language=response.language, asr_flags=response.asr_flags, asr_status=response.asr_status, asr_retried=response.asr_retried, translation_en=response.translation_en, translation_status=response.translation_status, translation_error=response.translation_error, models=response.models.model_dump(), timing_ms=response.timing_ms.model_dump())
        return response.model_copy(update={'recording_id': stored['id'], 'recording_sha256': stored['sha256'], 'recording_url': f"/v1/recordings/{stored['id']}/audio"})
    except HTTPException:
        raise
    except Exception as exc:
        recordings.update(stored['id'], status='failed', error=f'{type(exc).__name__}: {exc}')
        raise HTTPException(status_code=503, detail=f'Speech pipeline unavailable: {exc}') from exc

@app.get('/v1/recordings')
def list_recordings() -> list[dict[str, Any]]:
    return [recordings.public_metadata(item) for item in recordings.list_recordings()]

@app.get('/v1/recordings/{recording_id}')
def get_recording(recording_id: str) -> dict[str, Any]:
    from fir.asr.quality import assess
    metadata = recordings.get(recording_id)
    raw_text = metadata.get('transcript_raw_ta') or metadata.get('transcript_ta') or ''
    flags = metadata.get('asr_flags')
    if flags is None:
        flags = assess(raw_text).flags
        flags = [flag.value for flag in flags]
    asr_status = metadata.get('asr_status') or ('needs_review' if flags else 'complete')
    transcript = metadata.get('transcript_ta') if asr_status == 'complete' else ''
    return {**recordings.public_metadata(metadata), 'transcript_ta': transcript, 'transcript_raw_ta': raw_text, 'language': metadata.get('language'), 'asr_flags': flags, 'asr_status': asr_status, 'asr_retried': metadata.get('asr_retried', False), 'translation_en': metadata.get('translation_en')}

@app.get('/v1/recordings/{recording_id}/audio')
def play_recording(recording_id: str) -> FileResponse:
    metadata = recordings.get(recording_id)
    path = recordings.audio_path(recording_id)
    return FileResponse(path, media_type=metadata['content_type'], headers={'Cache-Control': 'private, no-store, max-age=0', 'X-Content-Type-Options': 'nosniff'}, filename=metadata['original_filename'], content_disposition_type='inline')

@app.post('/v1/recordings/{recording_id}/retry-transcript')
def retry_recording_transcript(recording_id: str) -> dict[str, Any]:
    from fir.asr.itn import normalise_numbers
    path = recordings.audio_path(recording_id)
    try:
        transcript = speech_runtime.retry_transcript(path)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'Transcription unavailable: {exc}') from exc
    flags = transcript.flags
    status = 'needs_review' if flags else 'complete'
    raw = transcript.text.strip()
    recordings.revise_transcript(recording_id, transcript_ta=normalise_numbers(raw) if status == 'complete' else '', transcript_raw_ta=raw, language=transcript.language, asr_flags=flags, asr_status=status, asr_retried=True)
    return get_recording(recording_id)

@app.post('/v1/translate/text', response_model=TranslationResponse)
def translate_text(body: TextTranslation) -> TranslationResponse:
    return speech_runtime.translate_text(body.text)

@app.post('/v1/complaint/text', response_model=DraftResponse)
def analyze_complaint(body: TextComplaint) -> DraftResponse:
    try:
        return draft_runtime.run(body.narrative.strip(), body.annotate, body.translation_en, body.recording_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@app.get('/v1/schema/if1')
def if1_schema() -> dict[str, Any]:
    import json
    path = REPO_ROOT / 'src' / 'fir' / 'schema' / 'if1_v1.schema.json'
    return json.loads(path.read_text(encoding='utf-8'))
if (_FRONTEND_DIST / 'assets').exists():
    app.mount('/assets', StaticFiles(directory=_FRONTEND_DIST / 'assets'), name='assets')

@app.get('/', include_in_schema=False)
def index():
    page = _FRONTEND_DIST / 'index.html'
    if page.exists():
        return FileResponse(page)
    return JSONResponse({'message': 'Frontend is not built', 'next': 'Run `cd frontend && npm install && npm run build`, then restart the API', 'api_docs': '/docs'})
