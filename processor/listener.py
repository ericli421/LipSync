"""Read-only audience projection of the existing final media stream."""
import time

from processor.bridge import Outbox


class ListenerOutbox(Outbox):
    """No private diagnostics or controls; reuse JPEGs at at most eight FPS."""
    def __init__(self):
        super().__init__()
        self.last_video = None

    def reset(self, status):
        self.last_video = None
        super().reset(status)

    def put(self, message):
        kind = message.get('type')
        if kind in ('session', 'waiting'):
            allowed = ('type', 'protocol_version', 'session_id', 'sample_rate', 'channels', 'format', 'mode')
            public = {key: message[key] for key in allowed if key in message}
        elif kind == 'audio':
            public = {key: message[key] for key in ('type', 'session_id', 'seq', 'first_sample', 'pcm_base64')}
        elif kind == 'video':
            now = time.monotonic()
            if self.last_video is not None and now-self.last_video < 1/8:
                return
            self.last_video = now
            public = {key: message[key] for key in ('type', 'session_id', 'seq', 'jpeg_base64')}
        elif kind == 'pipeline':
            test = message.get('test') or {}
            live = test.get('live') or {}
            enabled = bool(test.get('enabled'))
            source = test.get('monitor', 'original') if enabled and test.get('state') == 'running' else 'original'
            labels = {'original': 'Original microphone mix', 'voice1': 'Voice 1',
                      'voice2': 'Voice 2', 'focus': 'Focus · selected person'}
            public = {'type': 'pipeline', 'source': source,
                      'source_label': labels.get(source, 'Presenter output'),
                      'source_delay_ms': live.get('delay_ms', 0) if source != 'original' else 0,
                      'separation': enabled, 'state': test.get('state', 'off'),
                      'layers': {key: bool(value) for key, value in message.get('layers', {}).items()
                                 if key in ('voice_band', 'auto_volume')}}
            if message.get('session_id'):
                public['session_id'] = message['session_id']
        elif kind == 'discontinuity':
            public = {key: message[key] for key in ('type', 'session_id') if key in message}
        else:
            return
        super().put(public)
