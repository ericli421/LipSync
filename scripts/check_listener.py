"""Check the actual same-origin listener WebSocket; no controls or media capture."""
import asyncio
import json
import sys

import websockets
from websockets.exceptions import WebSocketException


async def main():
    url = sys.argv[1] if len(sys.argv) > 1 else 'ws://127.0.0.1:3000/live/listen'
    async with websockets.connect(url, open_timeout=5) as socket:
        state = json.loads(await asyncio.wait_for(socket.recv(), 5))
        if state.get('type') not in ('waiting', 'session'):
            raise RuntimeError('Unexpected listener session response')
        for _ in range(10):
            status = json.loads(await asyncio.wait_for(socket.recv(), 5))
            if status.get('type') == 'pipeline':
                if 'source_label' not in status or 'log' in status:
                    raise RuntimeError('Listener did not return public processing status')
                print('Judge media connection verified through the frontend.')
                return
        raise RuntimeError('Listener processing status missing')


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (OSError, TimeoutError, WebSocketException, ValueError, RuntimeError) as exc:
        print(f'Judge media connection failed: {exc}. Check that the backend is running, '
              'PROCESSOR_HTTP_URL is correct, and Next can proxy /live/listen. '
              'See .tools/startup/backend-error.log and dashboard-error.log.', file=sys.stderr)
        sys.exit(1)
