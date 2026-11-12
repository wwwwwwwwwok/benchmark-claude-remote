"""A stand-in OpenAI-compatible chat server for the force_answer.py regression test (Gemma 4 work, 2026-11-12): answers
every request with the same fixed reply (with log-probabilities, so that code path runs too) and appends the raw body
of every request it receives, byte for byte, as one line of <received file>.
usage: python -I mock_chat_server.py <port> <received file>
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port, received = int(sys.argv[1]), sys.argv[2]
ANSWER = '{"risk": 42, "hazard_type": "other", "main_hazard": "test", "main_hazard_camera": "front", "reason": "test"}'


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data": []}')

    def do_POST(self):
        raw = self.rfile.read(int(self.headers['Content-Length']))
        with open(received, 'ab') as f:
            f.write(raw + b'\n')
        body = json.loads(raw)
        lp = [{'token': t, 'logprob': -0.5, 'bytes': None,
               'top_logprobs': [{'token': t, 'logprob': -0.5, 'bytes': None},
                                {'token': 'x', 'logprob': -1.5, 'bytes': None}]} for t in ('{"', 'risk', '":')]
        out = {'id': 'x', 'object': 'chat.completion', 'created': 0, 'model': body['model'],
               'choices': [{'index': 0, 'finish_reason': 'stop', 'logprobs': {'content': lp},
                            'message': {'role': 'assistant', 'content': ANSWER}}],
               'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


HTTPServer(('localhost', port), H).serve_forever()
