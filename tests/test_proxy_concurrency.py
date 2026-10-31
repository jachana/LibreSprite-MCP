"""Deterministic single-flight regressions using the real Flask relay routes."""
import asyncio
import json
import threading
import unittest
from unittest.mock import Mock, patch

from libresprite_mcp.libresprite_proxy import LibrespriteProxy
from libresprite_mcp.mcp_server import MCPServer


class ProxyConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.proxy = LibrespriteProxy(port=0)
        self.release = threading.Event()
        self.threads = []

    def tearDown(self):
        self.release.set()
        # Also unblock the broken pre-fix implementation after an assertion fails.
        for _ in range(4):
            self.proxy._output_event.set()
            for thread in self.threads:
                thread.join(timeout=0.1)

    def start_call(self, script, ctx):
        result = {}

        def call():
            try:
                result['output'] = self.proxy.run_script(script, ctx)
            except Exception as exc:
                result['error'] = exc

        thread = threading.Thread(target=call, daemon=True)
        self.threads.append(thread)
        thread.start()
        return thread, result

    def reply(self, script, output):
        self.assertTrue(self.proxy._script_event.wait(timeout=2))
        with self.proxy.app.test_client() as client:
            self.assertEqual(client.get('/').json, {'script': script})
            self.assertEqual(client.post('/', json={'output': output}).json,
                             {'status': 'success'})

    def test_overlapping_admission_rejects_second_call_without_overwrite(self):
        entered = threading.Event()

        def pause_before_publish(message):
            if message.startswith('Sending'):
                entered.set()
                if not self.release.wait(timeout=3):
                    raise TimeoutError('Test did not release the first caller')

        first_ctx = Mock()
        first_ctx.info.side_effect = pause_before_publish
        first, first_result = self.start_call('first script', first_ctx)
        self.assertTrue(entered.wait(timeout=2))
        second, second_result = self.start_call('second script', Mock())
        second.join(timeout=0.5)
        self.assertFalse(second.is_alive(), 'Overlapping caller waited for an unrelated reply')
        self.assertIsInstance(second_result.get('error'), RuntimeError)
        self.assertIn('already in progress', str(second_result['error']))

        self.release.set()
        self.reply('first script', 'first output')
        first.join(timeout=2)
        self.assertFalse(first.is_alive())
        self.assertEqual(first_result, {'output': 'first output'})

        third, third_result = self.start_call('next script', Mock())
        self.reply('next script', 'next output')
        third.join(timeout=2)
        self.assertFalse(third.is_alive())
        self.assertEqual(third_result, {'output': 'next output'})
        with self.proxy.app.test_client() as client:
            self.assertEqual(client.post('/', json={'output': 'unsolicited'}).json,
                             {'status': 'ignored'})

    def test_busy_call_does_not_consume_active_reply(self):
        first, first_result = self.start_call('active script', Mock())
        self.assertTrue(self.proxy._script_event.wait(timeout=2))
        second, second_result = self.start_call('rejected script', Mock())
        second.join(timeout=0.5)
        self.assertFalse(second.is_alive())
        self.assertIsInstance(second_result.get('error'), RuntimeError)
        self.reply('active script', 'active output')
        first.join(timeout=2)
        self.assertFalse(first.is_alive())
        self.assertEqual(first_result, {'output': 'active output'})

    def test_admission_is_released_when_pre_publish_logging_raises(self):
        ctx = Mock()
        ctx.info.side_effect = ValueError('logging failed')
        with self.assertRaisesRegex(ValueError, 'logging failed'):
            self.proxy.run_script('never published', ctx)
        self.assertFalse(self.proxy._script_event.is_set())
        thread, result = self.start_call('next script', Mock())
        self.reply('next script', 'next output')
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, {'output': 'next output'})

    def test_mcp_tools_share_admission_in_both_directions(self):
        for first_name, second_name in [('read_pixel', 'run_script'),
                                        ('run_script', 'read_pixel')]:
            with self.subTest(first=first_name, second=second_name):
                self.release.clear()
                entered = threading.Event()
                first_ctx = Mock()

                def pause(message):
                    if message.startswith('Sending'):
                        entered.set()
                        if not self.release.wait(timeout=3):
                            raise TimeoutError('Test did not release MCP caller')

                first_ctx.info.side_effect = pause
                contexts = iter([first_ctx, Mock()])
                run_script = self.proxy.run_script
                server = MCPServer(self.proxy)

                def relay(script, ctx):
                    # Only replace logging; admission, events and HTTP routes are real.
                    return run_script(script, next(contexts))

                def start_tool(name):
                    result = {}

                    def call():
                        arguments = {'x': 1, 'y': 2} if name == 'read_pixel' else {'script': 'generic script'}
                        try:
                            result['content'] = asyncio.run(server.mcp.call_tool(name, arguments))
                        except Exception as exc:
                            result['error'] = exc

                    thread = threading.Thread(target=call, daemon=True)
                    self.threads.append(thread)
                    thread.start()
                    return thread, result

                with patch.object(self.proxy, 'run_script', side_effect=relay):
                    first, first_result = start_tool(first_name)
                    self.assertTrue(entered.wait(timeout=2))
                    second, second_result = start_tool(second_name)
                    second.join(timeout=0.5)
                    self.assertFalse(second.is_alive(), 'MCP caller waited instead of reporting busy')
                    self.assertIn('already in progress', str(second_result.get('error')))
                    self.release.set()
                    self.assertTrue(self.proxy._script_event.wait(timeout=2))
                    output = (json.dumps({'x': 1, 'y': 2, 'color_mode': 'indexed',
                                          'value': 7, 'palette_index': 7})
                              if first_name == 'read_pixel' else 'generic output')
                    self.reply(self.proxy._script, output)
                    first.join(timeout=2)
                    self.assertFalse(first.is_alive())
                    self.assertNotIn('error', first_result)
                    content = first_result['content']
                    if isinstance(content, tuple):
                        content = content[0]
                    if first_name == 'read_pixel':
                        self.assertEqual(json.loads(content[0].text), json.loads(output))
                    else:
                        self.assertEqual(content[0].text, output)


if __name__ == '__main__':
    unittest.main()
