"""Chat front door to the durable workspace engine and exact-action reviews."""
from dataclasses import asdict
import json

from .database import get_connection
from .repository import get_conversation, add_turn, list_messages
from .service import _TURN_LOCK, build_conversation_messages, conversation_config, MAX_MESSAGE_CHARS
from workspace import store, runs, context, tools


def _row(conversation_id):
    conn = get_connection()
    try:
        return conn.execute('SELECT * FROM chat_tool_turns WHERE conversation_id=? AND assistant_message_id IS NULL ORDER BY rowid DESC LIMIT 1', (conversation_id,)).fetchone()
    finally:
        conn.close()


def pending(conversation_id):
    row = _row(conversation_id)
    if not row:
        return False
    return store.run_get(row['run_id'])['status'] not in {'cancelled', 'failed', 'interrupted'}


def start(conversation_id, payload, allow_real):
    from .attachments import ground, capabilities
    with _TURN_LOCK:
        conversation = get_conversation(conversation_id)
        if not conversation:
            raise LookupError('Conversation not found')
        if pending(conversation_id):
            raise ValueError('This chat already has an unfinished workspace run')
        text = payload.get('content', '')
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_MESSAGE_CHARS:
            raise ValueError('Provide a message of 1–10,000 characters')
        selected = payload.get('tools')
        if not isinstance(selected, list) or not selected or any(x not in tools.TOOLS for x in selected):
            raise ValueError('Select tools from the workspace registry')
        selected = list(dict.fromkeys(selected))
        mock = payload.get('mock', True) is not False or not allow_real
        cfg = conversation_config(conversation.mode, conversation.model, mock)
        if conversation.mode == 'hybrid':
            raise ValueError('Workspace tools in Chat require Local or Remote mode. Choose a model explicitly before sending.')
        if not mock and not capabilities(conversation.mode, conversation.model)['tools']:
            raise ValueError('This local model does not advertise tool support. Choose an installed tool-capable model, or use a remote model that supports tools.')
        ids = payload.get('attachments', [])
        messages = build_conversation_messages(conversation_id, text, context_chars=5500, attachment_config=cfg, replay_images=not ids)
        messages, meta = ground(messages, ids, cfg, tool_mode=True)
        if any(x['text_chars'] > 700 for x in meta) and 'read_attachment' not in selected:
            raise ValueError('Enable “Read attachments” to use these documents with workspace tools')
        data = {'prompt': text, 'mode': conversation.mode, 'model': conversation.model,
                'mock': mock, 'tools': selected, 'scope': f'chat:{conversation_id}', '_chat_messages': messages}
        # Validate the bounded context before any inference or workspace side effect.
        if not mock:
            from ollama_client import local_system_prompt
            profile = 2048 if conversation.mode == 'local' and '27b' in str(cfg.local_model).lower() else 4096 if conversation.mode == 'local' else 8192
            context.native_messages({'payload': data}, selected, [], tools.TOOLS, profile,
                                    local_system_prompt(cfg.local_model) if conversation.mode == 'local' else '')
        run = runs.start('agent', data, allow_real)
        conn = get_connection()
        try:
            conn.execute('INSERT INTO chat_tool_turns(run_id,conversation_id,content,attachments) VALUES (?,?,?,?)',
                         (run['id'], conversation_id, text.strip(), json.dumps(meta)))
            conn.commit()
        except Exception:
            runs.cancel(run['id'])
            raise
        finally:
            conn.close()
        return {'run': run}


def poll(conversation_id):
    with _TURN_LOCK:
        if not get_conversation(conversation_id):
            raise LookupError('Conversation not found')
        row = _row(conversation_id)
        if not row:
            return {'run': None}
        run = store.run_get(row['run_id'])
        if run['status'] in {'completed', 'completed_with_errors'}:
            answer = run['result'].get('answer', '').strip()
            if not answer:
                raise RuntimeError('Workspace run produced no final answer; no chat turn saved')
            steps = [s for s in run['steps'] if s.get('type') == 'model']
            last = steps[-1] if steps else {}
            user, assistant = add_turn(conversation_id, row['content'], answer,
                route=last.get('route'), model_name=last.get('model'),
                latency_s=run['result'].get('usage', {}).get('active_seconds', 0),
                estimated_cost_usd=run['result'].get('usage', {}).get('cost_usd', 0),
                attachments=json.loads(row['attachments']), turn_key=run['id'])
            return {'run': run, 'completed': True, 'mock': run['payload'].get('mock', True),
                    'user_message': asdict(user), 'assistant_message': asdict(assistant)}
        return {'run': run}
