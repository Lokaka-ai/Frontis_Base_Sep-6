"""Opt-in, host-only request accounting for scientific runs; never replay outcomes."""
import json
import os
import time
import uuid
from pathlib import Path


def atomic_json(path, payload):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def recorded_query(client, messages, generation_kwargs, **structured):
    root = Path(os.environ['AIRA_REQUEST_LEDGER_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    path = root / (uuid.uuid4().hex + '.json')
    record = {'schema_version':1, 'slot':os.environ.get('AIRA_CANDIDATE_SLOT'),
              'model_id':getattr(client, 'model', None), 'base_url':getattr(client, 'base_url', None),
              'messages':messages, 'generation_kwargs':generation_kwargs,
              'structured':structured, 'attempts':[], 'status':'pending'}
    atomic_json(path, record)
    # Retry only transport failures with no delivered terminal completion. The
    # provider may have generated an unseen completion; mark that ambiguity.
    import litellm
    transient = (litellm.Timeout, litellm.APIConnectionError,
                 litellm.RateLimitError, litellm.InternalServerError)
    limit = int(os.environ.get('AIRA_TRANSPORT_ATTEMPTS','3'))
    if limit < 1 or generation_kwargs.get('transport_retries') != 0:
        raise ValueError('explicit ledger requires >=1 attempt and backend retries=0')
    for index in range(limit):
        attempt = {'number':index+1, 'status':'pending', 'started_at':time.time()}
        record['attempts'].append(attempt)
        atomic_json(path, record)
        try:
            output, usage = client.query(messages, **structured, **generation_kwargs)
        except Exception as exc:
            retryable = isinstance(exc, transient) or (isinstance(exc, litellm.APIError) and 500 <= int(getattr(exc, 'status_code', 0) or 0) < 600)
            attempt.update(status='failed', error_type=type(exc).__name__, status_code=getattr(exc, 'status_code', None),
                           finished_at=time.time(), terminal_response_received=False,
                           provider_generation_may_have_occurred=retryable)
            record['status'] = 'retrying' if retryable and index+1 < limit else 'failed'
            atomic_json(path, record)
            if record['status'] == 'failed':
                raise
            time.sleep(2**(index+1))
        else:
            attempt.update(status='complete',finished_at=time.time())
            record.update(status='complete', output=output, usage=usage)
            atomic_json(path, record)
            from frontis_mila.recording import event
            event('api_complete', request_ledger_id=path.stem)
            return output, {**usage, 'request_ledger_id':path.stem,
                            'physical_request_attempts':len(record['attempts'])}
