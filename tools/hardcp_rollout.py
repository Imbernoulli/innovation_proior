#!/usr/bin/env python3
"""Escalating rejection-sampling rollout driver for the hard-CP self-distillation pipeline.

For each query in a domain worklist, sample from the served model with a DOUBLING budget
(4 -> 8 -> 16 -> ... -> 1024), stopping as soon as ONE verifier-passing generation appears.
Keep the passing generation(s) + how many samples it took (the difficulty label). Give up on a
query only after 1024 samples yield nothing.

Design (matches experiments/DATA_FIX_ROUND2 autopsy): the SFT regression is under-reasoning +
wrong-idea on hard problems; the fix is the model's OWN verifier-passing long-reasoning traces on
frontier problems. This driver harvests exactly those.

Two adapters isolate the moving parts:
  - MODEL endpoint: OpenAI-compatible /v1/chat/completions (sglang or vllm). URL+model read from
    data_v4/_hardcp/server.json {"url":..., "model":...} or --url/--model.
  - VERIFIER: each domain dir data_v4/_hardcp/<domain>/verify.py exposes
    verify(generation_text, problem) -> {"passed": bool, "detail": str}.

Worklist: data_v4/_hardcp/<domain>/worklist.jsonl, one problem object per line (must have an 'id'
and the fields the domain's verify.py + prompt builder expect).

Resumable: a query whose result already exists in the output shard is skipped.
Usage: python tools/hardcp_rollout.py --domains math,code,reasoning,ifollow [--url ... --model ...]
"""
import argparse, asyncio, errno, json, os, sys, tempfile, time, zlib, importlib.util

REPO = '/srv/home/bohanlyu/innovation_proior'
HARDCP = os.path.join(REPO, 'data_v4', '_hardcp')
SCRATCH = '/tmp/claude-2065/-srv-home-bohanlyu-innovation-proior/6ed8424a-6c58-40da-8be5-c4e3e3548d9b/scratchpad'
os.makedirs(SCRATCH, exist_ok=True)
os.environ.setdefault('TMPDIR', SCRATCH)
tempfile.tempdir = os.environ['TMPDIR']
# Stop-at-first-pass escalation. The refined worklists are pre-filtered HARD, so we collect fast
# (resolve on the first verified pass) and apply the >50% drop POST-HOC on the kept set (re-measure
# only the solved problems). Round 0 (4) verifies all to record a cheap first-round rate signal.
BUDGET_SCHEDULE = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
GLOBAL_HTTP_CONCURRENCY = 64

# ---------- adapters ----------
def load_server(url, model):
    p = os.path.join(HARDCP, 'server.json')
    if (not url or not model) and os.path.isfile(p):
        cfg = json.load(open(p))
        url = url or cfg.get('url')
        model = model or cfg.get('model')
    if not url or not model:
        sys.exit('need --url and --model (or data_v4/_hardcp/server.json with {url, model})')
    return url.rstrip('/'), model

def load_backend(args):
    """Return a backend config: local vLLM 27B, or Poe qwen3.7-max for the hard tail.
    Local supports MULTIPLE endpoints (3 independent TP=2 services); a query is pinned to ONE of them
    (crc32(id) % N) so all its samples hit the same replica and reuse the prompt's prefix-cached KV."""
    if args.backend == 'poe':
        key = open(os.path.join(HARDCP, '.poe_key')).read().strip()
        return {'chat_urls': ['https://api.poe.com/v1/chat/completions'], 'model': args.poe_model,
                'headers': {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
                'schedule': [1, 2, 4, 8, 16], 'suffix': '.poe', 'stream': True}
    if args.backend == 'deepseek':
        # Tier-2 solver for the 27B's genuine hard-failures. OpenAI-compatible; reasoning in
        # 'reasoning_content' (already handled by the streaming/non-streaming parsers). Strong model,
        # so a shallow schedule 1..8 is enough. Writes traces/<domain>.deepseek.jsonl (source-tagged).
        key = open(os.path.join(HARDCP, '.deepseek_key')).read().strip()
        return {'chat_urls': ['https://api.deepseek.com/chat/completions'], 'model': args.deepseek_model,
                'headers': {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
                'schedule': [1, 2, 4, 8], 'suffix': '.deepseek', 'stream': True}
    p = os.path.join(HARDCP, 'server.json')
    cfg = json.load(open(p)) if os.path.isfile(p) else {}
    urls = (args.url.split(',') if args.url else None) or cfg.get('urls') or [cfg.get('url', 'http://127.0.0.1:30000')]
    model = args.model or cfg.get('model', 'Qwen3.6-27B')
    return {'chat_urls': [f'{u.rstrip("/")}/v1/chat/completions' for u in urls], 'model': model,
            'headers': {'Content-Type': 'application/json'},
            'schedule': [4, 8, 16, 32, 64, 128, 256, 512, 1024], 'suffix': ''}

def load_verifier(domain):
    p = os.path.join(HARDCP, domain, 'verify.py')
    spec = importlib.util.spec_from_file_location(f'verify_{domain}', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, 'verify'), f'{p} missing verify()'
    return mod.verify

# ---------- per-domain prompt building (elicit the format the verifier extracts) ----------
def build_messages(domain, prob):
    if domain == 'math':
        sys_p = ("You are an expert mathematician. Solve the problem. Think step by step, then give "
                 "the final answer on its own in \\boxed{}.")
        return [{'role': 'system', 'content': sys_p}, {'role': 'user', 'content': prob['problem']}]
    if domain == 'code':
        sys_p = ("You are an expert competitive programmer. Solve with a single self-contained "
                 "C++17 program that reads from standard input and writes to standard output. "
                 "Output the final program as one ```cpp code block.")
        stmt = prob.get('statement') or prob.get('question') or prob.get('problem')
        return [{'role': 'system', 'content': sys_p}, {'role': 'user', 'content': stmt}]
    # reasoning / ifollow: the row's prompt already carries the task + any format demand
    prompt = prob.get('prompt') or prob.get('problem') or prob.get('question')
    return [{'role': 'user', 'content': prompt}]

# ---------- endpoint call ----------
def short_exc(e):
    return f'{type(e).__name__}: {e}'

def parse_domains(raw_domains):
    if isinstance(raw_domains, str):
        raw_domains = [raw_domains]
    domains = []
    for chunk in raw_domains:
        domains.extend(d.strip() for d in chunk.split(',') if d.strip())
    return domains

def endpoint_for_id(backend, prob_id):
    urls = backend['chat_urls']
    return urls[zlib.crc32(str(prob_id).encode()) % len(urls)]

def make_endpoint_states(urls, total_concurrency):
    n = max(1, len(urls))
    total = max(1, total_concurrency)
    base = max(1, total // n)
    extra = total % n
    states = {}
    for i, url in enumerate(urls):
        limit = base + (1 if i < extra else 0)
        states[url] = {'url': url, 'sem': asyncio.Semaphore(limit), 'limit': limit,
                       'failures': 0, 'down_until': 0.0}
    return states

async def sample_once(session, backend, endpoint, request_sem, messages, max_tokens, temperature, args):
    payload = {'model': backend['model'], 'messages': messages, 'max_tokens': max_tokens,
               'temperature': temperature, 'top_p': 0.95}
    streaming = backend.get('stream')          # Poe stalls long non-streamed responses -> stream it
    if streaming:
        payload['stream'] = True
    url = endpoint['url']
    last_error = None
    for attempt in range(args.request_attempts):
        delay = endpoint.get('down_until', 0.0) - time.monotonic()
        if delay > 0:
            await asyncio.sleep(min(delay, args.server_retry_sleep))
        try:
            async with endpoint['sem']:
                if streaming:
                    content = ''; reasoning = ''
                    async with request_sem:
                        async with session.post(url, json=payload,
                                                headers=backend['headers'], timeout=_TIMEOUT) as r:
                            if r.status != 200:                       # 429 rate-limit / 5xx -> retry, don't accept as empty
                                body = (await r.text())[:120]
                                raise RuntimeError(f'status {r.status}: {body}')
                            async for raw in r.content:
                                line = raw.decode('utf-8', 'ignore').strip()
                                if not line.startswith('data:'):
                                    continue
                                data = line[5:].strip()
                                if data == '[DONE]':
                                    break
                                try:
                                    delta = json.loads(data)['choices'][0].get('delta', {})
                                except Exception:
                                    continue
                                content += delta.get('content') or ''
                                reasoning += delta.get('reasoning') or delta.get('reasoning_content') or ''
                    if not content and not reasoning:             # empty stream (silent throttle) -> retry
                        raise RuntimeError('poe empty stream')
                    endpoint['failures'] = 0
                    endpoint['down_until'] = 0.0
                    return {'reasoning': reasoning, 'answer': content}
                async with request_sem:
                    async with session.post(url, json=payload,
                                            headers=backend['headers'], timeout=_TIMEOUT) as r:
                        if r.status != 200:
                            body = (await r.text())[:200]
                            raise RuntimeError(f'status {r.status}: {body}')
                        d = await r.json(content_type=None)
                msg = d['choices'][0]['message']
                content = msg.get('content') or ''
                # vLLM returns thinking in 'reasoning'; sglang uses 'reasoning_content'. Answer in content.
                reasoning = msg.get('reasoning') or msg.get('reasoning_content') or ''
                endpoint['failures'] = 0
                endpoint['down_until'] = 0.0
                return {'reasoning': reasoning, 'answer': content}
        except asyncio.CancelledError:
            raise
        except Exception as e:
            last_error = e
            if attempt + 1 < args.request_attempts:
                await asyncio.sleep(min(args.request_retry_sleep * (attempt + 1), 30))
    endpoint['failures'] = endpoint.get('failures', 0) + 1
    if endpoint['failures'] >= args.endpoint_cooldown_failures:
        endpoint['down_until'] = time.monotonic() + args.endpoint_cooldown
        print(f"[endpoint {url}] cooling for {args.endpoint_cooldown}s after "
              f"{endpoint['failures']} failed samples; last={short_exc(last_error)}", flush=True)
    return None

async def verify_generation(domain, verify, answer, prob, verify_pool, args):
    try:
        # Reasoning reward code uses signal.alarm, so it must stay on the main thread.
        if domain == 'reasoning':
            return verify(answer, prob)
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(verify_pool, verify, answer, prob),
            timeout=args.verify_timeout)
    except asyncio.TimeoutError:
        return {'passed': False, 'detail': f'verify-timeout>{args.verify_timeout}s'}
    except Exception as e:
        return {'passed': False, 'detail': f'verify-error: {short_exc(e)}'}

async def run_query(session, backend, endpoints, request_sem, domain, prob, verify, verify_pool, args):
    """Stop-at-first-pass escalation over backend['schedule']; keep passing gens (shortest-first).
    Pin this query to ONE endpoint so all its samples reuse the prompt's prefix-cached KV."""
    seen = 0
    pid = prob.get('id', '<missing-id>')
    try:
        messages = build_messages(domain, prob)
    except Exception as e:
        return {'id': pid, 'domain': domain, 'passed': False,
                'error': f'prompt-error: {short_exc(e)}'}
    url = endpoint_for_id(backend, pid)
    endpoint = endpoints[url]
    first_round_rate = None
    for ri, cap in enumerate(backend['schedule']):
        if cap > args.max_budget:
            break
        need = cap - seen
        gens = []
        remaining = need
        while remaining > 0:
            n = min(remaining, args.sample_batch_size)
            batch = await asyncio.gather(*[
                sample_once(session, backend, endpoint, request_sem, messages,
                            args.max_tokens, args.temperature, args)
                for _ in range(n)], return_exceptions=True)
            for g in batch:
                if isinstance(g, Exception):
                    print(f'[{domain}:{pid}] sample task failed: {short_exc(g)}', flush=True)
                    gens.append(None)
                else:
                    gens.append(g)
            remaining -= n
        seen = cap
        valid = 0; npass = 0; passes = []
        for g in gens:
            if not g:
                continue
            valid += 1
            ans = g['answer']
            v = await verify_generation(domain, verify, ans, prob, verify_pool, args)
            if v.get('passed'):
                npass += 1
                if len(passes) < args.keep_per_query:
                    passes.append({'reasoning': g['reasoning'], 'answer': ans,   # BOTH saved, separate
                                   'reasoning_chars': len(g['reasoning']),
                                   'answer_chars': len(ans), 'detail': v.get('detail', '')})
                elif ri > 0:
                    break                    # early-stop in escalation rounds; round 0 verifies all
        if valid == 0:              # whole round returned nothing -> server unreachable; do NOT
            return {'id': pid, 'domain': domain, 'server_error': True,
                    'endpoint': url, 'samples_used': seen}  # no write: retry on resume
        if ri == 0:
            first_round_rate = round(npass / valid, 3)
        if passes:
            # Drop-as-too-easy (all of round 0 passed) ONLY when easy-dropping is enabled
            # (easy_threshold < 1.0, e.g. the 27B run's default 0.5 -> drop 4/4). For the
            # DeepSeek tier-2 (schedule starts at 1, easy_threshold 1.1) a 1/1 first round is
            # NOT easy — these are the 27B's hard failures — so KEEP every solve.
            if first_round_rate == 1.0 and args.easy_threshold < 1.0:
                return {'id': pid, 'domain': domain, 'passed': False, 'samples_used': seen,
                        'first_round_rate': first_round_rate, 'dropped_easy': True}
            passes.sort(key=lambda p: p['reasoning_chars'] + p['answer_chars'])  # shortest first
            return {'id': pid, 'domain': domain, 'passed': True, 'samples_used': seen,
                    'first_round_rate': first_round_rate,
                    'n_kept': len(passes), 'passes': passes}
    return {'id': pid, 'domain': domain, 'passed': False, 'samples_used': seen,
            'first_round_rate': first_round_rate}

# ---------- orchestration ----------
_TIMEOUT = None
async def append_jsonl(path, row, lock, args):
    line = json.dumps(row, ensure_ascii=False) + '\n'
    async with lock:
        for attempt in range(args.write_retries):
            pos = None
            try:
                with open(path, 'a') as out:
                    pos = out.tell()
                    out.write(line)
                    out.flush()
                return True
            except OSError as e:
                if pos is not None:
                    try:
                        with open(path, 'a') as out:
                            out.truncate(pos)
                            out.flush()
                    except OSError:
                        pass
                if attempt + 1 < args.write_retries:
                    kind = 'ENOSPC' if e.errno == errno.ENOSPC else short_exc(e)
                    print(f'write failed {kind} for {path}; retrying in '
                          f'{args.write_retry_sleep}s', flush=True)
                    await asyncio.sleep(args.write_retry_sleep)
                    continue
                print(f'write failed for {path}: {short_exc(e)}; result will retry on resume',
                      flush=True)
                return False

async def run_domain(session, backend, endpoints, request_sem, domain, args, query_sem, verify_pool):
    wl = os.path.join(HARDCP, domain, args.worklist)
    probs = [json.loads(l) for l in open(wl) if l.strip()]
    suffix = args.out_suffix if args.out_suffix is not None else backend["suffix"]
    out_path = os.path.join(HARDCP, 'traces', f'{domain}{suffix}.jsonl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.isfile(out_path):
        for l in open(out_path):                 # skip any corrupt/partial line (transient ENOSPC)
            l = l.strip()
            if not l:
                continue
            try:
                done.add(json.loads(l)['id'])
            except Exception:
                pass
    todo = [p for p in probs if p['id'] not in done]
    if args.limit:
        todo = todo[:args.limit]
    verify = load_verifier(domain)
    print(f'[{domain}] {len(probs)} problems, {len(todo)} to do (skip {len(done)} done)', flush=True)
    passed = 0; processed = 0
    write_lock = asyncio.Lock()
    queue = asyncio.Queue()
    for p in todo:
        queue.put_nowait(p)
    nworkers = min(max(1, args.query_concurrency), max(1, len(todo)))

    async def worker(worker_id):
        nonlocal passed, processed
        while True:
            try:
                p = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            pid = p.get('id', '<missing-id>')
            try:
                res = None
                for attempt in range(args.server_retries):
                    async with query_sem:
                        res = await run_query(session, backend, endpoints, request_sem, domain, p,
                                              verify, verify_pool, args)
                    if not res.get('server_error'):
                        break
                    if attempt + 1 < args.server_retries:
                        await asyncio.sleep(args.server_retry_sleep)
                if res is None:
                    res = {'id': pid, 'domain': domain, 'passed': False,
                           'error': 'internal-error: no result'}
                if res.get('server_error'):
                    print(f'[{domain}:{pid}] endpoint still unavailable after '
                          f'{args.server_retries} retries; leaving for resume', flush=True)
                    continue
                res['source'] = backend['model']   # which model produced this (27B vs qwen3.7-max)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                res = {'id': pid, 'domain': domain, 'passed': False,
                       'error': f'query-error: {short_exc(e)}',
                       'source': backend['model']}
                print(f'[{domain}:{pid}] query failed but driver continues: {short_exc(e)}',
                      flush=True)
            finally:
                queue.task_done()
            wrote = await append_jsonl(out_path, res, write_lock, args)
            if wrote:
                processed += 1
                if res.get('passed'):
                    passed += 1
                if processed % args.progress_every == 0 or (res.get('passed') and passed % 25 == 0):
                    print(f'[{domain}] processed={processed} passed={passed}', flush=True)

    results = await asyncio.gather(*[asyncio.create_task(worker(i)) for i in range(nworkers)],
                                   return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            print(f'[{domain}] worker failed: {short_exc(r)}', flush=True)
    print(f'[{domain}] DONE processed={processed} passed={passed}/{len(probs)}', flush=True)

async def main_async(args):
    global _TIMEOUT
    import aiohttp, concurrent.futures
    _TIMEOUT = aiohttp.ClientTimeout(total=args.request_timeout)
    domains = parse_domains(args.domains)
    # Dedicated pool keeps blocking verify() bounded without growing asyncio's
    # default executor under a large query fanout.
    verify_pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.verify_workers)
    backend = load_backend(args)
    endpoints = make_endpoint_states(backend['chat_urls'], args.concurrency)
    print(f"backend={args.backend} model={backend['model']} schedule={backend['schedule']} "
          f"out=*{backend['suffix']}.jsonl", flush=True)
    print('endpoint_limits=' + ', '.join(f'{u}->{s["limit"]}' for u, s in endpoints.items()),
          flush=True)
    conn_limit = sum(s['limit'] for s in endpoints.values()) + 8 * len(endpoints)
    conn_per_host = max(s['limit'] for s in endpoints.values()) + 8
    conn = aiohttp.TCPConnector(limit=conn_limit, limit_per_host=conn_per_host,
                                ttl_dns_cache=60, enable_cleanup_closed=True)
    query_sem = asyncio.Semaphore(args.query_concurrency)
    global_http = max(args.concurrency, GLOBAL_HTTP_CONCURRENCY)
    request_sem = asyncio.Semaphore(global_http)
    print(f'global_http_concurrency={global_http} (--concurrency={args.concurrency})', flush=True)
    try:
        async with aiohttp.ClientSession(connector=conn) as session:
            # All domains progress in parallel. Endpoint-local semaphores prevent one wedged
            # replica from consuming the whole client connection pool.
            # PER-DOMAIN query budget: one shared FIFO query_sem lets the first domain (reasoning)
            # hold the whole budget and STARVE code/math. Give each domain its own slice so all
            # four progress in parallel.
            qshare = max(4, args.query_concurrency // max(1, len(domains)))
            results = await asyncio.gather(*[
                run_domain(session, backend, endpoints, request_sem, d, args,
                           asyncio.Semaphore(qshare), verify_pool)
                for d in domains], return_exceptions=True)
            for d, r in zip(domains, results):
                if isinstance(r, Exception):
                    print(f'[{d}] domain failed but other domains were isolated: {short_exc(r)}',
                          flush=True)
    finally:
        verify_pool.shutdown(wait=True, cancel_futures=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domains', nargs='+', default=['math', 'code', 'reasoning', 'ifollow'])
    ap.add_argument('--backend', choices=['local', 'poe', 'deepseek'], default='local',
                    help='local vLLM 27B (schedule 4..1024), Poe qwen3.7-max, or DeepSeek V4 Pro tier-2 (schedule 1..8)')
    ap.add_argument('--poe-model', default='qwen3.7-max')
    ap.add_argument('--deepseek-model', default='deepseek-v4-pro')
    ap.add_argument('--worklist', default='worklist.jsonl',
                    help='which worklist file per domain (e.g. failed_27b.jsonl for the Poe hard-tail pass)')
    ap.add_argument('--out-suffix', default=None,
                    help="override the backend's output suffix so a deep re-roll pass writes to a "
                         "SEPARATE traces/<domain><suffix>.jsonl (e.g. .reroll) instead of colliding "
                         "with the first-pass file (whose ids would be skipped as 'done')")
    ap.add_argument('--url', default=None)
    ap.add_argument('--model', default=None)
    # Saturate the server WITHOUT crowding KV. The local setup here is two independent TP=2
    # services on GPU pairs 1,2 and 3,5.
    # Generations are LONG (hard problems reason ~30-50k tokens), so KV — not max_num_seqs — is the
    # binding limit. Start MODERATE and TUNE UP live: watch :30000/metrics, push num_requests_running
    # up until num_preemptions_total climbs or gpu_cache_usage_perc ~0.9, then hold. This semaphore is
    # the ONLY limiter (no proactive throttling) so we never end up rate-limiting while also flooding.
    ap.add_argument('--concurrency', type=int, default=64, help='max in-flight sample requests (tune up vs /metrics)')
    ap.add_argument('--query-concurrency', type=int, default=64,
                    help='global max queries escalating at once — keep >= concurrency to keep it full')
    # High so </think> CLOSES and the answer lands in content (verifier scores content). Must keep
    # prompt+max_tokens <= 65536 context. Hard problems that never finish -> content empty -> fail (ok).
    ap.add_argument('--max-tokens', type=int, default=57344)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--max-budget', type=int, default=128)
    ap.add_argument('--keep-per-query', type=int, default=8,
                    help='keep up to this many correct generations from the deciding round (shortest-first)')
    ap.add_argument('--easy-threshold', type=float, default=0.5,
                    help='DROP a problem if its pass rate over the first 8 samples exceeds this (too easy)')
    ap.add_argument('--request-timeout', type=int, default=1800)
    ap.add_argument('--request-attempts', type=int, default=3)
    ap.add_argument('--request-retry-sleep', type=int, default=2)
    ap.add_argument('--endpoint-cooldown-failures', type=int, default=8)
    ap.add_argument('--endpoint-cooldown', type=int, default=30)
    ap.add_argument('--server-retries', type=int, default=30)
    ap.add_argument('--server-retry-sleep', type=int, default=30)
    ap.add_argument('--sample-batch-size', type=int, default=32)
    ap.add_argument('--verify-timeout', type=int, default=300)
    ap.add_argument('--write-retries', type=int, default=3)
    ap.add_argument('--write-retry-sleep', type=int, default=5)
    ap.add_argument('--progress-every', type=int, default=25)
    ap.add_argument('--limit', type=int, default=0, help='process only first N problems/domain (0=all; for pilots)')
    ap.add_argument('--verify-workers', type=int, default=16, help='thread pool for blocking verify() calls')
    args = ap.parse_args()
    asyncio.run(main_async(args))

if __name__ == '__main__':
    main()
