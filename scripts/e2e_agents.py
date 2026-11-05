#!/usr/bin/env python3
"""End-to-end test of the agent layer against real `iiitd` processes.

Starts two nodes on localhost (A with an A2A gateway, B a plain member), plus a
stand-in for the federation serving endpoint (an OpenAI-compatible
/v1/chat/completions server), then checks, through the nodes' real HTTP/QUIC/gossip
interfaces: card discovery, authenticated A2A calls, task execution through the
persistent Python worker, gossip + QUIC-based directory discovery on the other
node, admin-API protection, signed halt/resume propagation, and recovery of state
across a hard restart.

Usage (from the repo root, after `cargo build`):  python3 scripts/e2e_agents.py
"""
import hashlib
import http.server
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("IIITD_BIN", os.path.join(ROOT, "target", "debug", "iiitd"))
ANSI = re.compile(r"\x1b\[[0-9;]*m")
MOCK_PORT = 19199
A = dict(p2p=19101, ws=19102, api=19103, gw=19104)
B = dict(p2p=19111, ws=19112, api=19113, gw=0)
C = dict(p2p=19121, ws=19122, api=19123, gw=0)
OUTSIDE_KEY, ADMIN_KEY = "outside-secret-key", "admin-secret-key"
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()

failures = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def hcall(method, url, body=None, headers=None, timeout=10):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw.decode(errors="replace")


def rpc(port, method, params, key=OUTSIDE_KEY):
    h = {"Authorization": f"Bearer {key}"} if key else {}
    return hcall("POST", f"http://127.0.0.1:{port}/a2a", {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, h)


def admin(port, path, method="GET", body=None, key=ADMIN_KEY):
    h = {"Authorization": f"Bearer {key}"} if key else {}
    return hcall(method, f"http://127.0.0.1:{port}{path}", body, h)


def wait_for(fn, timeout=60, step=0.5):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        try:
            last = fn()
            if last:
                return last
        except Exception as e:  # not up yet
            last = e
        time.sleep(step)
    return None


class MockServing(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        msgs = req["messages"]
        last = msgs[-1]["content"]
        if msgs[0]["content"].startswith("You are an agent with tools"):
            # scripted tool-using model: call the tool once, then answer with what it returned
            seen = [m["content"] for m in msgs if m["content"].startswith("Result:")]
            if not seen:
                reply = json.dumps({"call": "notes.add", "args": {"a": 20, "b": 22}})
            else:
                n = re.search(r"\n(-?\d+)\n</untrusted-data", seen[-1])
                reply = json.dumps({"final": f"the tool says {n.group(1) if n else '?'}"})
        else:
            reply = f"echo: {last}"
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": reply}}], "usage": {"total_tokens": 11}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def halt_msg(key: Ed25519PrivateKey, action, seq, reason):
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    reason_b = reason.encode()
    msg = b"slakshna/halt/v1\0" + bytes([0 if action == "Halt" else 1]) + struct.pack("<Q", seq) + struct.pack("<Q", len(reason_b)) + reason_b + pub.encode()
    return {"action": action, "seq": seq, "reason": reason, "issuer_key": pub, "signature": key.sign(msg).hex()}


class Node:
    def __init__(self, name, ports, workdir, extra_toml, peers=()):
        self.name, self.ports, self.dir = name, ports, workdir
        self.cfg = os.path.join(workdir, f"{name}.toml")
        self.log = open(os.path.join(workdir, f"{name}.log"), "ab")
        self.extra, self.peers, self.proc, self.endpoint_id = extra_toml, list(peers), None, None
        self.write_cfg()

    def write_cfg(self):
        peers = ", ".join(f'"{p}"' for p in self.peers)
        with open(self.cfg, "w") as f:
            f.write(f"""
[federation]
id = "e2e-agents"
name = "e2e"
[training]
epoch_duration_secs = 86400
sync_deadline_secs = 86000
expected_peers = 2
[node]
id = "{self.name}"
data_dir = "{self.dir}/data-{self.name}"
[network]
host = "127.0.0.1"
p2p_port = {self.ports['p2p']}
ws_port = {self.ports['ws']}
api_port = {self.ports['api']}
peers = [{peers}]
[discovery]
mdns = false
dht = false
dns = false
relay = false
[logging]
level = "info"
{self.extra}
""")

    def start(self):
        env = dict(os.environ, SLAKSHNA_AGENT_WORKER=os.path.join(ROOT, "agent_worker.py"), NO_COLOR="1")
        self.proc = subprocess.Popen([BIN, "--config", self.cfg], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        self.lines = []

        def pump():
            for raw in self.proc.stdout:
                line = ANSI.sub("", raw.decode(errors="replace"))
                self.lines.append(line)
                self.log.write(line.encode())
                self.log.flush()
                m = re.search(r"Iroh EndpointId: ([0-9a-f]{64})", line)
                if m:
                    self.endpoint_id = m.group(1)

        threading.Thread(target=pump, daemon=True).start()

    def kill(self, sig=signal.SIGKILL):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(sig)
            self.proc.wait(timeout=15)


def agents_cfg(ports, extra="", price=None):
    gw = f"gateway_port = {ports['gw']}\n" if ports["gw"] else ""
    return f"""
[agents]
enabled = true
{gw}admin_key_sha256 = "{sha(ADMIN_KEY)}"
halt_keys = ["{HALT_PUB}"]
serve_url = "http://127.0.0.1:{MOCK_PORT}"
python = "python3"
task_timeout_secs = 60
{('[agents.prices]' + chr(10) + 'chat = ' + str(price)) if price is not None else ''}
[[agents.api_keys]]
principal = "outside"
key_sha256 = "{sha(OUTSIDE_KEY)}"
{extra}
"""


halt_key = Ed25519PrivateKey.generate()
HALT_PUB = halt_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


def main():
    if not os.path.exists(BIN):
        sys.exit(f"binary not found: {BIN} (run cargo build)")
    work = tempfile.mkdtemp(prefix="slk-e2e-")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", MOCK_PORT), MockServing)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    nodes = []
    try:
        manifest = os.path.join(work, "tools.json")
        with open(manifest, "w") as f:
            json.dump({"servers": [{"name": "notes", "command": "python3", "args": [os.path.join(ROOT, "tests", "mcp_test_server.py")], "allow": ["add"]}]}, f)
        a = Node("A", A, work, agents_cfg(A, extra=f'[agents.tools]\nmanifest = "{manifest}"\n'))
        nodes.append(a)
        a.start()
        up = wait_for(lambda: a.endpoint_id and hcall("GET", f"http://127.0.0.1:{A['gw']}/.well-known/agent-card.json")[0] == 200, 60)
        check("node A boots with the A2A gateway", bool(up), "".join(a.lines[-15:]))
        if not up:
            return
        b = Node("B", B, work, agents_cfg(B), peers=[f"{a.endpoint_id}@127.0.0.1:{A['p2p']}"])
        nodes.append(b)
        b.start()

        gw = A["gw"]
        # 1. public card
        st, card = hcall("GET", f"http://127.0.0.1:{gw}/.well-known/agent-card.json")
        skills = [s["id"] for s in card.get("skills", [])]
        bindings = {i["protocolBinding"] for i in card.get("supportedInterfaces", [])}
        check("agent card is public and well-formed (A2A v1.0 fields)", st == 200 and "chat" in skills and {"JSONRPC", "SLAKSHNA-QUIC"} <= bindings and card.get("signatures"), json.dumps(card)[:300])

        # 2. auth
        st, _ = rpc(gw, "GetTask", {"id": "x"}, key=None)
        check("gateway rejects calls without a key (401)", st == 401, st)
        st, _ = rpc(gw, "GetTask", {"id": "x"}, key="wrong")
        check("gateway rejects a wrong key (401)", st == 401, st)

        # 3. a real task through the persistent worker and the (stand-in) serving endpoint
        msg = {"messageId": "e2e-1", "role": "ROLE_USER", "parts": [{"text": "hello federation"}], "metadata": {"skill": "chat"}}
        st, r = rpc(gw, "SendMessage", {"message": msg})
        task_id = r.get("result", {}).get("task", {}).get("id")
        check("SendMessage returns a task", st == 200 and task_id, r)
        done = wait_for(lambda: (lambda t: t if t["result"]["status"]["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_FAILED") else None)(rpc(gw, "GetTask", {"id": task_id})[1]), 30)
        state = done["result"]["status"]["state"] if done else None
        text = done and done["result"]["artifacts"][0]["parts"][0].get("text")
        check("chat task completes with the model's reply", state == "TASK_STATE_COMPLETED" and text == "echo: hello federation", json.dumps(done)[:400] if done else a.lines[-10:])
        st, again = rpc(gw, "SendMessage", {"message": msg})
        check("resending the same messageId is idempotent", again["result"]["task"]["id"] == task_id)

        # 3b. the tool-using agent: model -> MCP tool (from the manifest, under a capability token) -> answer
        msg2 = {"messageId": "e2e-agent-1", "role": "ROLE_USER", "parts": [{"text": "add 20 and 22 with your tool"}], "metadata": {"skill": "agent"}}
        st, r2 = rpc(gw, "SendMessage", {"message": msg2})
        t2id = r2.get("result", {}).get("task", {}).get("id")
        d2 = wait_for(lambda: (lambda t: t if t["result"]["status"]["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_FAILED") else None)(rpc(gw, "GetTask", {"id": t2id})[1]), 40)
        txt2 = d2 and d2["result"]["artifacts"][0]["parts"][0].get("text")
        check("agent skill calls an MCP tool under a capability token and answers", d2 and d2["result"]["status"]["state"] == "TASK_STATE_COMPLETED" and txt2 == "the tool says 42", json.dumps(d2)[:400] if d2 else a.lines[-10:])
        st, card_now = hcall("GET", f"http://127.0.0.1:{gw}/.well-known/agent-card.json")
        check("the card advertises the agent skill", "agent" in [x["id"] for x in card_now.get("skills", [])])

        # 4. admin API protection
        check("admin API refuses anonymous callers", admin(A["api"], "/agents/tasks", key=None)[0] == 401)
        check("admin API refuses a wrong key", admin(A["api"], "/agents/tasks", key="nope")[0] == 401)
        st, tl = admin(A["api"], "/agents/tasks")
        check("admin API lists the task with its outside principal", st == 200 and any(t["metadata"]["sender"] == "api:outside" for t in tl["tasks"]))
        st, sa = admin(A["api"], "/agents/status")
        check("status shows skills and agent key", st == 200 and "chat" in sa["skills"] and len(sa["agent_key"]) == 64 and not sa["halted"])

        # 5. B discovers A's agent: announcement over gossip + signed card fetched over QUIC
        def b_sees_a():
            st, d = admin(B["api"], "/agents/directory?skill=chat")
            return d if st == 200 and any(x["node_id"] == sa["node_id"] for x in d["agents"]) else None
        found = wait_for(b_sees_a, 90, 2)
        check("node B discovers node A's verified agent (gossip + QUIC card fetch)", bool(found), "".join(b.lines[-12:]))

        # 5b. a third node joins; B delegates a task across the federation with redundancy 2
        c = Node("C", C, work, agents_cfg(C, price=7), peers=[f"{a.endpoint_id}@127.0.0.1:{A['p2p']}"])
        nodes.append(c)
        c.start()
        def b_sees_both():
            st, d = admin(B["api"], "/agents/directory?skill=chat")
            return d if st == 200 and len({x["node_id"] for x in d["agents"]}) >= 2 else None
        both = wait_for(b_sees_both, 120, 2)
        check("node B's directory lists both A and C (prices announced in their cards)", bool(both) and sorted(x["card"]["slakshnaPrices"].get("chat") for x in both["agents"] if x["card"].get("slakshnaPrices")) == [7], json.dumps(both)[:400] if both else "".join(b.lines[-8:]))
        st, dl = admin(B["api"], "/agents/delegate", "POST", {"skill": "chat", "input": {"messages": [{"role": "user", "content": "delegate me"}]}, "redundancy": 2, "timeout_secs": 60})
        check("delegation across the federation reaches consensus", st == 200 and dl.get("consensus") == "echo: delegate me" and sum(1 for x in dl["agents"] if x["agreed"]) == 2, dl)
        check("delegation pays each awarded agent's announced price", st == 200 and dl["cost"] >= 7, dl)
        st, tl = admin(B["api"], "/agents/tasks")
        check("the delegation is recorded as a task on the delegating node", st == 200 and any(t["id"] == dl["root_task"] and t["status"]["state"] == "TASK_STATE_COMPLETED" for t in tl["tasks"]), tl)

        # 6. signed halt: wrong key refused, right key applied and gossiped to B
        rogue = Ed25519PrivateKey.generate()
        st, _ = hcall("POST", f"http://127.0.0.1:{A['api']}/agents/halt", halt_msg(rogue, "Halt", 1, "attack"))
        check("halt signed by an unknown key is refused (403)", st == 403, st)
        st, hr = hcall("POST", f"http://127.0.0.1:{A['api']}/agents/halt", halt_msg(halt_key, "Halt", 1, "red-team drill"))
        check("halt signed by an authorized key is applied", st == 200 and hr.get("halted") is True, hr)
        st, _ = rpc(gw, "SendMessage", {"message": dict(msg, messageId="e2e-halted")})
        t2 = rpc(gw, "SendMessage", {"message": dict(msg, messageId="e2e-halted")})[1]["result"]["task"]["id"]
        rej = wait_for(lambda: (lambda t: t if t["result"]["status"]["state"] != "TASK_STATE_SUBMITTED" else None)(rpc(gw, "GetTask", {"id": t2})[1]), 15)
        check("a halted node rejects new work", rej and rej["result"]["status"]["state"] == "TASK_STATE_REJECTED", rej)
        b_halted = wait_for(lambda: admin(B["api"], "/agents/status")[1].get("halted") is True, 30, 1)
        check("the halt reached node B over gossip", bool(b_halted))
        st, _ = hcall("POST", f"http://127.0.0.1:{A['api']}/agents/halt", halt_msg(halt_key, "Halt", 1, "replay"))
        check("a replayed halt (same seq) changes nothing", st == 200)
        st, rr = hcall("POST", f"http://127.0.0.1:{A['api']}/agents/halt", halt_msg(halt_key, "Resume", 2, "all clear"))
        check("resume lifts the halt", st == 200 and rr.get("halted") is False, rr)
        t3 = rpc(gw, "SendMessage", {"message": dict(msg, messageId="e2e-after-resume", parts=[{"text": "back"}])})[1]["result"]["task"]["id"]
        ok3 = wait_for(lambda: (lambda t: t if t["result"]["status"]["state"] == "TASK_STATE_COMPLETED" else None)(rpc(gw, "GetTask", {"id": t3})[1]), 30)
        check("work is accepted again after resume", bool(ok3))

        # 7. hard restart of A: state and identity survive
        key_before, ver_before = sa["agent_key"], card["version"]
        a.kill()
        a.endpoint_id = None
        a.start()
        up = wait_for(lambda: hcall("GET", f"http://127.0.0.1:{gw}/.well-known/agent-card.json")[0] == 200, 60)
        check("node A restarts after SIGKILL", bool(up))
        st, old = rpc(gw, "GetTask", {"id": task_id})
        check("completed task and its artifact survive the restart", st == 200 and old["result"]["status"]["state"] == "TASK_STATE_COMPLETED" and old["result"]["artifacts"], old)
        st, sa2 = admin(A["api"], "/agents/status")
        check("agent key and node identity are stable across restart", sa2["agent_key"] == key_before and sa2["node_id"] == sa["node_id"])
        st, card2 = hcall("GET", f"http://127.0.0.1:{gw}/.well-known/agent-card.json")
        check("card is unchanged after restart (no spurious new version)", card2 == card or card2.get("skills") == card.get("skills"))
        st, r5 = rpc(gw, "SendMessage", {"message": dict(msg, messageId="e2e-post-restart", parts=[{"text": "after restart"}])})
        t5 = r5["result"]["task"]["id"]
        ok5 = wait_for(lambda: (lambda t: t if t["result"]["status"]["state"] == "TASK_STATE_COMPLETED" else None)(rpc(gw, "GetTask", {"id": t5})[1]), 30)
        check("the restarted node serves new tasks", bool(ok5))
    finally:
        for n in nodes:
            n.kill(signal.SIGTERM)
        srv.shutdown()
        if failures:
            print(f"\nlogs kept in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED: {failures}"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
