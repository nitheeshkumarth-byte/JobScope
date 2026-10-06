"""The n8n resume-screening workflow is checked here, not executed.

Importing it costs a container and a model, so these tests pin the things
that break silently: connection wiring, the credential/token handling, and
the contract between the AI Agent's output and the node that parses it.
"""

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / "n8n" / "workflows" / "resume-screening.json"


def _load_workflow() -> dict:
    return json.loads(WORKFLOW.read_text(encoding="utf-8"))


def _load_import_module():
    spec = importlib.util.spec_from_file_location(
        "jobscope_n8n_import", ROOT / "n8n" / "import.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _node(workflow: dict, name: str) -> dict:
    for node in workflow["nodes"]:
        if node["name"] == name:
            return node
    raise AssertionError(f"node {name!r} is missing")


def test_every_connection_points_at_a_node_that_exists():
    wf = _load_workflow()
    names = {n["name"] for n in wf["nodes"]}
    for source, channels in wf["connections"].items():
        assert source in names, source
        for outputs in channels.values():          # one list per output slot
            for output_targets in outputs:
                for target in output_targets:
                    assert target["node"] in names, target["node"]
    # The model must reach the agent, and the webhook must reach the scorer:
    # those are the two wires the whole workflow exists to make.
    model_links = wf["connections"]["Ollama Chat Model"]["ai_languageModel"]
    assert model_links[0][0]["node"] == "AI Agent"
    assert wf["connections"]["Webhook"]["main"][0][0]["node"] == "Pack request"


def test_the_shared_secret_lives_in_a_credential_not_in_the_json():
    """A token baked into headerParameters would be readable by anyone who
    can read the workflow; it belongs in an httpHeaderAuth credential."""
    wf = _load_workflow()
    screen = _node(wf, "Screen resume")
    params = screen["parameters"]
    assert "headerParameters" not in params
    assert params["authentication"] == "genericCredentialType"
    assert params["genericAuthType"] == "httpHeaderAuth"
    assert "httpHeaderAuth" in screen["credentials"]
    # The import script is what fills these in.
    raw = WORKFLOW.read_text(encoding="utf-8")
    assert "REPLACE_SCREEN_TOKEN_CREDENTIAL" in raw
    assert "REPLACE_OLLAMA_CREDENTIAL" in raw


def test_the_agent_may_only_explain_and_must_emit_four_lines():
    agent = _node(_load_workflow(), "AI Agent")
    system = agent["parameters"]["options"]["systemMessage"]
    assert "Never invent" in system
    assert "already computed by JobScope" in system
    for label in ("Skills:", "Experience:", "Missing:", "Verdict:"):
        assert label in system
    # The input is the screening JSON itself, from the node that produced it.
    assert "$('Screen resume')" in agent["parameters"]["text"]


def test_both_code_nodes_hold_up_their_end_of_the_contract():
    wf = _load_workflow()
    pack = _node(wf, "Pack request")["parameters"]["jsCode"]
    assert "getBinaryDataBuffer" in pack
    assert "resume_base64" in pack
    shape = _node(wf, "Shape analysis")["parameters"]["jsCode"]
    assert "Skills|Experience|Missing|Verdict" in shape
    assert "'$json.output'" in shape or "$json && $json.output" in shape


def test_the_import_script_patches_credentials_and_the_model():
    mod = _load_import_module()
    wf = mod.prepare_workflow({"OLLAMA_MODEL": "qwen2.5:14b"},
                              ("cred-1", "Screen Token"),
                              ("cred-2", "Ollama"))
    assert _node(wf, "Screen resume")["credentials"]["httpHeaderAuth"]["id"] == "cred-1"
    assert _node(wf, "Ollama Chat Model")["credentials"]["ollamaApi"]["id"] == "cred-2"
    assert _node(wf, "Ollama Chat Model")["parameters"]["model"] == "qwen2.5:14b"
    # Unpatched (dry run) keeps the placeholders instead of inventing ids.
    dry = mod.prepare_workflow({}, None, None)
    creds = _node(dry, "Screen resume")["credentials"]["httpHeaderAuth"]
    assert creds["id"].startswith("REPLACE_")


def test_the_import_script_points_n8n_at_ollama_through_docker():
    mod = _load_import_module()
    # .env stores the host loopback address, which is wrong from inside the
    # container - the alias has to be substituted, and an override wins.
    assert mod.ollama_base_url({"OLLAMA_URL": "http://127.0.0.1:11434"}) == \
        "http://host.docker.internal:11434"
    assert mod.ollama_base_url(
        {"OLLAMA_URL": "http://localhost:11434"}) == \
        "http://host.docker.internal:11434"
    assert mod.ollama_base_url(
        {"N8N_OLLAMA_BASE_URL": "http://ollama:11434",
         "OLLAMA_URL": "http://127.0.0.1:11434"}) == "http://ollama:11434"
