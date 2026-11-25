"""Check configuration without printing any secret value.

  python scripts/check_setup.py          # shows set / blank for each variable
  python scripts/check_setup.py --live --chat-endpoint https://HOST:PORT/v1 --model MODEL --chat-ca-cert PATH
                                         # also makes one small call to each configured endpoint;
                                         # the API key is asked for with a hidden prompt
"""
import argparse

import _paths  # noqa: F401
from copilot.config import embeddings_configured, env, llm_configured
from copilot.rag import load_corpus
from copilot.runtime import add_endpoint_args, apply_endpoint_args

NAMES = ["COPILOT_CHAT_ENDPOINT", "COPILOT_CHAT_API_KEY", "COPILOT_MODEL_NAME", "COPILOT_CHAT_CA_CERT",
         "COPILOT_EMBED_ENDPOINT", "COPILOT_EMBED_API_KEY", "COPILOT_EMBED_MODEL", "COPILOT_EMBED_CA_CERT",
         "OPENFDA_API_KEY"]


def main() -> None:
    """Print which settings are set (never their values) and, with --live, call each endpoint once."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    add_endpoint_args(ap)
    args = ap.parse_args()
    apply_endpoint_args(args)
    print("Configuration (values are never shown):")
    for name in NAMES:
        print(f"  {name:26s} {'set' if env(name) else 'blank (feature off)'}")
    print(f"\nChat endpoint configured:       {llm_configured()}")
    print(f"Embeddings endpoint configured: {embeddings_configured()}")
    corpus = load_corpus()
    print(f"Corpus documents:               {len(corpus) if corpus else 0}")
    if args.live and llm_configured():
        from copilot.agent.llm import ChatClient
        try:
            ChatClient().chat([{"role": "user", "content": "Reply with the word OK."}], max_tokens=5)
            print("Live chat call: OK")
        except Exception as err:
            # The message names the missing file or HTTP status; it never contains the key.
            print(f"Live chat call failed: {type(err).__name__}: {str(err)[:200]}")
            if "CERTIFICATE_VERIFY_FAILED" in str(err):
                print("  The CA file does not match this endpoint. Each course port has its own CA, "
                      "so check the port and the certificate belong together.")
    if args.live and embeddings_configured():
        from copilot.agent.llm import EmbedClient
        try:
            EmbedClient().embed(["test"])
            print("Live embeddings call: OK")
        except Exception as err:
            print(f"Live embeddings call failed: {type(err).__name__}: {str(err)[:200]}")


if __name__ == "__main__":
    main()
