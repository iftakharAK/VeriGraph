

import os, argparse, json, itertools
import torch
from transformers import AutoTokenizer
from encoder import TextEncoder
from contradiction_scorer import PairwiseContradictionScorer
from graph_builder import build_pair_indices, construct_graph
from retriever_kcs_fid import KnowledgeConsistencyScorer, FiDCombiner
from gnn_reasoner import GNNReasoner
from explanation_generator import ExplanationGenerator
from reflection_verifier import ReflectionVerifier

def load_scorer(ckpt_path, encoder_name="microsoft/deberta-v3-small", device="cuda"):
    enc = TextEncoder(encoder_name)
    hidden = enc.model.config.hidden_size
    scorer = PairwiseContradictionScorer(hidden)
    state = torch.load(ckpt_path, map_location=device)
    scorer.load_state_dict(state["scorer"])
    enc.model.load_state_dict(state["encoder"])
    enc.model.to(device); scorer.to(device)
    enc.model.eval(); scorer.eval()
    return enc, scorer

def score_pairs(enc, scorer, statements, device="cuda", max_length=128):
    pairs = build_pair_indices(len(statements))
    s1 = [statements[i] for i, j in pairs]
    s2 = [statements[j] for i, j in pairs]
    pooled = enc.encode_pairs(s1, s2, device=device, max_length=max_length)
    with torch.no_grad():
        scores = scorer(pooled).detach().cpu().tolist()
    return pairs, {p: s for p, s in zip(pairs, scores)}

def run_pipeline(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc, scorer = load_scorer(args.ckpt, args.encoder, device)

    # Optional modules
    retriever = KnowledgeConsistencyScorer(args.kcs_model) if args.use_retrieval else None
    fid = FiDCombiner() if args.use_retrieval else None
    gnn = GNNReasoner(enc.model.config.hidden_size) if args.use_gnn else None
    expl = ExplanationGenerator(args.generator)
    verifier = ReflectionVerifier(args.verifier_model, tau=args.verify_tau) if args.use_verify else None

    outputs = []
    with open(args.input_jsonl, "r") as f:
        for sid, line in enumerate(f):
            obj = json.loads(line)
            sents = obj["statements"]
            pairs, score_map = score_pairs(enc, scorer, sents, device, args.max_length)
            edges = construct_graph(score_map, n_nodes=len(sents), threshold=args.edge_threshold, keep_best_acyclic=True)

            # Optional GNN reasoning over node features
            node_feats = enc.encode_pairs(sents, [""]*len(sents), device=device, max_length=args.max_length)  # crude per-node embedding using token[0]
            if gnn:
                node_feats = gnn(node_feats, edges, device=device)

            # For each directed contradictory edge (i->j), produce explanation
            for (i, j) in edges:
                s1, s2 = sents[i], sents[j]
                knowledge_text = ""
                if retriever:
                    # placeholder external docs: pass empty list OR your corpora
                    kept = retriever.filter_passages(query=f"{s1} || {s2}", passages=args.knowledge_corpus, tau=args.kcs_tau, k=args.kcs_topk)
                    knowledge_text = "\n".join(kept)
                fid_input = fid.build_fid_input(f"Explain contradiction between: '{s1}' and '{s2}'", kept) if (fid and knowledge_text) else knowledge_text
                explanation = expl.generate(s1, s2, knowledge=fid_input)

                verified = None; entail_prob = None
                if verifier:
                    verified, entail_prob = verifier.verify(s1, s2, explanation, knowledge_text)

                outputs.append({
                    "sample_id": sid,
                    "edge": [i, j],
                    "s1": s1,
                    "s2": s2,
                    "score": round(score_map[(i, j)], 3),
                    "explanation": explanation,
                    "verified": verified,
                    "entail_prob": None if entail_prob is None else round(entail_prob, 3)
                })

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    with open(args.out_jsonl, "w") as g:
        for o in outputs:
            g.write(json.dumps(o) + "\n")
    print(f"Saved results to {args.out_jsonl}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True, help="Unlabeled multi-statement samples (statements only).")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to pairwise scorer checkpoint (final.pt).")
    ap.add_argument("--out_jsonl", type=str, default="verigraph_results.jsonl")

    # Models
    ap.add_argument("--encoder", type=str, default="microsoft/deberta-v3-small")
    ap.add_argument("--generator", type=str, default="google/flan-t5-base")
    ap.add_argument("--verifier_model", type=str, default="microsoft/deberta-v3-base")
    ap.add_argument("--kcs_model", type=str, default="microsoft/deberta-v3-base")

    # Pipeline options
    ap.add_argument("--edge_threshold", type=float, default=0.5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--use_gnn", action="store_true")
    ap.add_argument("--use_retrieval", action="store_true")
    ap.add_argument("--use_verify", action="store_true")

    # Retrieval params
    ap.add_argument("--kcs_tau", type=float, default=0.7)
    ap.add_argument("--kcs_topk", type=int, default=5)
    ap.add_argument("--knowledge_corpus", nargs="*", default=[], help="List of strings or pass empty to skip.")

    # Verification
    ap.add_argument("--verify_tau", type=float, default=0.7)

    args = ap.parse_args()
    run_pipeline(args)