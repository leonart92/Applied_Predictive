"""Classification zero-shot par LLM sur le texte des tweets de CrisisMMD.

Supporte deux backends locaux reproductibles (température = 0, seed = 42) :
1. Ollama local (`llama3.2:latest` par défaut) : génération déterministe à
   température 0 (seed 42) avec règle de parsing stricte et repli documenté.
2. HuggingFace Transformers (`Qwen/Qwen2.5-3B-Instruct` par défaut) : calcule les
   probabilités exactes des étiquettes candidates (log-vraisemblance conditionnelle
   normalisée par softmax) et contraint la prédiction aux classes valides.

Lit les prompts depuis `src/llm_prompt.txt`, charge les splits via `src.data.get_split`,
et écrit les prédictions et métriques bootstrap via `src.evaluate`.

Usage :
    python -m src.llm_zero_shot [informative|humanitarian] [--backend ollama|hf] [--model NOM] [--export-protocol-b]
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import EVENTS, LABELS, SEED, get_split
from src.evaluate import evaluate_file, write_predictions

PROMPT_PATH = ROOT / "src" / "llm_prompt.txt"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_URL = "http://localhost:11434/api/chat"

# Classe majoritaire du train par tâche (utilisée uniquement si une réponse libre est totalement invalide)
FALLBACK_CLASS = {
    "informative": "informative",
    "humanitarian": "not_humanitarian",
}


def load_prompts(task: str) -> tuple[str, str]:
    """Extrait les sections [<TASK>_SYSTEM] et [<TASK>_USER] depuis src/llm_prompt.txt."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    prefix = task.upper()
    sys_match = re.search(rf"\[{prefix}_SYSTEM\]\n(.*?)(?=\n\[|\Z)", text, flags=re.S)
    usr_match = re.search(rf"\[{prefix}_USER\]\n(.*?)(?=\n\[|\Z)", text, flags=re.S)
    if not sys_match or not usr_match:
        raise ValueError(f"Sections [{prefix}_SYSTEM] ou [{prefix}_USER] introuvables dans {PROMPT_PATH}")
    return sys_match.group(1).strip(), usr_match.group(1).strip()


def parse_llm_output(raw_text: str, classes: tuple[str, ...], fallback: str) -> tuple[str, bool]:
    """Règle de parsing des réponses du LLM :
    1. Normalisation (minuscules, espaces/ponctuation remplacés par `_` si besoin).
    2. Correspondance exacte du premier jeton avec une classe valide.
    3. Recherche de la première classe valide mentionnée dans la réponse (en testant
       par longueur décroissante, donc `not_informative` avant `informative`).
    4. Si aucune classe n'est reconnue : repli sur `fallback` et signalement (`is_valid=False`).
    """
    cleaned = raw_text.strip().lower()
    cleaned_token = re.sub(r"[^a-z_]", "", cleaned.split()[0]) if cleaned.split() else ""
    if cleaned_token in classes:
        return cleaned_token, True

    normalized = re.sub(r"[\s\-]+", "_", cleaned)
    for cls in sorted(classes, key=len, reverse=True):
        if re.search(rf"\b{re.escape(cls)}\b", normalized):
            return cls, True

    return fallback, False


def predict_ollama(
    df: pd.DataFrame,
    task: str,
    model_name: str,
    system_prompt: str,
    user_template: str,
    classes: tuple[str, ...],
) -> tuple[list[str], np.ndarray, int]:
    """Exécute le LLM via l'API locale Ollama (température = 0, seed = 42)."""
    preds = []
    probas = np.zeros((len(df), len(classes)), dtype=np.float64)
    invalid_count = 0
    fallback = FALLBACK_CLASS[task]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for idx, tweet_text in enumerate(df["tweet_text"].tolist()):
        user_msg = user_template.format(tweet_text=tweet_text)
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "seed": SEED,
                "num_predict": 16,
            },
        }
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw_reply = data.get("message", {}).get("content", "")
        label, is_valid = parse_llm_output(raw_reply, classes, fallback)
        if not is_valid:
            invalid_count += 1
        preds.append(label)
        probas[idx, class_to_idx[label]] = 1.0

        if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
            print(f"[{task}] {idx + 1}/{len(df)} tweets traités (réponses invalides : {invalid_count})", flush=True)

    return preds, probas, invalid_count


def predict_hf(
    df: pd.DataFrame,
    task: str,
    model_name: str,
    system_prompt: str,
    user_template: str,
    classes: tuple[str, ...],
) -> tuple[list[str], np.ndarray, int]:
    """Évalue la log-vraisemblance exacte de chaque étiquette candidate avec HuggingFace Transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    label_token_ids = [tokenizer.encode(" " + c, add_special_tokens=False) for c in classes]
    preds = []
    probas = np.zeros((len(df), len(classes)), dtype=np.float64)

    with torch.no_grad():
        for idx, tweet_text in enumerate(df["tweet_text"].tolist()):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_template.format(tweet_text=tweet_text)},
            ]
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

            log_probs = []
            for cand_ids in label_token_ids:
                input_ids = torch.tensor([prompt_ids + cand_ids], device=device)
                logits = model(input_ids=input_ids).logits[0]
                lp = 0.0
                for pos, tok_id in enumerate(cand_ids):
                    logit_pos = len(prompt_ids) - 1 + pos
                    token_log_probs = torch.log_softmax(logits[logit_pos].float(), dim=-1)
                    lp += token_log_probs[tok_id].item()
                log_probs.append(lp / len(cand_ids))

            lp_arr = np.array(log_probs, dtype=np.float64)
            lp_arr -= lp_arr.max()
            p = np.exp(lp_arr)
            p /= p.sum()
            probas[idx] = p
            preds.append(classes[int(p.argmax())])

            if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
                print(f"[{task}] {idx + 1}/{len(df)} tweets traités sur {device}", flush=True)

    return preds, probas, 0


def main():
    parser = argparse.ArgumentParser(description="Classification zero-shot par LLM sur CrisisMMD.")
    parser.add_argument("task", nargs="?", default="informative", choices=list(LABELS))
    parser.add_argument("--backend", default="ollama", choices=["ollama", "hf"])
    parser.add_argument("--model", default=None, help="Nom exact du modèle (ex: llama3.2:latest ou Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--export-protocol-b", action="store_true", help="Exporte et évalue aussi les sous-ensembles de test du protocole B")
    args = parser.parse_args()

    task = args.task
    classes = LABELS[task]
    model_name = args.model or (DEFAULT_OLLAMA_MODEL if args.backend == "ollama" else DEFAULT_HF_MODEL)
    system_prompt, user_template = load_prompts(task)

    test_df = get_split(task, "A", "official", text_only=True)["test"]
    print(f"Tâche : {task} · Split test : {len(test_df)} exemples · Backend : {args.backend} ({model_name})", flush=True)

    if args.backend == "ollama":
        preds, probas, invalid_count = predict_ollama(test_df, task, model_name, system_prompt, user_template, classes)
    else:
        preds, probas, invalid_count = predict_hf(test_df, task, model_name, system_prompt, user_template, classes)

    pred_path = write_predictions(
        test_df, preds, probas, task=task, model="llm", protocol="A", fold="official"
    )
    metric_path = evaluate_file(pred_path)
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))

    print(f"\nPrédictions protocole A : {pred_path.relative_to(ROOT)}")
    print(f"Métriques protocole A   : {metric_path.relative_to(ROOT)}")
    if invalid_count > 0:
        print(f"Réponses hors-format ramenées à la classe de repli ({FALLBACK_CLASS[task]}) : {invalid_count}/{len(test_df)}")

    ci = metrics["bootstrap"]["intervals"]["f1_macro"]
    print(
        f"F1 macro : {metrics['f1_macro']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}] · "
        f"F1 pondéré : {metrics['f1_weighted']:.4f} · Accuracy : {metrics['accuracy']:.4f}\n"
    )

    if args.export_protocol_b:
        full_pred_df = pd.read_csv(pred_path, dtype=str, keep_default_na=False)
        proba_cols = [f"proba_{label}" for label in classes]
        for event_name in EVENTS:
            b_test = get_split(task, "B", event_name, text_only=True)["test"]
            sub = full_pred_df.loc[full_pred_df["event_name"].eq(event_name)].reset_index(drop=True)
            b_pred_path = write_predictions(
                b_test,
                sub["y_pred"].to_numpy(),
                sub[proba_cols].to_numpy(dtype=float),
                task=task,
                model="llm",
                protocol="B",
                fold=event_name,
            )
            evaluate_file(b_pred_path)
        print(f"Sous-ensembles et métriques du protocole B exportés pour les {len(EVENTS)} événements.")


if __name__ == "__main__":
    main()
