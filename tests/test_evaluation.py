"""
Run end-to-end evaluation on fine-tuned text models
(IMDB or Yelp) and save metrics + RA analysis to JSON.

Usage
-----
python scripts/text_evaluation.py \
    --dataset imdb \
    --ckpt checkpoints/bert_imdb/best_model.pt \
    --out results/imdb_eval.json
"""
import argparse, json, os, torch, numpy as np
from tqdm import tqdm
from ra.model_factory import ModelFactory
from ra.dataset_utils import DatasetLoader
from ra.ra import ReverseAttribution
from metrics import expected_calibration_error, compute_brier_score

@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------ #
    # Model + tokenizer
    # ------------------------------------------------------------------ #
    model = ModelFactory.create_text_model(
        model_name=args.hf_model,
        num_classes=2,
        checkpoint_path=args.ckpt,
    ).to(device).eval()

    ra = ReverseAttribution(model, device=device)
    loader = DatasetLoader()
    dl = loader.create_text_dataloader(
        args.dataset, "test", model.tokenizer,
        batch_size=args.batch, shuffle=False
    )

    # holders
    y_true, y_pred, probs, ra_results = [], [], [], []

    print("⏳ running inference …")
    for batch in tqdm(dl):
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device)
        )
        p = torch.softmax(logits, -1)
        preds = p.argmax(-1).cpu().tolist()

        y_pred.extend(preds)
        probs.extend(p.cpu().numpy())
        y_true.extend(batch["labels"].tolist())

        # optional RA on errors
        if args.ra and len(ra_results) < args.ra_samples:
            for i in range(len(preds)):
                if preds[i] != batch["labels"][i].item():
                    res = ra.explain(
                        batch["input_ids"][i : i + 1],
                        y_true=batch["labels"][i].item(),
                    )
                    ra_results.append(res)
                    if len(ra_results) == args.ra_samples:
                        break

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    probs  = np.vstack(probs)
    acc    = (y_true == y_pred).mean()
    ece    = expected_calibration_error(
        np.max(probs, 1), (y_true == y_pred).astype(int)
    )
    brier  = compute_brier_score(probs, y_true)

    out = dict(
        dataset=args.dataset,
        accuracy=float(acc),
        ece=float(ece),
        brier=float(brier),
        num_samples=int(len(y_true)),
        ra_summary={
            "avg_a_flip": float(np.mean([r["a_flip"] for r in ra_results]))
            if ra_results else None,
            "evaluated": len(ra_results),
        },
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"✅  saved results to {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["imdb", "yelp"], required=True)
    p.add_argument("--hf_model", default="bert-base-uncased")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--ra", action="store_true", help="run RA on errors")
    p.add_argument("--ra_samples", type=int, default=200)
    p.add_argument("--out", required=True)
    main(p.parse_args())
