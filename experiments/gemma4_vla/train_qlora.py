"""QLoRA fine-tune Gemma 4 E2B to emit a FAST action chunk from a single image.

The answer is not text: each FAST id gets its own token in the compact vocabulary, so a 50-step
trajectory is <=59 tokens instead of ~170 digit characters. That keeps the autoregressive chain
short and makes every generated token a valid FAST id by construction.

Trained: LoRA on the language-model layers + the (tiny) embedding tables, whose action-token rows
start out random and carry all the meaning of the new tokens. lm_head is tied to embed_tokens.
"""
import argparse, json, os, random, sys, time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import AutoProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_utils import MODEL, load_compact_gemma4, load_ids  # noqa: E402

OUT = os.path.join(HERE, "out")
PROMPT = "<|image|>Predict the action trajectory.\n"


def build_example(proc, img_path, fast_tokens, remap, n_base, eos):
    """Prompt+image from the processor, then the action tokens appended as raw ids."""
    img = Image.open(img_path).convert("RGB")
    enc = proc(text=PROMPT, images=[img], return_tensors="pt")
    enc = {k: v.to("cuda") for k, v in enc.items()}
    prefix = remap[enc["input_ids"]]  # [1, P] in compact space

    answer = torch.tensor([[n_base + t for t in fast_tokens] + [eos]], device="cuda")
    enc["input_ids"] = torch.cat([prefix, answer], dim=1)
    enc["attention_mask"] = torch.cat(
        [enc["attention_mask"], torch.ones_like(answer)], dim=1
    )
    # Appended tokens are plain text (type 0), not vision.
    enc["mm_token_type_ids"] = torch.cat(
        [enc["mm_token_type_ids"], torch.zeros_like(answer)], dim=1
    )
    labels = enc["input_ids"].clone()
    labels[:, : prefix.shape[1]] = -100  # supervise the action tokens + eos only
    enc["labels"] = labels
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=450)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--out", type=str, default=os.path.join(OUT, "adapter_act"))
    args = ap.parse_args()

    pairs = json.load(open(os.path.join(OUT, "pairs.json")))
    random.seed(0)
    random.shuffle(pairs)
    pairs = pairs[: args.n_samples]
    print(f"training on {len(pairs)} samples")

    base_ids = load_ids(OUT, "base_ids.json")
    proc = AutoProcessor.from_pretrained(MODEL)
    eos = base_ids.index(proc.tokenizer.eos_token_id)
    model, remap, n_base = load_compact_gemma4(base_ids)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Language-model layers only: the vision tower wraps its Linear4bit in a custom
        # Gemma4ClippableLinear that PEFT cannot adapt.
        target_modules=r".*language_model\.layers\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    )
    model = get_peft_model(model, lora)

    # The action tokens are new, so their embedding rows must be learned. Both tables are small
    # now (2060 rows), so just train them whole; lm_head is tied to embed_tokens and comes along.
    tm = model.base_model.model.model.language_model
    embeds = [tm.embed_tokens.weight, tm.embed_tokens_per_layer.weight]
    for w in embeds:
        w.requires_grad_(True)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        random.shuffle(pairs)
        run_loss, n = 0.0, 0
        for i, p in enumerate(pairs):
            enc = build_example(proc, os.path.join(OUT, p["img"]), p["tokens"], remap, n_base, eos)
            out = model(**enc)
            (out.loss / args.grad_accum).backward()
            run_loss += out.loss.item()
            n += 1
            if (i + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                print(
                    f"ep {ep} step {step} loss {run_loss / n:.4f} "
                    f"peak_mem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB "
                    f"elapsed {time.time() - t0:.0f}s",
                    flush=True,
                )
                run_loss, n = 0.0, 0

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    # save_pretrained only stores the LoRA deltas; the retrained embeddings must go with them.
    torch.save({"embed_tokens": embeds[0].detach().cpu(),
                "embed_tokens_per_layer": embeds[1].detach().cpu()},
               os.path.join(args.out, "embeddings.pt"))
    json.dump({"prompt": PROMPT, "horizon": 50, "action_dim": 6, "n_base": n_base},
              open(os.path.join(args.out, "fmt.json"), "w"))
    print("saved adapter + embeddings to", args.out)


if __name__ == "__main__":
    main()
