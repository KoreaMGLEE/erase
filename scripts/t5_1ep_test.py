import torch, json, numpy as np, sys
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_constant_schedule
from datasets import load_dataset
from collections import Counter

torch.manual_seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

mode = sys.argv[1]       # sentinel or answer
scheduler_type = sys.argv[2]  # constant, warmup, linear
gpu = int(sys.argv[3])
split_name = sys.argv[4]

tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-base', legacy=True)
ds = load_dataset('glue', 'mnli')
train_ds, val_ds = ds['train'], ds['validation_matched']
LABELS = ['yes', 'maybe', 'no']
first_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS}
sentinel_id = tokenizer.encode('<extra_id_0>', add_special_tokens=False)[0]

with open(f'/workspace/erase/outputs/plan0/data/{split_name}_indices.json') as f:
    indices = json.load(f)

device = torch.device(f'cuda:{gpu}')
model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-base', torch_dtype=torch.float32).to(device)
optimizer = Adafactor(model.parameters(), lr=5e-5, scale_parameter=False, relative_step=False, warmup_init=False)
spe = len(indices) // 16
total_steps = spe

if scheduler_type == 'constant':
    scheduler = get_constant_schedule(optimizer)
elif scheduler_type == 'warmup':
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps))
elif scheduler_type == 'linear':
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

perm = np.random.RandomState(1).permutation(len(indices))
model.train()
for step in range(spe):
    bi = [indices[perm[(step*16+k)%len(indices)]] for k in range(16)]
    if mode == 'sentinel':
        inps = [f'mnli hypothesis: {train_ds[i]["hypothesis"]} premise: {train_ds[i]["premise"]} answer: <extra_id_0>' for i in bi]
        tgts = [f'<extra_id_0> {LABELS[train_ds[i]["label"]]}' for i in bi]
    else:
        inps = [f'mnli hypothesis: {train_ds[i]["hypothesis"]} premise: {train_ds[i]["premise"]} answer:' for i in bi]
        tgts = [LABELS[train_ds[i]["label"]] for i in bi]
    enc = tokenizer(inps, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
    tgt = tokenizer(tgts, padding=True, truncation=True, max_length=16, return_tensors='pt').input_ids.to(device)
    tgt[tgt == tokenizer.pad_token_id] = -100
    out = model(**enc, labels=tgt)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); scheduler.step(); optimizer.zero_grad()

model.eval()
correct = total = 0
pred_counts = Counter()
with torch.no_grad():
    for i in range(0, len(val_ds), 16):
        batch = val_ds[i:i+16]
        if mode == 'sentinel':
            inps = [f'mnli hypothesis: {h} premise: {p} answer: <extra_id_0>' for h,p in zip(batch['hypothesis'], batch['premise'])]
        else:
            inps = [f'mnli hypothesis: {h} premise: {p} answer:' for h,p in zip(batch['hypothesis'], batch['premise'])]
        enc = tokenizer(inps, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
        bs = enc['input_ids'].size(0)
        if mode == 'sentinel':
            dec = torch.full((bs,2), tokenizer.pad_token_id, dtype=torch.long, device=device)
            dec[:,1] = sentinel_id
            out = model(**enc, decoder_input_ids=dec)
            logits = out.logits[:,1,:]
        else:
            dec = torch.full((bs,1), tokenizer.pad_token_id, dtype=torch.long, device=device)
            out = model(**enc, decoder_input_ids=dec)
            logits = out.logits[:,0,:]
        lid = [first_ids[l] for l in LABELS]
        preds = logits[:, lid].argmax(dim=-1)
        for j in range(bs):
            if preds[j].item() == batch['label'][j]: correct += 1
            pred_counts[LABELS[preds[j].item()]] += 1
            total += 1
print(f'{mode}+{scheduler_type} {split_name}: acc={correct/total:.4f} dist={dict(pred_counts)}', flush=True)
