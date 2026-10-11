"""Extract frozen ESM-2 residue and sequence embeddings from a FASTA file."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('fasta',type=Path)
parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
parser.add_argument('--window',type=int,default=1022,help='ESM-2 positional limit excluding BOS/EOS')
parser.add_argument('--stride',type=int,default=511)
parser.add_argument('--out',type=Path,default=Path('results/esm'))
args=parser.parse_args()
if args.device=='cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA requested but unavailable; run this command outside sandbox')
records={}
for line in args.fasta.read_text().splitlines():
    if line.startswith('>'):
        key=line[1:].split()[0]; records[key]=''
    elif line.strip():
        records[key]+=line.strip()
model_id='facebook/esm2_t6_8M_UR50D'
tokenizer=AutoTokenizer.from_pretrained(model_id)
model=AutoModel.from_pretrained(model_id).to(args.device).eval()
out=args.out; out.mkdir(parents=True,exist_ok=True)

def embed(sequence):
    inputs=tokenizer(sequence,return_tensors='pt',return_special_tokens_mask=True)
    special=inputs.pop('special_tokens_mask').to(args.device).bool()
    inputs={k:v.to(args.device) for k,v in inputs.items()}
    with torch.inference_mode():
        hidden=model(**inputs).last_hidden_state
    return hidden[inputs['attention_mask'].bool()&~special].float().cpu().numpy()

def windows(length):
    starts=list(range(0,max(length-args.window,0)+1,args.stride))
    if starts[-1]+args.window<length:
        starts.append(length-args.window)
    return starts
manifest=[]
for index,(key,sequence) in enumerate(records.items()):
    # Long sequences: overlapping windows, residue embeddings averaged where windows overlap.
    total=np.zeros((len(sequence),model.config.hidden_size)); hits=np.zeros(len(sequence))
    starts=windows(len(sequence))
    for start in starts:
        part=embed(sequence[start:start+args.window])
        total[start:start+len(part)]+=part; hits[start:start+len(part)]+=1
    residues=(total/hits[:,None]).astype(np.float32)
    assert len(residues)==len(sequence)
    np.savez_compressed(out/f'target_{index}.npz',residue_embeddings=residues,
                        mean_embedding=residues.mean(axis=0))
    manifest.append({'id':key,'length':len(sequence),'file':f'target_{index}.npz','windows':len(starts),
                     'sequence_sha256':hashlib.sha256(sequence.encode()).hexdigest()})
(out/'manifest.json').write_text(json.dumps({'model':model_id,
    'revision':getattr(model.config,'_commit_hash',None),'device':args.device,
    'window':args.window,'stride':args.stride,'targets':manifest},indent=2)+'\n')
print(f'Embedded {len(records)} sequences on {args.device}')
