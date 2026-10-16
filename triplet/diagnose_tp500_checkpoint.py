"""Diagnose a completed TripletPower run without retraining.

Run from the TripletPower project root:
python3 -m triplet.diagnose_tp500_checkpoint --data_path "$HOME/Target 1/X1_K1_200k_L11.npz" --ckpt_dir Output/triplet_pytorch/profiling/tp500_true_semihard_rmsprop/ckpt --n_traces 500 --attack_size 5000 --tracewindow 1800 2800 --target_byte 2 --seed 42 --val_size 500 --expected_pairs 26128

Copy this file to triplet/diagnose_tp500_checkpoint.py first.
"""

from pathlib import Path
from collections import Counter
import argparse
import gc
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier

from tools.ascad_loader import load_dataset, dissemble_data_dict
from tools.loadData import get_labels
from tools.model_zoo_pytorch import build_cnn_best
from triplet.triplet_pytorch import extract_embeddings


def distribution(y):
    return dict(sorted(Counter(np.asarray(y, dtype=np.int64).tolist()).items()))


def stored_labels_to_hw(stored, path):
    arr=np.asarray(stored).reshape(-1)
    if str(path).lower().endswith('.h5'):
        return np.asarray([int(v).bit_count() for v in arr], dtype=np.int64)
    return arr.astype(np.int64)


def key_as_hex(key):
    arr=np.asarray(key)
    if arr.ndim!=1:
        raise ValueError(f'This diagnostic requires a fixed 1D key; got key shape={arr.shape}')
    return ' '.join(f'{int(v):02x}' for v in arr)


def build_pairs(y):
    anchors=[]
    positives=[]
    for cls in np.unique(y):
        ids=np.flatnonzero(y==cls)
        ii,jj=np.triu_indices(len(ids), k=1)
        anchors.append(ids[ii])
        positives.append(ids[jj])
    return np.concatenate(anchors), np.concatenate(positives)


def mining_diagnostic(embeddings, labels, margin):
    vectors=F.normalize(torch.from_numpy(embeddings).float(), dim=1).numpy()
    sims=vectors@vectors.T
    anchors,positives=build_pairs(labels)
    total=len(anchors)
    count_available=0
    for start in range(0,total,512):
        a=anchors[start:start+512]
        p=positives[start:start+512]
        sim_ap=sims[a,p]
        sim_an=sims[a]
        other=(labels[a,None]!=labels[None,:])
        is_semihard=(sim_an>sim_ap[:,None]-margin)&(sim_an<sim_ap[:,None])&other
        count_available+=int(np.any(is_semihard,axis=1).sum())
    same=(labels[:,None]==labels[None,:])
    is_upper=np.triu(np.ones_like(same, dtype=bool),k=1)
    mean_pos=float(sims[same & is_upper].mean())
    mean_neg=float(sims[(~same) & is_upper].mean())
    masked=sims.copy()
    masked[same]=-np.inf
    hardest_negative_sim=masked.max(axis=1)[anchors]
    positive_sim=sims[anchors,positives]
    hard_fraction=float(np.mean(hardest_negative_sim>=positive_sim))
    margin_fraction=float(np.mean(hardest_negative_sim>positive_sim-margin))
    hardest_loss=float(np.maximum(hardest_negative_sim-positive_sim+margin,0).mean())
    print(f'  Mining state: positive_pairs={total}, semihard_available={count_available}/{total} ({count_available/total:.2%}), fallback_needed={(total-count_available)/total:.2%}')
    print(f'  Hardest-negative check: pairs_with_hard_negative={hard_fraction:.2%}, pairs_violating_margin={margin_fraction:.2%}, hardest_triplet_loss={hardest_loss:.4f}')
    print(f'  Training cosine similarity: same_label_mean={mean_pos:.4f}, different_label_mean={mean_neg:.4f}')


def loo_knn_accuracy(embeddings, labels, k):
    z=F.normalize(torch.from_numpy(embeddings).float(),dim=1).numpy()
    distances=np.maximum(1-z@z.T,0.0)
    np.fill_diagonal(distances,np.inf)
    ids=np.argpartition(distances,kth=k-1,axis=1)[:,:k]
    selected_dist=np.take_along_axis(distances,ids,axis=1)
    zero=(selected_dist<=1e-10)
    weights=1.0/np.maximum(selected_dist,1e-12)
    contains_zero=zero.any(axis=1)
    weights[contains_zero]=zero[contains_zero].astype(np.float64)
    classes=np.unique(labels)
    sums=np.column_stack([(weights*(labels[ids]==cls)).sum(axis=1) for cls in classes])
    pred=classes[np.argmax(sums,axis=1)]
    return float(np.mean(pred==labels))


def embedding_summary(name, embeddings):
    norms=np.linalg.norm(embeddings,axis=1)
    dead=float(np.mean(norms<1e-8))
    active=float(np.mean(embeddings>1e-8))
    print(f'  {name} embeddings: shape={embeddings.shape}, mean_norm={norms.mean():.4f}, zero_vector_rate={dead:.2%}, positive_value_fraction={active:.2%}')


def main():
    parser=argparse.ArgumentParser(description='Post-hoc TripletPower kNN / mining / label diagnostics (no training)')
    parser.add_argument('--data_path',type=str,required=True)
    parser.add_argument('--ckpt_dir',type=Path,required=True)
    parser.add_argument('--n_traces',type=int,default=500)
    parser.add_argument('--val_size',type=int,default=500)
    parser.add_argument('--attack_size',type=int,default=5000)
    parser.add_argument('--tracewindow',type=int,nargs=2,default=[1800,2800])
    parser.add_argument('--target_byte',type=int,default=2)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--n_neighbors',type=int,default=10)
    parser.add_argument('--margin',type=float,default=0.5)
    parser.add_argument('--expected_pairs',type=int,default=None)
    args=parser.parse_args()
    if args.n_neighbors>=args.n_traces:
        raise ValueError('n_neighbors must be less than n_traces for leave-one-out evaluation')
    np.random.seed(args.seed)
    print('=== Load profiling data ===')
    p_data=load_dataset(data_path=args.data_path,attack_size=args.attack_size,which_one='train')
    x_all,stored_train,pt_all,key_train=dissemble_data_dict(data_dict=p_data,tracewindow=tuple(args.tracewindow),which_one='train')
    train_indices=np.random.choice(len(x_all),size=args.n_traces,replace=False)
    remaining=np.setdiff1d(np.arange(len(x_all)),train_indices,assume_unique=True)
    if args.val_size>len(remaining):
        raise ValueError('Insufficient remaining profiling traces for val_size')
    val_indices=np.random.default_rng(args.seed+2026).choice(remaining,size=args.val_size,replace=False)
    x_train=np.asarray(x_all[train_indices],dtype=np.float32).copy()
    x_val=np.asarray(x_all[val_indices],dtype=np.float32).copy()
    pt_train=np.asarray(pt_all[train_indices]).copy()
    pt_val=np.asarray(pt_all[val_indices]).copy()
    stored_train_selected=np.asarray(stored_train)[train_indices].copy()
    stored_val_selected=np.asarray(stored_train)[val_indices].copy()
    train_key=np.asarray(key_train).copy()
    y_train=np.asarray(get_labels(pt_train,int(train_key[args.target_byte]),args.target_byte,'HW'),dtype=np.int64)
    y_val=np.asarray(get_labels(pt_val,int(train_key[args.target_byte]),args.target_byte,'HW'),dtype=np.int64)
    train_match=float(np.mean(stored_labels_to_hw(stored_train_selected,args.data_path)==y_train))
    val_match=float(np.mean(stored_labels_to_hw(stored_val_selected,args.data_path)==y_val))
    print('  Profiling key:',key_as_hex(train_key))
    print(f'  Train HW label match={train_match:.4f}, val HW label match={val_match:.4f}')
    print(f'  Train HW class counts: {distribution(y_train)}')
    print(f'  Val HW class counts: {distribution(y_val)}')
    pair_count=len(build_pairs(y_train)[0])
    print(f'  Reconstructed pair count: {pair_count}')
    if args.expected_pairs is not None and pair_count!=args.expected_pairs:
        raise RuntimeError(f'Reconstructed pair count {pair_count} differs from logged {args.expected_pairs}; cannot claim to have reconstructed the same profiling subset')
    del p_data,x_all,pt_all,stored_train
    gc.collect()

    print('\n=== Load attack data ===')
    a_data=load_dataset(data_path=args.data_path,attack_size=args.attack_size,which_one='test')
    x_attack,stored_attack,pt_attack,key_attack=dissemble_data_dict(data_dict=a_data,tracewindow=tuple(args.tracewindow),which_one='test')
    attack_key=np.asarray(key_attack).copy()
    y_attack=np.asarray(get_labels(pt_attack,int(attack_key[args.target_byte]),args.target_byte,'HW'),dtype=np.int64)
    attack_match=float(np.mean(stored_labels_to_hw(stored_attack,args.data_path)==y_attack))
    print('  Attack key:',key_as_hex(attack_key))
    print(f'  Profiling and attack key equal: {np.array_equal(train_key,attack_key)}')
    print(f'  Attack HW label match={attack_match:.4f}')
    print(f'  Attack HW class counts: {distribution(y_attack)}')
    majority=int(np.bincount(y_attack,minlength=9).argmax())
    majority_accuracy=float(np.mean(y_attack==majority))
    print(f'  Attack majority-class baseline: HW={majority}, accuracy={majority_accuracy:.4f}')
    x_attack=np.asarray(x_attack,dtype=np.float32)
    del a_data
    gc.collect()

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\nDevice:',device)
    model=build_cnn_best(input_shape=(x_train.shape[1],1),emb_size=256,classification=False).to(device)
    checkpoints=[args.ckpt_dir/'triplet_best.pt',args.ckpt_dir/'triplet_final.pt']
    for ckpt in checkpoints:
        if not ckpt.is_file():
            print(f'Checkpoint not found: {ckpt}')
            continue
        print(f'\n=== Checkpoint: {ckpt.name} ===')
        model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True))
        model.eval()
        emb_train=extract_embeddings(x_train,model,batch_size=256)
        emb_val=extract_embeddings(x_val,model,batch_size=256)
        emb_attack=extract_embeddings(x_attack,model,batch_size=256)
        embedding_summary('Train',emb_train)
        embedding_summary('Profiling holdout',emb_val)
        embedding_summary('Attack',emb_attack)
        mining_diagnostic(emb_train,y_train,args.margin)
        loo_acc=loo_knn_accuracy(emb_train,y_train,args.n_neighbors)
        knn=KNeighborsClassifier(n_neighbors=args.n_neighbors,weights='distance',metric='cosine',algorithm='brute')
        knn.fit(emb_train,y_train)
        pred_val=knn.predict(emb_val)
        pred_attack=knn.predict(emb_attack)
        acc_val=float(np.mean(pred_val==y_val))
        acc_attack=float(np.mean(pred_attack==y_attack))
        print(f'  Leave-one-out train kNN HW accuracy={loo_acc:.4f}')
        print(f'  Unseen profiling holdout kNN HW accuracy={acc_val:.4f}')
        print(f'  Attack kNN HW accuracy={acc_attack:.4f}')
        print(f'  Attack predicted HW distribution: {distribution(pred_attack)}')
        print(f'  Attack per-class accuracy: { {int(cls): round(float(np.mean(pred_attack[y_attack==cls]==cls)),3) for cls in np.unique(y_attack)} }')

    print('\nInterpretation: high LOO train + low profiling holdout indicates overfitting; high profiling holdout + low attack indicates distribution/key/domain shift or attack-label mismatch; high fallback fraction indicates few true semihard negatives remain at this checkpoint.')


if __name__=='__main__':
    main()
