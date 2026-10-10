import pandas as pd
import numpy as np
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import RMSprop
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from tqdm.auto import tqdm

from tools.ascad_loader import load_dataset, dissemble_data_dict
from tools.loadData import get_labels
from tools.model_zoo_pytorch import CNN_Best

def build_pos_pairs_for_id(classid, classid_to_ids):  # classid --> e.g. 0
    # pos_pairs is actually the combination C(10,2)
    # e.g. if we have 10 example [0,1,2,...,9]
    # and want to create a pair [a, b], where (a, b) are different and order does not matter
    # e.g. [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
    # (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)...]
    # C(10, 2) = 45
    traces = classid_to_ids[classid]
    pos_pair_list = []
    traceNum = len(traces)
    for i in range(traceNum):
        for j in range(i+1, traceNum):
            pos_pair = (traces[i], traces[j])
            pos_pair_list.append(pos_pair)
    random.shuffle(pos_pair_list)
    return pos_pair_list

def build_positive_pairs(class_id_range, classes_to_ids):
    # class_id_range = range(0, num_classes)
    listX1 = []
    listX2 = []
    for class_id in class_id_range:
        pos = build_pos_pairs_for_id(class_id, classes_to_ids)
        # -- pos [(1, 9), (0, 9), (3, 9), (4, 8), (1, 4),...] --> (anchor example, positive example)
        for pair in pos:
            listX1 += [pair[0]]  # identity
            listX2 += [pair[1]]  # positive example
    perm = np.random.permutation(len(listX1))
    # random.permutation([1,2,3]) --> [2,1,3] just random
    # random.permutation(5) --> [1,0,4,3,2]
    # In this case, we just create the random index
    # Then return pairs of (identity, positive example)
    # that each element in pairs in term of its index is randomly ordered.
    return np.array(listX1)[perm], np.array(listX2)[perm]


def sample_positive_pairs_epoch(id_2_label):
    by_class=defaultdict(list)

    for idx, cls in id_2_label.items():
        by_class[int(cls)].append(int(idx))

    pairs=[]

    for ids in by_class.values():
        if len(ids)<2:
            continue

        for i, anchor in enumerate(ids):
            j=random.randrange(len(ids)-1)

            if j>=i:
                j+=1

            pairs.append((anchor, ids[j]))

    if not pairs:
        raise ValueError("No valid anchor-positive pairs")

    random.shuffle(pairs)
    a_ids, p_ids=zip(*pairs)

    return np.asarray(a_ids, dtype=np.int64), np.asarray(p_ids, dtype=np.int64)


def build_class_index_maps(labels):
    label_2_id = defaultdict(list)
    id_2_label = {}

    for i in range(len(labels)):
        class_label = int(labels[i])

        label_2_id[class_label].append(i)
        id_2_label[i] = class_label

    return label_2_id, id_2_label

def limit_per_class(x,  labels, sample_num_limit: int,):
    original_label_2_id, _ = (build_class_index_maps(labels))

    selected_indices=[]

    for class_id, indices in original_label_2_id.items():
        if (sample_num_limit is None or sample_num_limit <= 0 or len(indices) <= sample_num_limit):
            chosen=list(indices)

        else:
            chosen=random.sample(indices, sample_num_limit)

        selected_indices.extend(chosen)

    random.shuffle(selected_indices)

    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    x_limited=x[selected_indices]
    labels_limited=labels[selected_indices]

    label_2_id, id_2_label=build_class_index_maps(labels_limited)

    return x_limited, labels_limited, label_2_id, id_2_label, selected_indices

def getCLSidDict(data_path, n_traces, attack_size, sample_num_limit, leakage_model, target_byte=2, selected_indices=None, tracewindow=(0, 700)):
    if str(data_path).endswith(".npz"):
        data_dict = load_dataset(str(data_path), attack_size=attack_size, which_one="train")
    else:
        data_dict=load_dataset(str(data_path), which_one="train")
    x,ascad_label,plain_text, key=dissemble_data_dict(data_dict, tracewindow=tracewindow, which_one="train")

    if n_traces > len(x):
        raise ValueError("the number of n_traces bigger than that of x")

    if selected_indices is None:
        if n_traces > len(x):
            raise ValueError("the number of n_traces bigger than that of x")
        selected_n_indices=np.random.choice(len(x), size=n_traces, replace=False)
    else:
        selected_n_indices=np.asarray(selected_indices, dtype=np.int64)

        if selected_n_indices.ndim != 1:
            raise ValueError("selected_indices must be a 1-D array")

        if len(selected_n_indices) != n_traces:
            raise ValueError(f"selected_indices has {len(selected_n_indices)} entries, expected n_traces={n_traces}")

        if selected_n_indices.min() < 0 or selected_n_indices.max() >= len(x):
            raise ValueError("selected_indices contains out-of-range indices")

    x_n=x[selected_n_indices]
    plain_text_n=plain_text[selected_n_indices]

    key_byte = key[target_byte]

    print("key shape:", np.asarray(key).shape)
    print("key:", key)  

    labels_n=get_labels(plain_text_n, key_byte, target_byte, leakage_model=leakage_model)

    stored_label = ascad_label[selected_n_indices]

    if str(data_path).endswith(".h5"):
        stored_hw = np.array(
        [bin(int(v)).count("1") for v in stored_label],
        dtype=np.int64
        )
    else:
        stored_hw = stored_label.astype(np.int64)

    match_rate = np.mean(stored_hw == labels_n)

    print("Stored-label vs computed-HW match:", match_rate)
    print( "=================================" )
    print( "ASCAD stored-label vs computed-HW match:", match_rate)

    x_limited, labels_limited, label_2_id, id_2_label, selected_u_indices = limit_per_class(x_n, labels_n, sample_num_limit)

    return ( x_n, labels_n, x_limited, labels_limited, label_2_id, id_2_label, ) 

def cosine_triplet_loss(a_embed: torch.Tensor, p_embed: torch.Tensor, n_embed: torch.Tensor, alpha: float=0.5, reduction="mean"):
    p_sim=F.cosine_similarity(a_embed, p_embed, dim=1)
    n_sim=F.cosine_similarity(a_embed, n_embed, dim=1)

    raw_loss=n_sim-p_sim+alpha
    loss=torch.clamp(raw_loss, min=0)

    if reduction=="mean":
        return loss.mean()
    elif reduction=="mean_nonzero":
        return loss.sum()/(loss>0).sum().clamp(min=1)
    else:
        raise ValueError(f"Unsupported loss reduction: {reduction}")
    

def build_similarities(model, traces: torch.Tensor, batch_size=1024):
    with torch.no_grad():
        embs_list=[]
        num_traces=traces.size(0)

        for i in range(0, num_traces, batch_size):
            batch_traces=traces[i:i+batch_size]
            batch_embs=model(batch_traces)
            embs_list.append(batch_embs)

        embs=torch.cat(embs_list, dim=0)
        embs=F.normalize(embs, p=2, dim=-1)
        all_sims=torch.matmul(embs, embs.T)

    return all_sims.detach().cpu().numpy()

def intersect(a,b):
    return list(set(a) & set(b))

def build_negatives(a_ids, p_ids, neg_ids, id_2_label, 
                alpha_mine, all_sims=None, num_retries: int=50, negative_mode="current", valid_neg_ids_by_class=None,
                stats=None, legacy_label_2_id=None, mixed_violation_prob=0.10):
    if negative_mode not in {"current", "true_semihard", "random_valid", "tf_legacy", "mixed_valid"}:
        raise ValueError(f"unsupported negative_mode: {negative_mode}")

    if stats is None:
        stats=defaultdict(int)

    if negative_mode == "tf_legacy":

        if legacy_label_2_id is None:
            raise ValueError(
                "tf_legacy mode requires legacy_label_2_id"
            )

        final_neg = []

        if all_sims is None:

            sampled = random.sample(neg_ids, len(a_ids))

            for a_id, neg_id in zip(a_ids, sampled):
                stats["total"] += 1
                stats["legacy_random"] += 1
                if (id_2_label[neg_id] == id_2_label[a_id]):
                    stats["same_class_negative"] += 1

            return sampled

        for a_id, p_id in zip(a_ids, p_ids):
            legacy_anchor_class = ( legacy_label_2_id[a_id] )

            pos_sim = all_sims[ a_id, p_id, ]

            possible_ids = np.where( ( all_sims[a_id] + alpha_mine ) > pos_sim )[0]

            possible_ids = intersect( neg_ids, possible_ids, )

            neg_id = None

            for _ in range(num_retries):

                if len(possible_ids) == 0:
                    break

                candidate = random.choice( possible_ids )
                # EXACT legacy-source class test
                if ( legacy_label_2_id[candidate] != legacy_anchor_class ):
                    neg_id = candidate
                    stats["legacy_candidate"] += 1
                    break

            # Old unrestricted fallback
            if neg_id is None:

                neg_id = random.choice(neg_ids)

                stats["legacy_fallback"] += 1

            final_neg.append(int(neg_id))

            stats["total"] += 1

            # ====================================================
            # Everything below is DIAGNOSTIC ONLY.
            # Does NOT change selected negative.
            # ====================================================

            actual_anchor_class = (id_2_label[a_id])

            actual_neg_class = (id_2_label[neg_id])
            if (actual_anchor_class == actual_neg_class): 
                stats["same_class_negative"] += 1

            neg_sim = all_sims[a_id, neg_id]

            if neg_sim >= pos_sim:
                stats["hard"] += 1

            elif ( neg_sim > pos_sim - alpha_mine):
                stats["semihard"] += 1

            else:
                stats["easy"] += 1

        assert (len(final_neg) == len(a_ids))

        return final_neg

    if valid_neg_ids_by_class is None:
        classes=set(id_2_label.values())

        valid_neg_ids_by_class={class_id:[idx for idx in neg_ids if id_2_label[idx] != class_id] for class_id in classes}    

    final_neg=[]    

    def choose_random_valid(anchor_class):
        candidates=valid_neg_ids_by_class[anchor_class]

        if len(candidates)==0:
            raise RuntimeError(f"no valid negative exists for class {anchor_class}")

        return random.choice(candidates)
    # ============================================================
    # Epoch 0:
    # no similarity matrix yet
    # ============================================================
    if all_sims is None:
        if negative_mode == "current":
            sampled=random.sample(neg_ids, len(a_ids))

            for a_id, neg_id in zip(a_ids, sampled,):
                stats['total']+=1
                stats["random_unfiltered"]+=1

                if (id_2_label[neg_id] == id_2_label[a_id]): 
                    stats["same_class_negative"] += 1

            return sampled

        for a_id in a_ids:

            anchor_class = id_2_label[a_id]
            neg_id = choose_random_valid(anchor_class)

            final_neg.append(neg_id)

            stats["total"] += 1
            stats["random_valid"] += 1
            if negative_mode == "mixed_valid":
                stats["mixed_random_valid"]+=1

        return final_neg    

    # ============================================================
    # Epoch >= 1
    # ============================================================
    for a_id, p_id in zip(a_ids, p_ids):
        anchor_class=id_2_label[a_id]
        pos_sim=all_sims[a_id, p_id]

        # ========================================================
        # MODE A:
        # CURRENT IMPLEMENTATION
        # ========================================================
        if negative_mode=="current":
            possible_ids=np.where((all_sims[a_id]+alpha_mine) > pos_sim)[0]
            possible_ids=intersect(possible_ids, neg_ids)

            appended=False

            for _ in range(num_retries):
                if len(possible_ids) == 0:
                    break

                neg_id=random.choice(possible_ids)

                if (id_2_label[neg_id]!=anchor_class):
                    final_neg.append(neg_id)
                    neg_sim=all_sims[a_id, neg_id]

                    stats["total"]+=1

                    if neg_sim >= pos_sim:
                        stats["hard"]+=1
                    else:
                        stats["semihard"]+=1

                    appended=True
                    break

            if not appended:
                neg_id=random.choice(neg_ids)
                final_neg.append(neg_id)
                stats["total"]+=1
                stats["fallback_random"]+=1

                if id_2_label[neg_id]==anchor_class:
                    stats["same_class_negative"]+=1
        # ========================================================
        # MODE B:
        # TRUE SEMI-HARD
        #
        # s_pos - alpha < s_neg < s_pos
        # ========================================================
        elif(negative_mode=="true_semihard"):
            similarities=all_sims[a_id]
            possible_ids=np.where(((similarities+alpha_mine)>pos_sim) & (similarities < pos_sim))[0]

            possible_ids=intersect(possible_ids, neg_ids)

            possible_ids=[idx for idx in possible_ids if (id_2_label[idx]!=anchor_class)]

            if len(possible_ids) > 0:

                # possible_ids=sorted(possible_ids, key=lambda idx:all_sims[a_id, idx], reverse=True)
                # top_k_ids=possible_ids[:min(10, len(possible_ids))]
                neg_id=random.choice(possible_ids)

                final_neg.append(neg_id)

                stats["total"] += 1
                stats["semihard"] += 1

            else:
                neg_id = choose_random_valid(
                    anchor_class
                )

                final_neg.append(
                    neg_id
                )

                stats["total"] += 1
                stats[
                    "fallback_random_valid"
                ] += 1
        # ========================================================
        # MODE C:
        # RANDOM DIFFERENT-CLASS NEGATIVE
        # ========================================================

        elif ( negative_mode == "random_valid" ):

            neg_id = choose_random_valid( anchor_class )
            final_neg.append( neg_id )

            stats["total"] += 1
            stats["random_valid"] += 1

        # =========================================================================
        # MODE D:
        # Mixed Valid
        # =========================================================================

        elif negative_mode=="mixed_valid":
            use_violation=(random.random()<mixed_violation_prob)

            neg_id=None

            if use_violation:
                possible_ids=np.where((all_sims[a_id]+alpha_mine)>pos_sim)[0]
                possible_ids=intersect(possible_ids, neg_ids)
                possible_ids=[idx for idx in possible_ids if id_2_label[idx] != anchor_class]

                if len(possible_ids)>0:
                    neg_id=random.choice(possible_ids)
                    stats["mixed_violation"]+=1
                else:
                    stats["mixed_violation_fallback"]+=1

            if neg_id is None:

                neg_id = choose_random_valid( anchor_class )
                stats[ "mixed_random_valid" ] += 1
            neg_sim = all_sims[
                    a_id,
                    neg_id
                ]

            if neg_sim >= pos_sim:
                stats["hard"] += 1

            elif ( neg_sim > pos_sim - alpha_mine ):

                stats["semihard"] += 1

            else:
                stats["easy"] += 1

            final_neg.append( int(neg_id) )
            stats["total"] += 1                      

    assert len(final_neg) == len(a_ids), (
            f"build_negatives mismatch: "
            f"anchors={len(a_ids)}, "
            f"negatives={len(final_neg)}"
        )

    return final_neg

class AnchorPositiveDataset(Dataset):
    def __init__(self, a_ids, p_ids):
        self.a_ids=list(a_ids)
        self.p_ids=list(p_ids)

        if len(self.a_ids) != len(self.p_ids):
            raise ValueError("the length of two list is not equal")

    def __len__(self):
        return len(self.a_ids)

    def __getitem__(self, idx):
        return (int(self.a_ids[idx]), int(self.p_ids[idx]))

class TripletBatchCollator():
    def __init__(self, all_traces, neg_ids, id_2_label, alpha_mine=0.5, all_sims=None, negative_mode="current", legacy_label_2_id=None, mixed_violation_prob=0.10):
        self.all_traces=all_traces
        self.neg_ids=neg_ids
        self.id_2_label=id_2_label
        self.alpha_mine=alpha_mine
        self.all_sims=all_sims 
        self.negative_mode=negative_mode
        self.stats=defaultdict(int)

        classes=sorted(set(id_2_label.values()))

        self.valid_neg_ids_by_class={class_id: [idx for idx in neg_ids if (id_2_label[idx]!=class_id)] for class_id in classes} 
        self.legacy_label_2_id = (legacy_label_2_id)
        self.mixed_violation_prob=mixed_violation_prob   

    def __call__(self, batch):
        a_ids=[item[0] for item in batch]
        p_ids=[item[1] for item in batch]

        n_ids = build_negatives(
            a_ids=a_ids,
            p_ids=p_ids,
            neg_ids=self.neg_ids,
            id_2_label=self.id_2_label,
            alpha_mine=self.alpha_mine,
            all_sims=self.all_sims,
            negative_mode=self.negative_mode,
            valid_neg_ids_by_class=self.valid_neg_ids_by_class,
            stats=self.stats,
            legacy_label_2_id=self.legacy_label_2_id,
            mixed_violation_prob=self.mixed_violation_prob
        )

        assert len(a_ids)==len(p_ids)==len(n_ids), f"Triplet batch mismatch: anchors={len(a_ids)}, positives={len(p_ids)}, negatives={len(n_ids)}"

        a_batch=self.all_traces[a_ids]
        p_batch=self.all_traces[p_ids]
        n_batch=self.all_traces[n_ids]

        return (torch.as_tensor(a_batch, dtype=torch.float32).unsqueeze(-1), torch.as_tensor(p_batch, dtype=torch.float32).unsqueeze(-1), \
                torch.as_tensor(n_batch, dtype=torch.float32).unsqueeze(-1))

def train_one_epoch(model, dataloader, optimizer, device, epoch=None, alpha_value=0.5, loss_reduction="mean"):
    model.train()

    running_loss=0.0
    pos_sim_sum=0.0
    neg_sim_sum=0.0
    violation_count=0
    sample_count=0
    head_grad_sum=0.0
    conv_grad_sum=0.0
    grad_checks=0

    progress_bar = tqdm( dataloader, desc=f"Epoch {epoch}" if epoch is not None else "Training", leave=False, dynamic_ncols=True)

    for barch_idx, (a, p, n) in enumerate(progress_bar):
        a,p,n=a.to(device), p.to(device), n.to(device)
        optimizer.zero_grad()

        anchor=model(a)
        positive=model(p)
        negative=model(n)

        with torch.no_grad():
            p_sim=F.cosine_similarity(anchor, positive, dim=1)
            n_sim=F.cosine_similarity(anchor, negative, dim=1)
            pos_sim_sum+=p_sim.sum().item()
            neg_sim_sum+=n_sim.sum().item()
            violation_count+=((n_sim-p_sim+alpha_value)>0).sum().item()
            sample_count+=a.size(0)


        loss=cosine_triplet_loss(anchor, positive, negative, alpha_value, reduction=loss_reduction)

        loss.backward()
        if barch_idx in (0, len(dataloader)//2, len(dataloader)-1):
            head_grad=model.output_layer[0].weight.grad
            conv_grad=model.feature_extractor[1].weight.grad

            head_grad_sum+=head_grad.detach().norm().item() if head_grad is not None else 0.0
            conv_grad_sum+=conv_grad.detach().norm().item() if conv_grad is not None else 0.0
            grad_checks+=1
        optimizer.step()
        running_loss+=loss.item()

    avg_loss=running_loss/len(dataloader)
    mean_pos_sim=pos_sim_sum/sample_count
    mean_neg_sim=neg_sim_sum/sample_count
    mean_gap=mean_pos_sim-mean_neg_sim
    violation_fraction=violation_count/sample_count

    diagnostics={"mean_pos_sim":mean_pos_sim,"mean_neg_sim":mean_neg_sim,"mean_gap":mean_gap,"violation_fraction":violation_fraction}
    diagnostics["head_grad_norm"]=head_grad_sum/max(grad_checks,1)
    diagnostics["conv_grad_norm"]=conv_grad_sum/max(grad_checks,1)

    return avg_loss, diagnostics

def train_tripletpower(model, all_traces, a_ids, p_ids, id_2_label, device, ckpt_path, epochs=100, batch_size=100,\
                    learning_rate=1e-5, alpha_value=0.5, alpha_mine=None, negative_mode="current", legacy_label_2_id=None,
                    mixed_violation_prob=0.10, loss_reduction="mean", pair_mode="all_pairs", validation_fn=None, val_every_steps=100, pair_swap=False, pair_swap_seed=42):
    if alpha_mine is None:
        alpha_mine=alpha_value

    print(f"Triplet loss reduction: {loss_reduction}")    
    best_loss=10.0

    Path(ckpt_path).parent.mkdir(exist_ok=True, parents=True)

    model.to(device)

    if pair_mode=="dynamic":
        class_counts=defaultdict(int)

        for cls in id_2_label.values():
            class_counts[cls]+=1

        neg_ids=[idx for idx, cls in id_2_label.items() if class_counts[cls]>=2]
    else:
        neg_ids=list(set(a_ids)|set(p_ids))

    optimizer=torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9, eps=1e-7, momentum=0.0, centered=False)
    dataset=AnchorPositiveDataset(a_ids, p_ids) if pair_mode=="all_pairs" else None
    if pair_swap and pair_mode!="all_pairs":
        raise ValueError("pair_swap requires pair_mode=all_pairs")

    swap_rng=np.random.default_rng(pair_swap_seed)

    all_traces_tensor=torch.from_numpy(all_traces).float().unsqueeze(-1).to(device)
    loss_log=[]
    global_step=0
    best_val_rank=float("inf")

    epoch_bar = tqdm(range(epochs), desc="TripletPower", dynamic_ncols=True)

    for epoch in epoch_bar:

        if epoch==0:
            all_sims=None

        else:
            model.eval()
            all_sims=build_similarities(model, all_traces_tensor)

        if pair_mode=="dynamic":
            a_epoch, p_epoch=sample_positive_pairs_epoch(id_2_label)
            dataset=AnchorPositiveDataset(a_epoch, p_epoch)

        elif pair_swap:
            a_epoch=np.asarray(a_ids).copy()
            p_epoch=np.asarray(p_ids).copy()

            swap=swap_rng.random(len(a_epoch))<0.5

            a_epoch[swap], p_epoch[swap]=p_epoch[swap].copy(), a_epoch[swap].copy()

            dataset=AnchorPositiveDataset(a_epoch, p_epoch)

            tqdm.write(f"[epoch {epoch}] A/P swapped={swap.mean():.2%}")
            
        collator_fn=TripletBatchCollator(all_traces, neg_ids, id_2_label, alpha_mine, 
                                        all_sims, negative_mode=negative_mode, 
                                        legacy_label_2_id=legacy_label_2_id, mixed_violation_prob=mixed_violation_prob)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator_fn, drop_last=(pair_mode=="all_pairs"))

        loss, diagnostics=train_one_epoch(model, loader, optimizer, device, epoch=epoch, alpha_value=alpha_value, loss_reduction=loss_reduction)
        tqdm.write(f"[epoch {epoch}] grad | head={diagnostics['head_grad_norm']:.3e} | conv1={diagnostics['conv_grad_norm']:.3e}")
        previous_step=global_step
        global_step+=len(loader)

        tqdm.write(f"[epoch {epoch}] pairs={len(dataset)} | updates={len(loader)} | global_step={global_step}")
        stats = collator_fn.stats

        total = max(
            stats["total"],
            1,
        )

        tqdm.write(
            f"[epoch {epoch}] "
            f"mining={negative_mode} | "
            f"hard={stats['hard']/total:.3f} | "
            f"semihard={stats['semihard']/total:.3f} | "
            f"easy={stats['easy']/total:.3f} | "
            f"fallback_valid={stats['fallback_random_valid']/total:.3f} | "
            f"mixed_violation={stats['mixed_violation']/total:.3f} | "
            f"mixed_fallback={stats['mixed_violation_fallback']/total:.3f} | "
            f"mixed_random={stats['mixed_random_valid']/total:.3f} | "
            f"same_class={stats['same_class_negative']/total:.3f}"
        )
        tqdm.write(f"[epoch {epoch}] margins | loss_alpha={alpha_value:.3f} | mine_alpha={alpha_mine:.3f}")
        tqdm.write(f"[epoch {epoch}] similarity | pos={diagnostics['mean_pos_sim']:.4f} | neg={diagnostics['mean_neg_sim']:.4f} | gap={diagnostics['mean_gap']:.4f} | violation={diagnostics['violation_fraction']:.3f}")
        
        epoch_bar.set_postfix(loss=f"{loss:.6f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        loss_log.append(loss)

        if loss < best_loss:
            old_best=best_loss
            best_loss=loss

            torch.save(model.state_dict(), ckpt_path)

            tqdm.write(f"Best loss improved: " f"{old_best:.6f} -> {best_loss:.6f}")

            tqdm.write(f"Saved checkpoint to: {ckpt_path}")

        if validation_fn is not None and ((global_step//val_every_steps > previous_step//val_every_steps) or epoch==epochs-1):
            model.eval()
            val_rank=validation_fn(model, epoch, global_step, diagnostics)

            if val_rank<best_val_rank:
                best_val_rank=val_rank
                val_ckpt=Path(ckpt_path).parent/"triplet_val_best.pt"
                torch.save(model.state_dict(), val_ckpt)

                tqdm.write(f"[VAL BEST] step={global_step} rank={val_rank:.3f} saved={val_ckpt}")

        if (epoch+1) % 40 == 0:
            learning_rate /= 2

            for param in optimizer.param_groups:
                param["lr"]=learning_rate

        model.eval()
        with torch.no_grad():
            emb = model(
                all_traces_tensor
            )

            norms = torch.linalg.vector_norm(
                emb,
                dim=1,
            )

        zero_fraction = (
            norms < 1e-8
        ).float().mean().item()

        print(
            f"[epoch {epoch}] "
            f"embedding norm mean="
            f"{norms.mean().item():.6f}, "
            f"zero fraction="
            f"{zero_fraction:.6f}"
        )

    final_ckpt_path = (Path(ckpt_path).parent / "triplet_final.pt")
    torch.save(model.state_dict(), final_ckpt_path,)

    print(f"Saved FINAL TripletPower model to: " f"{final_ckpt_path}")

    model.eval()

    print(
        f"Training finished. Returning FINAL epoch model. "
        f"Best observed training loss={best_loss:.6f}"
    )

    return model, loss_log

def extract_embeddings(traces, model, batch_size:int=1024) -> np.ndarray:
    model.eval()

    traces=torch.as_tensor(traces, dtype=torch.float32,)

    if traces.ndim==2:
        traces=traces.unsqueeze(-1)

    device=next(model.parameters()).device
    num_traces=traces.size(0)
    embs_list=[]

    with torch.no_grad():
        for i in range(0, num_traces, batch_size):
            batch_traces=traces[i:i+batch_size].to(device)
            embed=model(batch_traces)
            embs_list.append(embed.cpu())

    return torch.cat(embs_list, dim=0).numpy()

def train_knn(model,traces,labels:np.ndarray,n_neighbors=10,leakage_model="HW"):
    embeddings=extract_embeddings(traces,model)

    assert embeddings.ndim==2 and embeddings.shape[1]==model.output_dim,"wrong dimension of embeddings."
    assert labels.ndim==1,"wrong dimension of labels."
    assert len(embeddings)==len(labels)

    num_classes=9 if leakage_model=="HW" else 256
    missing_classes=sorted(set(range(num_classes))-set(labels.astype(int)))

    if missing_classes:
        print(f"[WARNING] Missing classes in kNN training: {missing_classes}")

    classifier=KNeighborsClassifier(n_neighbors=n_neighbors,weights="distance",metric="cosine",algorithm="brute")
    classifier.fit(embeddings,labels)

    print("KNN classes:",classifier.classes_)

    return classifier

def predict_knn_prob(model: CNN_Best, classifier:KNeighborsClassifier, traces, leakage_model="HW"):
    embeddings=extract_embeddings(traces, model)
    raw_probabilities=classifier.predict_proba(embeddings)

    if leakage_model == "HW":
        num_classes = 9
    elif leakage_model == "ID":
        num_classes = 256
    else:
        raise ValueError(
            f"Unsupported leakage model: {leakage_model}"
        )

    classes = classifier.classes_.astype(int)

    expected_classes = set(range(num_classes))
    actual_classes = set(classes)

    missing_classes = sorted( expected_classes - actual_classes )

    if missing_classes:
        print( f"[WARNING] Missing KNN classes: " f"{missing_classes}" )

    probabilities = np.zeros((raw_probabilities.shape[0], num_classes), dtype=raw_probabilities.dtype )

    probabilities[:, classes] = raw_probabilities

    print("KNN classes:", classes)
    print( "Raw probability shape:", raw_probabilities.shape )
    print( "Final probability shape:", probabilities.shape )

    return probabilities



