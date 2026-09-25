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

def getCLSidDict(data_path, n_traces, sample_num_limit, leakage_model, target_byte=2, selected_indices=None):

    data_dict=load_dataset(str(data_path), which_one="train")
    x,_,plain_text, key=dissemble_data_dict(data_dict, tracewindow=(0, 700), which_one="train")

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

    labels_n=get_labels(plain_text_n, key_byte, target_byte, leakage_model=leakage_model)

    x_limited, labels_limited, label_2_id, id_2_label, selected_u_indices = limit_per_class(x_n, labels_n, sample_num_limit)

    return ( x_n, labels_n, x_limited, labels_limited, label_2_id, id_2_label, ) 

def cosine_triplet_loss(a_embed: torch.Tensor, p_embed: torch.Tensor, n_embed: torch.Tensor, alpha: float=0.5):
    p_sim=F.cosine_similarity(a_embed, p_embed, dim=1)
    n_sim=F.cosine_similarity(a_embed, n_embed, dim=1)

    raw_loss=n_sim-p_sim+alpha
    loss=torch.clamp(raw_loss, min=0)

    return loss.mean()

def build_similarities(model, traces: torch.Tensor):
    with torch.no_grad():
        embs=model(traces)

        embs=F.normalize(embs, p=2, dim=-1)
        all_sims=torch.matmul(embs, embs.T)

    return all_sims.detach().cpu().numpy()

def intersect(a,b):
    return list(set(a) & set(b))

def build_negatives(a_ids, p_ids, neg_ids, id_2_label, alpha_value, all_sims=None, num_retries: int=50):
    if all_sims is None:
        return random.sample(neg_ids, len(a_ids))

    final_neg=[]

    for (a_id, p_id) in zip(a_ids, p_ids):
        a_class=id_2_label[a_id]

        sim=all_sims[a_id, p_id]

        possible_ids=np.where((all_sims[a_id]+alpha_value) > sim)[0]
        possible_ids=intersect(possible_ids, neg_ids)

        appended=False

        for i in range(num_retries):
            if len(possible_ids)==0:
                break

            neg_id=random.choice(possible_ids)

            if id_2_label[neg_id] != a_class:
                final_neg.append(neg_id)
                appended=True

                break

        if not appended:
            neg_id=random.choice(neg_ids)
            final_neg.append(neg_id)

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
    def __init__(self, all_traces, neg_ids, id_2_label, alpha_value=0.5, all_sims=None):
       self.all_traces=all_traces
       self.neg_ids=neg_ids
       self.id_2_label=id_2_label
       self.alpha_value=alpha_value
       self.all_sims=all_sims 

    def __call__(self, batch):
        a_ids=[item[0] for item in batch]
        p_ids=[item[1] for item in batch]

        n_ids=build_negatives(a_ids, p_ids, self.neg_ids, self.id_2_label, self.alpha_value, self.all_sims)

        assert len(a_ids)==len(p_ids)==len(n_ids), f"Triplet batch mismatch: anchors={len(a_ids)}, positives={len(p_ids)}, negatives={len(n_ids)}"
        a_batch=self.all_traces[a_ids]
        p_batch=self.all_traces[p_ids]
        n_batch=self.all_traces[n_ids]

        return (torch.as_tensor(a_batch, dtype=torch.float32).unsqueeze(-1), torch.as_tensor(p_batch, dtype=torch.float32).unsqueeze(-1), \
                torch.as_tensor(n_batch, dtype=torch.float32).unsqueeze(-1))

def train_one_epoch(model, dataloader, optimizer, device, epoch=None, alpha_value=0.5):
    model.train()

    running_loss=0.0

    progress_bar = tqdm( dataloader, desc=f"Epoch {epoch}" if epoch is not None else "Training", leave=False, dynamic_ncols=True)

    for barch_idx, (a, p, n) in enumerate(progress_bar):
        a,p,n=a.to(device), p.to(device), n.to(device)
        optimizer.zero_grad()

        anchor=model(a)
        positive=model(p)
        negative=model(n)

        loss=cosine_triplet_loss(anchor, positive, negative, alpha_value)

        loss.backward()

        optimizer.step()
        running_loss+=loss.item()

    avg_loss=running_loss/len(dataloader)

    return avg_loss  

def train_tripletpower(model, all_traces, a_ids, p_ids, id_2_label, device, ckpt_path, epochs=100, batch_size=100,\
                    learning_rate=1e-5, alpha_value=0.5):
    best_loss=10.0

    Path(ckpt_path).parent.mkdir(exist_ok=True, parents=True)

    model.to(device)

    neg_ids=list(set(a_ids)| set(p_ids))

    optimizer=torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9, eps=1e-7, momentum=0.0, centered=False)
    dataset=AnchorPositiveDataset(a_ids, p_ids)

    all_traces_tensor=torch.from_numpy(all_traces).float().unsqueeze(-1).to(device)
    loss_log=[]

    epoch_bar = tqdm(range(epochs), desc="TripletPower", dynamic_ncols=True)

    for epoch in epoch_bar:

        if epoch==0:
            all_sims=None

        else:
            model.eval()
            all_sims=build_similarities(model, all_traces_tensor)

        collator_fn=TripletBatchCollator(all_traces, neg_ids, id_2_label, alpha_value, all_sims)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator_fn, drop_last=True)

        loss=train_one_epoch(model, loader, optimizer, device, epoch=epoch, alpha_value=alpha_value)

        epoch_bar.set_postfix(loss=f"{loss:.6f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        loss_log.append(loss)

        if loss < best_loss:
            old_best=best_loss
            best_loss=loss

            torch.save(model.state_dict(), ckpt_path)

            tqdm.write(f"Best loss improved: " f"{old_best:.6f} -> {best_loss:.6f}")

            tqdm.write(f"Saved checkpoint to: {ckpt_path}")

        if (epoch+1) % 40 == 0:
            learning_rate /= 2

            for param in optimizer.param_groups:
                param["lr"]=learning_rate

    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    return model, loss_log

def extract_embeddings(traces, model) -> np.ndarray:
    model.eval()

    traces=torch.as_tensor(traces, dtype=torch.float32)

    if traces.ndim==2:
        traces=traces.unsqueeze(-1)

    device=next(model.parameters()).device
    traces=traces.to(device)

    with torch.no_grad():
        
        embed=model(traces)
        embed=embed.detach().cpu().numpy()

    return embed

def train_knn(model, traces, labels:np.ndarray, n_neighbors=10):
    embeddings = extract_embeddings(traces, model)

    assert embeddings.ndim == 2 and embeddings.shape[1] == model.output_dim, f"wrong dimension of embeddings."
    assert len(labels.shape) == 1, f"wrong dimension of labels."
    assert len(embeddings) == len(labels)

    classifier=KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", metric="cosine", algorithm="brute")

    classifier.fit(embeddings, labels)

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



