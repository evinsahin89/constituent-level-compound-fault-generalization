"""Auxiliary closed-set multimodal classification experiment for MCC5-THU.

This script evaluates conventional classification of the original fault-state
labels using synchronized triaxial vibration and three-phase current signals.
A multimodal convolutional neural network is trained with five-fold run-level
stratified cross-validation.

Signal normalization is estimated from training runs only within each fold,
and validation/test recordings remain physically disjoint from training.
Window-level predictions are aggregated to run-level class predictions.
"""
from pathlib import Path
from collections import OrderedDict
import gc, json, random, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
warnings.filterwarnings('ignore')
ROOT = Path('D:\\Multi-mode Fault Diagnosis Datasets of Three-phase')
PROTOCOL_DIR = ROOT / 'EXPERIMENT_PROTOCOL'
CACHE_DIR = ROOT / 'SIGNAL_CACHE_6CH'
RESULT_DIR = ROOT / 'ALL_CLASS_MULTIMODAL_RESULTS'
RESULT_DIR.mkdir(parents=True, exist_ok=True)
SPLITS = ['train', 'validation', 'P1', 'P2', 'P3', 'P4']
SEED = 42
N_FOLDS = 5
VAL_FRACTION = 0.2
FS = 12800
EXPECTED_SAMPLES = 12800 * 90
WINDOW_SAMPLES = int(FS * 2.0)
VIB_IDX = [0, 1, 2]
CUR_IDX = [3, 4, 5]
MAX_EPOCHS = 25
PATIENCE = 5
LR = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT = 0.25
BATCH_SIZE = 8 if torch.cuda.is_available() else 4
NUM_WORKERS = 0
ENCODER_DIM = 128
FUSION_DIM = 256
RUN_STATS_CHUNK = 200000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
print('Device:', DEVICE)
print('CUDA available:', torch.cuda.is_available())
print('AMP enabled:', USE_AMP)
print('Batch size:', BATCH_SIZE)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)

def make_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def load_inventory():
    run_frames, win_frames = ([], [])
    print('\n' + '=' * 100)
    print('LOAD COMPLETE INVENTORY')
    print('=' * 100)
    for split in SPLITS:
        rp = PROTOCOL_DIR / f'runs_{split}.csv'
        wp = PROTOCOL_DIR / f'windows_{split}.csv'
        if not rp.exists():
            raise FileNotFoundError(rp)
        if not wp.exists():
            raise FileNotFoundError(wp)
        r = pd.read_csv(rp)
        w = pd.read_csv(wp)
        print(f'{split:12s}: {len(r):4d} runs | {len(w):6d} windows')
        run_frames.append(r)
        win_frames.append(w)
    runs = pd.concat(run_frames, ignore_index=True)
    windows = pd.concat(win_frames, ignore_index=True)
    for c in ['run_id', 'fault_raw']:
        if c not in runs.columns:
            raise RuntimeError(f"Run manifests do not contain '{c}'.")
    for c in ['window_id', 'run_id', 'start_sample', 'end_sample']:
        if c not in windows.columns:
            raise RuntimeError(f"Window manifests do not contain '{c}'.")
    runs['run_id'] = runs['run_id'].astype(str)
    runs['fault_raw'] = runs['fault_raw'].astype(str)
    windows['run_id'] = windows['run_id'].astype(str)
    windows['window_id'] = windows['window_id'].astype(str)
    tmp = runs[['run_id', 'fault_raw']].drop_duplicates()
    conflicts = tmp.groupby('run_id').size()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise RuntimeError(f'Conflicting run labels: {conflicts.index.tolist()[:10]}')
    runs = runs.drop_duplicates('run_id').reset_index(drop=True)
    windows = windows.drop_duplicates('window_id').reset_index(drop=True)
    if len(runs) != 288:
        raise RuntimeError(f'Expected 288 runs; found {len(runs)}.')
    lengths = windows['end_sample'] - windows['start_sample']
    if not (lengths == WINDOW_SAMPLES).all():
        raise RuntimeError('Window length mismatch.')
    if set(runs.run_id) != set(windows.run_id):
        raise RuntimeError('Run/window inventory mismatch.')
    counts = runs['fault_raw'].value_counts().sort_index()
    print(f'\nTotal runs   : {len(runs)}')
    print(f'Total windows: {len(windows)}')
    print(f'N fault_raw  : {len(counts)}')
    print('\nRUNS PER CLASS')
    print(counts.to_string())
    if counts.min() < N_FOLDS:
        raise RuntimeError(f'Insufficient class support for 5-fold CV. Minimum runs/class={counts.min()}')
    class_names = sorted(runs['fault_raw'].unique())
    class_to_id = {name: i for i, name in enumerate(class_names)}
    runs['class_id'] = runs['fault_raw'].map(class_to_id).astype(int)
    pd.DataFrame({'class_id': range(len(class_names)), 'fault_raw': class_names}).to_csv(RESULT_DIR / 'class_mapping.csv', index=False, encoding='utf-8-sig')
    return (runs, windows, class_names)

def validate_cache(runs):
    print('\n' + '=' * 100)
    print('SIGNAL CACHE CHECK')
    print('=' * 100)
    bad = []
    for i, run_id in enumerate(runs.run_id, 1):
        p = CACHE_DIR / f'{run_id}.npy'
        if not p.exists():
            bad.append((run_id, 'missing'))
            continue
        try:
            a = np.load(p, mmap_mode='r')
            if a.shape != (EXPECTED_SAMPLES, 6):
                bad.append((run_id, str(a.shape)))
            else:
                _ = float(a[-1, -1])
            del a
        except Exception as e:
            bad.append((run_id, repr(e)))
        if i % 50 == 0:
            print(f'{i}/{len(runs)} cache kontrol edildi')
    if bad:
        raise RuntimeError(f'Missing or invalid cache entries: {bad[:10]}')
    print(f'\nPASS: {len(runs)} cache files validated.')

def load_or_build_run_stats(runs):
    out = RESULT_DIR / 'run_signal_stats.csv'
    if out.exists():
        s = pd.read_csv(out)
        s['run_id'] = s['run_id'].astype(str)
        needed = {'run_id', 'n_samples'}
        for c in range(6):
            needed |= {f'sum_{c}', f'sumsq_{c}'}
        if needed.issubset(s.columns) and set(s.run_id) == set(runs.run_id):
            print('\nrun_signal_stats.csv found; reusing existing statistics.')
            return s
    print('\n' + '=' * 100)
    print('BUILD PER-RUN SIGNAL STATS')
    print('=' * 100)
    rows = []
    for i, run_id in enumerate(runs.run_id, 1):
        a = np.load(CACHE_DIR / f'{run_id}.npy', mmap_mode='r')
        sums = np.zeros(6, dtype=np.float64)
        sumsqs = np.zeros(6, dtype=np.float64)
        n = 0
        for start in range(0, a.shape[0], RUN_STATS_CHUNK):
            stop = min(start + RUN_STATS_CHUNK, a.shape[0])
            x = np.asarray(a[start:stop], dtype=np.float64)
            sums += x.sum(0)
            sumsqs += np.square(x).sum(0)
            n += len(x)
        row = {'run_id': run_id, 'n_samples': n}
        for c in range(6):
            row[f'sum_{c}'] = sums[c]
            row[f'sumsq_{c}'] = sumsqs[c]
        rows.append(row)
        del a
        gc.collect()
        if i % 20 == 0 or i == len(runs):
            print(f'{i}/{len(runs)} runs processed')
    s = pd.DataFrame(rows)
    s.to_csv(out, index=False, encoding='utf-8-sig')
    return s

def fold_normalization(train_run_ids, run_stats):
    ids = set(map(str, train_run_ids))
    s = run_stats[run_stats.run_id.isin(ids)]
    if len(s) != len(ids):
        raise RuntimeError('Run statistics are missing for fold normalization.')
    n = float(s.n_samples.sum())
    sums = np.array([s[f'sum_{c}'].sum() for c in range(6)], dtype=np.float64)
    sumsqs = np.array([s[f'sumsq_{c}'].sum() for c in range(6)], dtype=np.float64)
    mean = sums / n
    var = sumsqs / n - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    return (mean.astype(np.float32), std.astype(np.float32))

def create_fold_assignments(runs):
    rows = []
    y = runs.class_id.to_numpy()
    X = np.zeros((len(runs), 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (trainval_idx, test_idx) in enumerate(skf.split(X, y), 1):
        trainval = runs.iloc[trainval_idx].reset_index(drop=True)
        test = runs.iloc[test_idx].reset_index(drop=True)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED + fold)
        ix_train, ix_val = next(sss.split(np.zeros((len(trainval), 1)), trainval.class_id.to_numpy()))
        parts = {'train': trainval.iloc[ix_train], 'validation': trainval.iloc[ix_val], 'test': test}
        for split, df in parts.items():
            for _, r in df.iterrows():
                rows.append({'fold': fold, 'split': split, 'run_id': r.run_id, 'fault_raw': r.fault_raw, 'class_id': int(r.class_id)})
    a = pd.DataFrame(rows)
    a.to_csv(RESULT_DIR / 'fold_assignments.csv', index=False, encoding='utf-8-sig')
    for fold in range(1, N_FOLDS + 1):
        f = a[a.fold == fold]
        sets = {s: set(f.loc[f.split == s, 'run_id']) for s in ['train', 'validation', 'test']}
        if sets['train'] & sets['validation'] or sets['train'] & sets['test'] or sets['validation'] & sets['test']:
            raise RuntimeError(f'Fold {fold}: leakage')
        if sets['train'] | sets['validation'] | sets['test'] != set(runs.run_id):
            raise RuntimeError(f'Fold {fold}: incomplete inventory')
        print(f"Fold {fold}: train={len(sets['train'])}, val={len(sets['validation'])}, test={len(sets['test'])}")
    return a

class MotorDataset(Dataset):

    def __init__(self, windows, run_to_class, mean, std, mmap_cache_size=32):
        self.df = windows.reset_index(drop=True)
        self.run_to_class = {str(k): int(v) for k, v in run_to_class.items()}
        self.mean = np.asarray(mean, np.float32).reshape(1, 6)
        self.std = np.asarray(std, np.float32).reshape(1, 6)
        self.cache_size = mmap_cache_size
        self.mmaps = OrderedDict()

    def __len__(self):
        return len(self.df)

    def _get(self, run_id):
        if run_id in self.mmaps:
            a = self.mmaps.pop(run_id)
            self.mmaps[run_id] = a
            return a
        a = np.load(CACHE_DIR / f'{run_id}.npy', mmap_mode='r')
        self.mmaps[run_id] = a
        if len(self.mmaps) > self.cache_size:
            self.mmaps.popitem(last=False)
        return a

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        run_id = str(r.run_id)
        start = int(r.start_sample)
        end = int(r.end_sample)
        x = np.asarray(self._get(run_id)[start:end], dtype=np.float32).copy()
        if x.shape != (WINDOW_SAMPLES, 6):
            raise RuntimeError(f'{run_id}: bad window {x.shape}')
        x = ((x - self.mean) / self.std).T
        return {'vibration': torch.from_numpy(x[VIB_IDX]), 'current': torch.from_numpy(x[CUR_IDX]), 'target': torch.tensor(self.run_to_class[run_id], dtype=torch.long), 'run_id': run_id, 'window_id': str(r.window_id)}

def make_fold_loaders(windows, assignments, fold, run_to_class, mean, std):
    fold_a = assignments[assignments.fold == fold]
    loaders = {}
    for split in ['train', 'validation', 'test']:
        ids = set(fold_a.loc[fold_a.split == split, 'run_id'].astype(str))
        w = windows[windows.run_id.isin(ids)].copy().reset_index(drop=True)
        ds = MotorDataset(w, run_to_class, mean, std)
        loaders[split] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=split == 'train', num_workers=NUM_WORKERS, pin_memory=DEVICE.type == 'cuda', drop_last=False, worker_init_fn=seed_worker, generator=make_generator(SEED + fold))
        print(f'{split:12s}: {len(ids):3d} runs | {len(w):6d} windows')
    return loaders

class ConvBlock(nn.Module):

    def __init__(self, ci, co, k, stride):
        super().__init__()
        g = min(8, co)
        while co % g != 0:
            g -= 1
        self.net = nn.Sequential(nn.Conv1d(ci, co, k, stride=stride, padding=k // 2, bias=False), nn.GroupNorm(g, co), nn.GELU(), nn.MaxPool1d(2, 2))

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(ConvBlock(3, 32, 15, 4), ConvBlock(32, 64, 11, 2), ConvBlock(64, 128, 7, 2), ConvBlock(128, 192, 5, 2), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(192, ENCODER_DIM), nn.GELU(), nn.Dropout(DROPOUT))

    def forward(self, x):
        return self.net(x)

class MultimodalClassifier(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.vib = Encoder()
        self.cur = Encoder()
        self.head = nn.Sequential(nn.Linear(ENCODER_DIM * 2, FUSION_DIM), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(FUSION_DIM, n_classes))

    def forward(self, vibration, current):
        z = torch.cat([self.vib(vibration), self.cur(current)], dim=1)
        return self.head(z)

def metrics(y_true, y_pred):
    return {'accuracy': float(accuracy_score(y_true, y_pred)), 'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)), 'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 'n_samples': int(len(y_true))}

def aggregate_runs(probs, targets, run_ids):
    out_ids, out_probs, out_targets = ([], [], [])
    for rid in pd.unique(run_ids):
        m = run_ids == rid
        t = np.unique(targets[m])
        if len(t) != 1:
            raise RuntimeError(f'{rid}: inconsistent targets')
        out_ids.append(rid)
        out_probs.append(probs[m].mean(0))
        out_targets.append(int(t[0]))
    out_probs = np.asarray(out_probs)
    out_targets = np.asarray(out_targets, dtype=int)
    return {'run_ids': np.asarray(out_ids), 'probabilities': out_probs, 'targets': out_targets, 'predictions': out_probs.argmax(1)}

@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, targets, run_ids, window_ids = ([], [], [], [])
    for b in loader:
        v = b['vibration'].to(DEVICE, non_blocking=True)
        c = b['current'].to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            p = torch.softmax(model(v, c), dim=1)
        probs.append(p.cpu().numpy())
        targets.append(b['target'].numpy())
        run_ids.extend(list(b['run_id']))
        window_ids.extend(list(b['window_id']))
    probs = np.concatenate(probs)
    targets = np.concatenate(targets).astype(int)
    return {'probabilities': probs, 'targets': targets, 'predictions': probs.argmax(1), 'run_ids': np.asarray(run_ids), 'window_ids': np.asarray(window_ids)}

def per_class_df(y_true, y_pred, class_names):
    labels = np.arange(len(class_names))
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({'class_id': labels, 'fault_raw': class_names, 'precision': p, 'recall': r, 'f1': f1, 'support': sup.astype(int)})

def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    loss_sum, n = (0.0, 0)
    for b in loader:
        v = b['vibration'].to(DEVICE, non_blocking=True)
        c = b['current'].to(DEVICE, non_blocking=True)
        y = b['target'].to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = model(v, c)
            loss = criterion(logits, y)
        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        bs = len(y)
        loss_sum += float(loss.item()) * bs
        n += bs
    return loss_sum / n

@torch.no_grad()
def validate_epoch(model, loader, criterion):
    model.eval()
    loss_sum, n = (0.0, 0)
    probs, targets, run_ids = ([], [], [])
    for b in loader:
        v = b['vibration'].to(DEVICE, non_blocking=True)
        c = b['current'].to(DEVICE, non_blocking=True)
        y = b['target'].to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = model(v, c)
            loss = criterion(logits, y)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        targets.append(y.cpu().numpy())
        run_ids.extend(list(b['run_id']))
        bs = len(y)
        loss_sum += float(loss.item()) * bs
        n += bs
    probs = np.concatenate(probs)
    targets = np.concatenate(targets).astype(int)
    run_pred = aggregate_runs(probs, targets, np.asarray(run_ids))
    run_macro = f1_score(run_pred['targets'], run_pred['predictions'], average='macro', zero_division=0)
    return (loss_sum / n, float(run_macro))

def train_fold(fold, loaders, n_classes):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = MultimodalClassifier(n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-06)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    best_macro = -1.0
    best_loss = np.inf
    best_epoch = 0
    wait = 0
    hist = []
    ckpt = RESULT_DIR / f'best_model_fold_{fold}.pt'
    t0 = time.time()
    print('\n' + '=' * 100)
    print(f'FOLD {fold} TRAINING')
    print('=' * 100)
    for epoch in range(1, MAX_EPOCHS + 1):
        e0 = time.time()
        tr_loss = train_epoch(model, loaders['train'], criterion, optimizer, scaler)
        va_loss, va_macro = validate_epoch(model, loaders['validation'], criterion)
        scheduler.step(va_loss)
        lr = optimizer.param_groups[0]['lr']
        sec = time.time() - e0
        print(f'Fold {fold} | Epoch {epoch:02d}/{MAX_EPOCHS:02d} | train_loss={tr_loss:.5f} | val_loss={va_loss:.5f} | val_run_macro_f1={va_macro:.4f} | lr={lr:.2e} | {sec:.1f}s')
        hist.append({'fold': fold, 'epoch': epoch, 'train_loss': tr_loss, 'validation_loss': va_loss, 'validation_run_macro_f1': va_macro, 'learning_rate': lr, 'epoch_seconds': sec})
        improved = va_macro > best_macro + 1e-06 or (abs(va_macro - best_macro) <= 1e-06 and va_loss < best_loss - 1e-06)
        if improved:
            best_macro = va_macro
            best_loss = va_loss
            best_epoch = epoch
            wait = 0
            torch.save({'model_state_dict': model.state_dict(), 'fold': fold, 'epoch': epoch, 'validation_run_macro_f1': va_macro, 'validation_loss': va_loss, 'n_classes': n_classes}, ckpt)
            print('  -> Best model saved.')
        else:
            wait += 1
        if wait >= PATIENCE:
            print('  -> Early stopping.')
            break
    pd.DataFrame(hist).to_csv(RESULT_DIR / f'training_history_fold_{fold}.csv', index=False, encoding='utf-8-sig')
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state['model_state_dict'])
    minutes = (time.time() - t0) / 60.0
    print(f'Best epoch: {best_epoch}')
    print(f'Best val run macro-F1: {best_macro:.4f}')
    print(f'Fold training time: {minutes:.2f} min')
    return (model, best_epoch, best_macro, best_loss, minutes)

def save_fold_results(fold, pred, class_names):
    wm = metrics(pred['targets'], pred['predictions'])
    rp = aggregate_runs(pred['probabilities'], pred['targets'], pred['run_ids'])
    rm = metrics(rp['targets'], rp['predictions'])
    df = pd.DataFrame({'fold': fold, 'run_id': rp['run_ids'], 'true_class_id': rp['targets'], 'pred_class_id': rp['predictions']})
    df['true_fault_raw'] = [class_names[i] for i in df.true_class_id]
    df['pred_fault_raw'] = [class_names[i] for i in df.pred_class_id]
    for i, name in enumerate(class_names):
        safe = name.replace(' ', '_').replace('/', '_')
        df[f'prob_{i:02d}_{safe}'] = rp['probabilities'][:, i]
    df.to_csv(RESULT_DIR / f'run_predictions_fold_{fold}.csv', index=False, encoding='utf-8-sig')
    pc = per_class_df(rp['targets'], rp['predictions'], class_names)
    pc.insert(0, 'fold', fold)
    pc.to_csv(RESULT_DIR / f'per_class_fold_{fold}.csv', index=False, encoding='utf-8-sig')
    labels = np.arange(len(class_names))
    cm = confusion_matrix(rp['targets'], rp['predictions'], labels=labels)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(RESULT_DIR / f'confusion_matrix_fold_{fold}.csv', encoding='utf-8-sig')
    row_sums = cm.sum(1, keepdims=True)
    cmn = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    pd.DataFrame(cmn, index=class_names, columns=class_names).to_csv(RESULT_DIR / f'confusion_matrix_normalized_fold_{fold}.csv', encoding='utf-8-sig')
    return (wm, rm, df, pc)

def main():
    runs, windows, class_names = load_inventory()
    validate_cache(runs)
    run_stats = load_or_build_run_stats(runs)
    assignments = create_fold_assignments(runs)
    run_to_class = runs.set_index('run_id')['class_id'].to_dict()
    fold_rows, oof_parts, pc_parts = ([], [], [])
    all_start = time.time()
    for fold in range(1, N_FOLDS + 1):
        print('\n\n' + '#' * 100)
        print(f'# FOLD {fold}/{N_FOLDS}')
        print('#' * 100)
        fa = assignments[assignments.fold == fold]
        train_ids = fa.loc[fa.split == 'train', 'run_id'].astype(str).tolist()
        val_ids = fa.loc[fa.split == 'validation', 'run_id'].astype(str).tolist()
        test_ids = fa.loc[fa.split == 'test', 'run_id'].astype(str).tolist()
        mean, std = fold_normalization(train_ids, run_stats)
        print('Fold train-only mean:', mean)
        print('Fold train-only std :', std)
        with open(RESULT_DIR / f'normalization_fold_{fold}.json', 'w', encoding='utf-8') as f:
            json.dump({'fold': fold, 'train_run_count': len(train_ids), 'mean': mean.tolist(), 'std': std.tolist()}, f, indent=2, ensure_ascii=False)
        loaders = make_fold_loaders(windows, assignments, fold, run_to_class, mean, std)
        model, best_epoch, best_macro, best_loss, minutes = train_fold(fold, loaders, len(class_names))
        test_pred = predict(model, loaders['test'])
        wm, rm, rdf, pc = save_fold_results(fold, test_pred, class_names)
        oof_parts.append(rdf)
        pc_parts.append(pc)
        fold_rows.append({'fold': fold, 'train_runs': len(train_ids), 'validation_runs': len(val_ids), 'test_runs': len(test_ids), 'best_epoch': best_epoch, 'best_validation_run_macro_f1': best_macro, 'best_validation_loss': best_loss, 'window_accuracy': wm['accuracy'], 'window_balanced_accuracy': wm['balanced_accuracy'], 'window_macro_f1': wm['macro_f1'], 'window_weighted_f1': wm['weighted_f1'], 'run_accuracy': rm['accuracy'], 'run_balanced_accuracy': rm['balanced_accuracy'], 'run_macro_f1': rm['macro_f1'], 'run_weighted_f1': rm['weighted_f1'], 'fold_minutes': minutes})
        print('\nRUN LEVEL TEST')
        for k, v in rm.items():
            print(f'{k:24s}: {v:.4f}' if isinstance(v, float) else f'{k:24s}: {v}')
        del model, loaders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    fs = pd.DataFrame(fold_rows)
    fs.to_csv(RESULT_DIR / 'FOLD_SUMMARY.csv', index=False, encoding='utf-8-sig')
    oof = pd.concat(oof_parts, ignore_index=True)
    vc = oof.run_id.value_counts()
    if len(oof) != 288 or len(vc) != 288 or (not (vc == 1).all()):
        raise RuntimeError('OOF coverage is not exactly 288 x 1.')
    oof.to_csv(RESULT_DIR / 'OUT_OF_FOLD_RUN_PREDICTIONS.csv', index=False, encoding='utf-8-sig')
    yt = oof.true_class_id.to_numpy(int)
    yp = oof.pred_class_id.to_numpy(int)
    oof_metrics = metrics(yt, yp)
    labels = np.arange(len(class_names))
    cm = confusion_matrix(yt, yp, labels=labels)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(RESULT_DIR / 'OOF_CONFUSION_MATRIX.csv', encoding='utf-8-sig')
    rs = cm.sum(1, keepdims=True)
    cmn = np.divide(cm, rs, out=np.zeros_like(cm, dtype=float), where=rs != 0)
    pd.DataFrame(cmn, index=class_names, columns=class_names).to_csv(RESULT_DIR / 'OOF_CONFUSION_MATRIX_NORMALIZED.csv', encoding='utf-8-sig')
    oof_pc = per_class_df(yt, yp, class_names)
    oof_pc.to_csv(RESULT_DIR / 'OOF_PER_CLASS.csv', index=False, encoding='utf-8-sig')
    pc_all = pd.concat(pc_parts, ignore_index=True)
    rows = []
    for (cid, name), g in pc_all.groupby(['class_id', 'fault_raw']):
        rows.append({'class_id': cid, 'fault_raw': name, 'precision_mean': g.precision.mean(), 'precision_std': g.precision.std(ddof=1), 'recall_mean': g.recall.mean(), 'recall_std': g.recall.std(ddof=1), 'f1_mean': g.f1.mean(), 'f1_std': g.f1.std(ddof=1)})
    pd.DataFrame(rows).to_csv(RESULT_DIR / 'PER_CLASS_CV_SUMMARY.csv', index=False, encoding='utf-8-sig')
    cv = {}
    for m in ['run_accuracy', 'run_balanced_accuracy', 'run_macro_f1', 'run_weighted_f1']:
        cv[m + '_mean'] = float(fs[m].mean())
        cv[m + '_std'] = float(fs[m].std(ddof=1))
    total_minutes = (time.time() - all_start) / 60.0
    with open(RESULT_DIR / 'FINAL_SUMMARY.json', 'w', encoding='utf-8') as f:
        json.dump({'n_runs': len(runs), 'n_windows': len(windows), 'n_classes': len(class_names), 'n_folds': N_FOLDS, 'cv_run_metrics': cv, 'oof_run_metrics': oof_metrics, 'total_minutes': total_minutes}, f, indent=2, ensure_ascii=False)
    with open(RESULT_DIR / 'experiment_config.json', 'w', encoding='utf-8') as f:
        json.dump({'seed': SEED, 'protocol': 'run-level stratified 5-fold closed-set CV', 'label': 'original fault_raw', 'n_folds': N_FOLDS, 'validation_fraction': VAL_FRACTION, 'window_seconds': 2.0, 'batch_size': BATCH_SIZE, 'max_epochs': MAX_EPOCHS, 'patience': PATIENCE, 'learning_rate': LR, 'weight_decay': WEIGHT_DECAY, 'loss': 'CrossEntropyLoss', 'checkpoint': 'validation run-level macro-F1; tie-break val loss', 'class_names': class_names}, f, indent=2, ensure_ascii=False)
    print('\n' + '=' * 120)
    print('5-FOLD CLOSED-SET RESULTS')
    print('=' * 120)
    print(fs[['fold', 'run_accuracy', 'run_balanced_accuracy', 'run_macro_f1', 'run_weighted_f1', 'best_epoch', 'fold_minutes']].to_string(index=False))
    print('\nCV RUN-LEVEL MEAN ± STD')
    print(f"Accuracy          : {cv['run_accuracy_mean']:.4f} ± {cv['run_accuracy_std']:.4f}")
    print(f"Balanced accuracy : {cv['run_balanced_accuracy_mean']:.4f} ± {cv['run_balanced_accuracy_std']:.4f}")
    print(f"Macro-F1          : {cv['run_macro_f1_mean']:.4f} ± {cv['run_macro_f1_std']:.4f}")
    print(f"Weighted-F1       : {cv['run_weighted_f1_mean']:.4f} ± {cv['run_weighted_f1_std']:.4f}")
    print('\nOOF RUN-LEVEL')
    for k, v in oof_metrics.items():
        print(f'{k:24s}: {v:.4f}' if isinstance(v, float) else f'{k:24s}: {v}')
    print(f'\nTotal experiment time: {total_minutes:.2f} min')
    print(f'\nResults:\n{RESULT_DIR}')
    print('\nKey output files:')
    print('1. FOLD_SUMMARY.csv')
    print('2. OUT_OF_FOLD_RUN_PREDICTIONS.csv')
    print('3. OOF_CONFUSION_MATRIX.csv')
    print('4. OOF_CONFUSION_MATRIX_NORMALIZED.csv')
    print('5. OOF_PER_CLASS.csv')
    print('6. PER_CLASS_CV_SUMMARY.csv')
    print('7. FINAL_SUMMARY.json')
if __name__ == '__main__':
    main()
