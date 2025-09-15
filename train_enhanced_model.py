import os
import sys
import json
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt  # Add this import at the top of the file
from src.models.speech_model import EnhancedTwiSpeechModel, AdvancedTrainer
from src.features.augmented_dataset import AugmentedTwiDataset
from src.utils.training_pipeline import TrainingPipeline
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn.utils as nn_utils

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Helper to apply focal scaling (used inline to avoid reassigning trainer.criterion type)
def _apply_focal_loss(logits, targets, base_ce_per_sample, gamma: float):
    """
    logits: (B, C)
    targets: (B,)
    base_ce_per_sample: (B,) unreduced CE loss (no smoothing)
    """
    with torch.no_grad():
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        pt = probs[torch.arange(logits.size(0), device=logits.device), targets]
        focal_factor = (1 - pt).clamp(min=1e-6).pow(gamma)
    return focal_factor * base_ce_per_sample
# ---------------- Slot-aware utilities ----------------
def slot_collate(batch):
    """
    Collate function that supports datasets returning either:
      (feature_tensor, label_tensor) OR
      (feature_tensor, label_tensor, slots_dict)
    Handles variable temporal lengths by zero-padding each (C, T_i) tensor
    in the batch to the max T within that batch.

    Returns:
      features: (B, C, T_max)
      labels:   (B,)
      slots:    list[dict] length B (with '_orig_len' added per sample)
    """
    import torch

    features = []
    labels = []
    slots_list = []
    lengths = []

    # Separate samples
    for sample in batch:
        if len(sample) == 2:
            feat, lab = sample
            sl = {}
        elif len(sample) == 3:
            feat, lab, sl = sample
        else:
            raise ValueError(f"Unexpected sample structure length={len(sample)}")
        # Ensure tensor
        if not torch.is_tensor(feat):
            feat = torch.tensor(feat, dtype=torch.float32)
        # Enforce (C, T) channel-first; if (T, C) transpose
        if feat.ndim != 2:
            raise ValueError(f"Expected 2D feature tensor, got shape {feat.shape}")
        if feat.shape[0] < feat.shape[1] and feat.shape[0] not in (39, 117):  # heuristic safeguard
            pass  # leave as-is
        lengths.append(feat.shape[-1])
        features.append(feat)
        labels.append(lab)
        slots_list.append(sl)

    # Determine max length in batch
    max_len = max(lengths) if lengths else 0

    # Pad
    padded = []
    for feat, orig_len, sl in zip(features, lengths, slots_list):
        if feat.shape[-1] < max_len:
            pad_width = max_len - feat.shape[-1]
            pad_tensor = torch.zeros(feat.shape[0], pad_width, dtype=feat.dtype)
            feat = torch.cat([feat, pad_tensor], dim=-1)
        # Record original length for potential masking downstream
        sl['_orig_len'] = int(orig_len)
        padded.append(feat)

    feature_batch = torch.stack(padded, dim=0)  # (B, C, T_max)
    label_batch = torch.stack(labels, dim=0)

    return feature_batch, label_batch, slots_list


def train_epoch_with_slots(trainer, model, device, loader, epoch, accumulation_steps: int = 1):
    """
    Slot-aware training loop (replaces trainer.train_epoch) that:
      - Accepts batches of (inputs, targets, slots)
      - Preserves gradient accumulation & optional grad clipping
      - Records history metrics in trainer.history
    """
    import torch
    model.train()
    optimizer = trainer.optimizer
    criterion = trainer.criterion
    running_loss = 0.0
    correct = 0
    total = 0

    # Ensure history keys exist
    for k in ['train_loss', 'train_acc', 'learning_rates']:
        if k not in trainer.history:
            trainer.history[k] = []

    optimizer.zero_grad(set_to_none=True)
    for step, (inputs, targets, batch_slots) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        if trainer.config.get('use_focal_loss'):
            # Build per-sample CE (no smoothing) for focal scaling
            per_sample_ce = torch.nn.functional.cross_entropy(
                outputs,
                targets,
                weight=getattr(trainer.criterion, 'weight', None),
                reduction='none'
            )
            focal_gamma = trainer.config.get('focal_gamma', 2.0)
            focal_loss = _apply_focal_loss(outputs, targets, per_sample_ce, focal_gamma).mean()
            loss = focal_loss / accumulation_steps
        else:
            loss = criterion(outputs, targets) / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            # Optional gradient clipping
            clip_val = trainer.config.get('clip_grad_norm') if hasattr(trainer, 'config') else None
            if clip_val:
                nn_utils.clip_grad_norm_(model.parameters(), clip_val)
            optimizer.step()
            if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
                trainer.scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total if total > 0 else 0.0

    trainer.history['train_loss'].append(epoch_loss)
    trainer.history['train_acc'].append(epoch_acc)
    trainer.history['learning_rates'].append(optimizer.param_groups[0]['lr'])

    logging.info(
        f"[Epoch {epoch+1}] Slot-aware train -> Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%"
    )

    return epoch_loss, epoch_acc

def train_enhanced_model(data_dir="data/processed", model_dir="data/models/enhanced", epochs=100):
    """Train an enhanced model with all the improvements"""
    # 1. Configure the model with optimized parameters
    config = {
        'data_dir': data_dir,
        'model_dir': model_dir,
        'batch_size': 64,
        'learning_rate': 0.002,
        'num_epochs': epochs,
        'early_stopping_patience': 15,
        'hidden_dim': 128,
        'dropout': 0.3,
        'num_heads': 8,
        'weight_decay': 0.01,
        'clip_grad_norm': 1.0,
        'accumulation_steps': 2,
        'random_seed': 42,
        'use_focal_loss': False,  # (H) can toggle True later when baseline learning established
        'focal_gamma': 2.0,
        'use_weighted_sampler': True,
        'track_macro_f1': True,
        'early_stopping_metric': 'val_loss',  # switched to validation loss for stability (A/B)
        'early_stopping_mode': 'min',         # minimize validation loss (A/B)
        'ema_decay': 0.995,                 # Exponential Moving Average for model weights
        'save_split_indices': True,
        'minority_class_threshold': 5,
        'adaptive_label_smoothing': True,
        'label_smoothing_max': 0.15,
        'label_smoothing_min': 0.0,
        'label_smoothing_warmup_epochs': 10
    }

    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Set random seed for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['random_seed'])
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Deterministic algorithms (fallback if not supported)
    except Exception:
        pass



    # 2. Load data using existing pipeline
    pipeline = TrainingPipeline(config)
    features, labels, label_map, slots = pipeline.load_data()

    # 2b. (Optional) Enrich features with delta & delta-delta (stack along channel axis)
    try:
        import librosa
        enriched = []
        for f in features:
            # Expect shape (C, T)
            base = f
            delta = librosa.feature.delta(base)
            delta2 = librosa.feature.delta(base, order=2)
            stacked = np.concatenate([base, delta, delta2], axis=0)
            enriched.append(stacked)
        original_dim = features[0].shape[0]
        features = enriched
        logger.info(f"Delta enrichment applied: {original_dim} -> {features[0].shape[0]} feature dims")
    except Exception as e:
        logger.warning(f"Delta feature enrichment skipped due to error: {e}")

    # 3. Create dataset with augmentation (including optional SpecAugment)
    augmented_dataset = AugmentedTwiDataset(
        features=features,  # (H) baseline simplification keeps enrichment but you can disable by skipping delta step above
        labels=labels,
        label_to_idx=label_map,
        augment=False,  # (H) disable feature-domain augmentation for baseline stabilization
        spec_augment=False,  # (H) disable SpecAugment initially
        spec_time_mask_max=12,
        spec_freq_mask_max=8,
        spec_num_time_masks=2,
        spec_num_freq_masks=2,
        spec_prob=0.6
    )  # (H) label smoothing retained; can set to 0 in config if still unstable

    # 4. Get class weights for balancing
    class_weights = augmented_dataset.get_class_weights()

    # 5. Create train/val/test split
    indices = np.arange(len(augmented_dataset))
    label_indices = [augmented_dataset.label_to_idx[label] for label in labels]

    # Presence-aware split ensuring each class (where possible) appears in validation and test (A)
    # Strategy:
    # 1. Group indices by class.
    # 2. For each class:
    #    - If >=3 samples: allocate 1 to val, 1 to test, rest go to a remainder pool.
    #    - If 2 samples: allocate 1 to val, 1 to test.
    #    - If 1 sample: allocate 1 to val (test will not have that class).
    # 3. Split the remainder pool into train/val/test proportions (80/10/10) without removing the guaranteed allocations.
    # 4. Combine guaranteed + supplemental splits; ensure uniqueness.
    rng = np.random.default_rng(42)
    from collections import defaultdict
    per_class = defaultdict(list)
    for idx in indices:
        per_class[label_indices[idx]].append(idx)
    for cls_id in per_class:
        rng.shuffle(per_class[cls_id])

    guaranteed_val = set()
    guaranteed_test = set()
    used = set()

    for cls_id, cls_list in per_class.items():
        n = len(cls_list)
        if n >= 3:
            guaranteed_val.add(cls_list[0])
            guaranteed_test.add(cls_list[1])
            used.update(cls_list[0:2])
        elif n == 2:
            guaranteed_val.add(cls_list[0])
            guaranteed_test.add(cls_list[1])
            used.update(cls_list)
        elif n == 1:
            guaranteed_val.add(cls_list[0])
            used.add(cls_list[0])

    # Remainder pool
    remainder = [i for i in indices if i not in used]
    rng.shuffle(remainder)

    if len(remainder) > 0:
        rem_train_cut = int(len(remainder) * 0.8)
        rem_val_cut = int(len(remainder) * 0.9)
        rem_train = remainder[:rem_train_cut]
        rem_val_extra = remainder[rem_train_cut:rem_val_cut]
        rem_test_extra = remainder[rem_val_cut:]
    else:
        rem_train = rem_val_extra = rem_test_extra = []

    # Assemble splits
    train_indices = rem_train  # training gets only remainder initially; (minority upsampling later will duplicate as needed)
    val_indices = list(guaranteed_val) + rem_val_extra
    test_indices = list(guaranteed_test) + rem_test_extra

    # Deduplicate (just in case of overlap) while preserving order roughly
    def _dedup(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    train_indices = _dedup(train_indices)
    val_indices = _dedup(val_indices)
    test_indices = _dedup(test_indices)

    logger.info(f"Presence-aware split created: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    # Minority upsampling before creating Subset datasets (D)
    minority_min_count = config.get('minority_upsample_min_count', 6)
    if minority_min_count > 0:
        from collections import Counter
        cls_counts = Counter([label_indices[i] for i in train_indices])
        extra_indices = []
        for cls_id, cnt in cls_counts.items():
            if cnt < minority_min_count:
                needed = minority_min_count - cnt
                # Collect existing indices for this class
                existing = [i for i in train_indices if label_indices[i] == cls_id]
                # Duplicate (round-robin) to reach target
                while needed > 0:
                    for ei in existing:
                        extra_indices.append(ei)
                        needed -= 1
                        if needed <= 0:
                            break
        if extra_indices:
            import numpy as _np
            train_indices = _np.concatenate([train_indices, _np.array(extra_indices)])
            logger.info(f"Minority upsampling added {len(extra_indices)} samples "
                        f"(new train size: {len(train_indices)})")

    # Create subset datasets (after upsampling)
    # Ensure indices are plain Python lists of ints for Subset (avoids type issues with numpy arrays)
    if not isinstance(train_indices, list):
        train_indices = [int(i) for i in train_indices]
    if not isinstance(val_indices, list):
        val_indices = [int(i) for i in val_indices]
    if not isinstance(test_indices, list):
        test_indices = [int(i) for i in test_indices]

    train_dataset = Subset(augmented_dataset, train_indices)
    val_dataset = Subset(augmented_dataset, val_indices)
    test_dataset = Subset(augmented_dataset, test_indices)

    # Log absent classes in each split (C)
    all_classes = set(range(len(label_map)))
    train_cls = {label_indices[i] for i in train_indices}
    val_cls = {label_indices[i] for i in val_indices}
    test_cls = {label_indices[i] for i in test_indices}
    missing_train = all_classes - train_cls
    missing_val = all_classes - val_cls
    missing_test = all_classes - test_cls
    if missing_train:
        logger.warning(f"Classes absent in TRAIN split: {sorted(list(missing_train))}")
    if missing_val:
        logger.warning(f"Classes absent in VAL split: {sorted(list(missing_val))}")
    if missing_test:
        logger.warning(f"Classes absent in TEST split: {sorted(list(missing_test))}")

    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Validation set: {len(val_dataset)} samples")
    logger.info(f"Test set: {len(test_dataset)} samples")
    # Save reproducible split indices
    if config.get('save_split_indices', True):
        import json as _json
        split_path = os.path.join(model_dir, "splits.json")
        with open(split_path, "w") as _f:
            _json.dump({
                "train_indices": [int(x) for x in (train_indices.tolist() if isinstance(train_indices, np.ndarray) else list(train_indices))],
                "val_indices": [int(x) for x in (val_indices.tolist() if isinstance(val_indices, np.ndarray) else list(val_indices))],
                "test_indices": [int(x) for x in (test_indices.tolist() if isinstance(test_indices, np.ndarray) else list(test_indices))]
            }, _f, indent=2)
        logger.info(f"Saved split indices to {split_path}")

    # 6. Create data loaders (weighted sampler for imbalance if enabled)
    if config.get('use_weighted_sampler', True):
        sample_class_weights = class_weights.numpy() if hasattr(class_weights, 'numpy') else class_weights
        sample_weights = [float(sample_class_weights[label_indices[i]]) for i in train_indices]
        weighted_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=weighted_sampler,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=slot_collate
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=slot_collate
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=slot_collate
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=slot_collate
    )

    # Update steps per epoch for scheduler
    config['steps_per_epoch'] = len(train_loader)

    # 7. Initialize the enhanced model
    if len(features) > 0:
        input_dim = features[0].shape[0]
        logger.info(f"Using input dimension: {input_dim}")
    else:
        input_dim = 94  # Default
        logger.warning(f"No features found, using default input_dim={input_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = EnhancedTwiSpeechModel(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_classes=len(label_map),
        dropout=config['dropout'],
        num_heads=config['num_heads']
    )

    # 8. Initialize the trainer with advanced features
    trainer = AdvancedTrainer(model, device, config)

    # Set criterion with class weights (focal scaling applied later if enabled)
    if class_weights is not None and not config.get('use_focal_loss'):
        trainer.criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=0.1
        )

    # 9. Training loop with early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    # Generic early stopping metric tracking
    mode = config.get('early_stopping_mode', 'min')
    monitor_metric = config.get('early_stopping_metric', 'val_loss')
    if mode == 'max':
        best_metric = -float('inf')
        compare = lambda current, best: current > best
    else:
        best_metric = float('inf')
        compare = lambda current, best: current < best
    ema_decay = config.get('ema_decay', None)
    ema_state = None
    if ema_decay:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(config['num_epochs']):
        # Train one epoch (slot-aware)
        train_loss, train_acc = train_epoch_with_slots(
            trainer,
            model,
            device,
            train_loader,
            epoch,
            config.get('accumulation_steps', 1)
        )

        # Evaluate on validation set
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_targets_all, val_preds_all = [], []

        with torch.no_grad():
            for inputs, targets, _slots in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                if config.get('use_focal_loss'):
                    per_sample_ce = torch.nn.functional.cross_entropy(
                        outputs,
                        targets,
                        weight=getattr(trainer.criterion, 'weight', None),
                        reduction='none'
                    )
                    focal_loss = _apply_focal_loss(
                        outputs, targets, per_sample_ce, config.get('focal_gamma', 2.0)
                    ).mean()
                    loss = focal_loss
                else:
                    loss = trainer.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                val_targets_all.extend(targets.cpu().numpy())
                val_preds_all.extend(predicted.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        if config.get('track_macro_f1', True):
            try:
                from sklearn.metrics import f1_score
                val_f1 = f1_score(val_targets_all, val_preds_all, average='macro')
            except Exception as e:
                logger.warning(f"Macro-F1 computation failed: {e}")
                val_f1 = 0.0
        else:
            val_f1 = 0.0

        # Update trainer history
        trainer.history['val_loss'].append(val_loss)
        trainer.history['val_acc'].append(val_acc)
        if 'val_f1' not in trainer.history:
            trainer.history['val_f1'] = []
        trainer.history['val_f1'].append(val_f1)

        # Print progress
        # (J) Track prediction distribution on validation set
        val_pred_counts = {}
        for p in val_preds_all:
            val_pred_counts[p] = val_pred_counts.get(p, 0) + 1
        top_val_preds = sorted(val_pred_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(
            f"Epoch {epoch+1}/{config['num_epochs']} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
            f"Val Macro-F1: {val_f1:.4f}, "
            f"LR: {trainer.optimizer.param_groups[0]['lr']:.6f} | "
            f"Top Val Preds (id:count): {top_val_preds}"
        )

        # Determine current monitored metric value
        current_metric = val_f1 if monitor_metric == 'val_f1' else val_loss
        # Apply adaptive label smoothing schedule (for next epoch) if enabled and not using focal
        if config.get('adaptive_label_smoothing', False) and not config.get('use_focal_loss'):
            max_ls = config.get('label_smoothing_max', 0.15)
            min_ls = config.get('label_smoothing_min', 0.0)
            warmup = config.get('label_smoothing_warmup_epochs', 10)
            if epoch < warmup:
                # Cosine decay from max to min across warmup
                import math
                progress = epoch / max(1, warmup)
                current_ls = min_ls + 0.5*(max_ls - min_ls)*(1 + math.cos(progress * math.pi))
            else:
                current_ls = min_ls
            # Recreate criterion with new smoothing (keeping weights)
            if hasattr(trainer.criterion, 'weight'):
                weight_tensor = trainer.criterion.weight
            else:
                weight_tensor = None
            trainer.criterion = torch.nn.CrossEntropyLoss(
                weight=weight_tensor,
                label_smoothing=current_ls
            )
            if epoch == 0 or (epoch+1) % 5 == 0:
                logger.info(f"Adaptive label smoothing set to {current_ls:.4f}")
        # Update EMA after epoch (weights already stepped inside training)
        if ema_decay and ema_state is not None:
            with torch.no_grad():
                current_sd = model.state_dict()
                for k, v in current_sd.items():
                    # Skip non-floating point tensors (e.g., BatchNorm counters) to avoid dtype cast errors
                    if not torch.is_floating_point(v):
                        # Just mirror the current value to keep structure consistent
                        ema_state[k] = v.detach().clone()
                        continue
                    # If dtype changed (unlikely), reinitialize EMA entry
                    if ema_state[k].dtype != v.dtype:
                        ema_state[k] = v.detach().clone()
                        continue
                    ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)
        # Check for improvement on chosen metric
        if compare(current_metric, best_metric):
            best_metric = current_metric
            best_val_loss = min(best_val_loss, val_loss)
            patience_counter = 0
            best_model_state = model.state_dict().copy()

            # Save the best model
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': config
                },
                os.path.join(model_dir, "best_model.pt")
            )

            logger.info(f"Improved {monitor_metric} to {current_metric:.6f}. Best model saved!")
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation. Patience: {patience_counter}/{config['early_stopping_patience']}")

            # Early stopping
            if patience_counter >= config['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0 or epoch == config['num_epochs'] - 1:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict() if hasattr(trainer, 'scheduler') else None,
                    'history': trainer.history,
                    'config': config
                },
                os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt")
            )

    # 10. Load the best model for evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)

    # 11. Final evaluation on test set
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets, _slots in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if config.get('use_focal_loss'):
                per_sample_ce = torch.nn.functional.cross_entropy(
                    outputs,
                    targets,
                    weight=getattr(trainer.criterion, 'weight', None),
                    reduction='none'
                )
                loss = _apply_focal_loss(
                    outputs, targets, per_sample_ce, config.get('focal_gamma', 2.0)
                ).mean()
            else:
                loss = trainer.criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_loss = test_loss / len(test_loader)
    # (J) Test prediction distribution
    test_pred_counts = {}
    for p in all_preds:
        test_pred_counts[p] = test_pred_counts.get(p, 0) + 1
    top_test_preds = sorted(test_pred_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    test_acc = 100.0 * test_correct / test_total
    # Compute test macro F1
    try:
        from sklearn.metrics import f1_score
        test_f1 = f1_score(all_targets, all_preds, average='macro')
    except Exception:
        test_f1 = None

    logger.info(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%"
                f"{' Test Macro-F1: '+format(test_f1, '.4f') if test_f1 is not None else ''} | "
                f"Top Test Preds (id:count): {top_test_preds}")

    # 12. Generate confusion matrix and classification report
    try:
        from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
        import warnings as _sk_warnings
        # Suppress repetitive sklearn UserWarnings about high class / sample ratio
        _sk_warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="sklearn.metrics._classification"
        )
        # import matplotlib.pyplot as plt

        # Get class names
        idx_to_label = {v: k for k, v in label_map.items()}
        class_names = [idx_to_label.get(i, f"Unknown-{i}") for i in range(len(label_map))]

        # Compute confusion matrix (force inclusion of all classes to match display labels)
        all_label_indices = list(range(len(label_map)))
        cm = confusion_matrix(all_targets, all_preds, labels=all_label_indices)

        # Plot and save confusion matrix
        plt.figure(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        # Use numeric rotation (int) instead of string to satisfy Matplotlib
        disp.plot(cmap='Blues', xticks_rotation='vertical')
        plt.title('Confusion Matrix (All Classes)')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))

        # Generate and save classification report
        # Full label set to keep alignment even if some classes absent (A/C)
        labels_full = list(range(len(label_map)))
        report = classification_report(
            all_targets,
            all_preds,
            labels=labels_full,
            target_names=class_names,
            output_dict=True,
            zero_division='warn'
        )
        logger.info(f"Classification Report:\n{report}")

        with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
            from pprint import pformat as _pformat
            f.write(_pformat(report))
        # Per-class metrics JSON generation removed (previous block caused static analysis issues).
        # If reintroducing later, ensure robust type checks on classification_report output.

    except ImportError:
        logger.warning("scikit-learn not installed. Skipping confusion matrix and classification report.")

    # 13. Save training history plot & logits histogram (J)
    plt.figure(figsize=(15, 12))

    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(trainer.history['train_loss'], label='Train')
    plt.plot(trainer.history['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot accuracies
    plt.subplot(2, 2, 2)
    plt.plot(trainer.history['train_acc'], label='Train')
    plt.plot(trainer.history['val_acc'], label='Validation')
    if 'val_f1' in trainer.history:
        plt.plot(trainer.history['val_f1'], label='Val Macro-F1')
    plt.title('Accuracy / F1 Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(trainer.history['learning_rates'])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(alpha=0.3)

    # Add summary text
    plt.subplot(2, 2, 4)
    plt.axis('off')

    # Format summary text
    summary_text = (
        f"Training Summary:\n\n"
        f"Total Epochs: {len(trainer.history['train_loss'])}\n\n"
        f"Final Metrics:\n"
        f"  Train Loss: {trainer.history['train_loss'][-1]:.4f}\n"
        f"  Validation Loss: {trainer.history['val_loss'][-1]:.4f}\n"
        f"  Train Accuracy: {trainer.history['train_acc'][-1]:.2f}%\n"
        f"  Validation Accuracy: {trainer.history['val_acc'][-1]:.2f}%\n"
        f"  Validation Macro-F1: {trainer.history.get('val_f1', ['N/A'])[-1] if 'val_f1' in trainer.history else 'N/A'}\n"
        f"  Test Accuracy: {test_acc:.2f}%\n"
        f"  Test Macro-F1: {('N/A' if test_f1 is None else f'{test_f1:.4f}')}\n\n"
        f"Best Validation:\n"
        f"  Loss: {best_val_loss:.4f}\n\n"
        f"Model Information:\n"
        f"  Input Dimension: {input_dim}\n"
        f"  Hidden Dimension: {config['hidden_dim']}\n"
        f"  Number of Classes: {len(label_map)}"
    )

    plt.text(0.05, 0.95, summary_text, va='top')

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    plt.close()
    # (J) Logits histogram (sampled)
    try:
        import torch.nn.functional as F_hist
        model.eval()
        sample_logits = []
        with torch.no_grad():
            for i, (inp, tgt, _sl) in enumerate(test_loader):
                if i >= 3:  # limit batches
                    break
                logits = model(inp.to(device))
                sample_logits.append(logits.cpu())
        if sample_logits:
            all_logits = torch.cat(sample_logits, dim=0)
            probs = F_hist.softmax(all_logits, dim=1)
            avg_confidence = probs.max(dim=1).values.mean().item()
            plt.figure(figsize=(10, 4))
            plt.hist(probs.max(dim=1).values.numpy(), bins=20, alpha=0.7)
            plt.title(f'Max Probability Distribution (avg={avg_confidence:.3f})')
            plt.xlabel('Max Class Probability')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, 'max_prob_hist.png'))
            plt.close()
    except Exception as _e_hist:
        logger.warning(f"Failed to create logits histogram: {_e_hist}")

    # 14. Save model information
    model_info = {
        'input_dim': input_dim,
        'hidden_dim': config['hidden_dim'],
        'num_classes': len(label_map),
        'model_type': 'EnhancedTwiSpeechModel',
        'final_metrics': {
            'train_loss': float(trainer.history['train_loss'][-1]),
            'val_loss': float(trainer.history['val_loss'][-1]),
            'train_acc': float(trainer.history['train_acc'][-1]),
            'val_acc': float(trainer.history['val_acc'][-1]),
            'val_macro_f1': float(trainer.history['val_f1'][-1]) if 'val_f1' in trainer.history else None,
            'test_acc': float(test_acc),
            'test_macro_f1': float(test_f1) if test_f1 is not None else None
        },
        'best_val_loss': float(best_val_loss),
        'training_config': config,
        'num_parameters': sum(p.numel() for p in model.parameters())
    }

    with open(os.path.join(model_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=2)

    logger.info(f"Training complete! Model and statistics saved to {model_dir}")

    return model, trainer, test_acc

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train enhanced Twi speech recognition model")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory with processed data")
    parser.add_argument("--model-dir", type=str, default="data/models/enhanced", help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs to train")

    args = parser.parse_args()

    train_enhanced_model(args.data_dir, args.model_dir, args.epochs)
