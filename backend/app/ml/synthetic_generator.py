"""
Recovera AI 2.0: Longitudinal Clinical Cohort Data Generator
Generates realistic post-discharge recovery sequences following clinical distributions
with explicit missingness masks, irregular intervals, and patient-separated splits.
"""
import numpy as np
from typing import Dict, Any, Tuple, List

def generate_longitudinal_cohort(
    num_patients: int = 500,
    seq_length: int = 14,
    random_seed: int = 42
) -> Dict[str, Any]:
    np.random.seed(random_seed)
    
    patient_sequences = []
    patient_ids = [f"PT_{i:04d}" for i in range(num_patients)]
    
    for pid in patient_ids:
        # Assign clinical cohort archetype
        cohort_type = np.random.choice(["normal", "delayed_spike", "erratic"], p=[0.65, 0.22, 0.13])
        
        # Initial pain baseline
        init_pain = np.random.uniform(5.5, 8.5)
        pain_seq = []
        fever_seq = []
        sleep_seq = []
        adherence_seq = []
        mask_seq = [] # 1 = submitted, 0 = missing
        
        curr_pain = init_pain
        for day in range(seq_length):
            # Missingness probability (higher in erratic cohort)
            missing_prob = 0.35 if cohort_type == "erratic" else (0.12 if day > 7 else 0.05)
            is_missing = np.random.random() < missing_prob
            
            # Trajectory dynamics
            if cohort_type == "normal":
                decay = np.random.uniform(0.35, 0.65)
                noise = np.random.normal(0, 0.25)
                curr_pain = max(0.5, curr_pain - decay + noise)
                has_fever = 0.0
                adherence = np.random.uniform(85.0, 100.0)
            elif cohort_type == "delayed_spike":
                if 4 <= day <= 7:
                    # Acute complication or infection spike
                    curr_pain = min(9.5, curr_pain + np.random.uniform(0.6, 1.4))
                    has_fever = 1.0 if np.random.random() < 0.65 else 0.0
                    adherence = np.random.uniform(40.0, 75.0)
                else:
                    curr_pain = max(1.0, curr_pain - np.random.uniform(0.2, 0.5))
                    has_fever = 0.0
                    adherence = np.random.uniform(70.0, 95.0)
            else: # erratic
                curr_pain = np.clip(curr_pain + np.random.normal(0, 0.8), 1.0, 9.0)
                has_fever = 1.0 if np.random.random() < 0.15 else 0.0
                adherence = np.random.uniform(30.0, 90.0)

            sleep_hours = np.clip(7.5 - (curr_pain * 0.35) + np.random.normal(0, 0.5), 3.0, 9.5)
            
            if is_missing:
                # Missing check-in: observation is not recorded, mask is 0
                mask_seq.append(0.0)
                pain_seq.append(np.nan) # explicit missing
                fever_seq.append(0.0)
                sleep_seq.append(np.nan)
                adherence_seq.append(adherence)
            else:
                mask_seq.append(1.0)
                pain_seq.append(round(curr_pain, 1))
                fever_seq.append(has_fever)
                sleep_seq.append(round(sleep_hours, 1))
                adherence_seq.append(round(adherence, 1))

        patient_sequences.append({
            "patient_id": pid,
            "cohort_type": cohort_type,
            "pain": np.array(pain_seq),
            "fever": np.array(fever_seq),
            "sleep": np.array(sleep_seq),
            "adherence": np.array(adherence_seq),
            "mask": np.array(mask_seq),
        })

    return {
        "num_patients": num_patients,
        "seq_length": seq_length,
        "patients": patient_sequences
    }


def extract_sliding_windows(
    cohort: Dict[str, Any],
    window_size: int = 5
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Constructs (X, y, patient_ids) using fixed-size observation windows.
    Target: next-day pain score.
    Input features per time-step (4 features):
    [observed_pain_normalized, fever, sleep_normalized, adherence_normalized]
    plus explicit missingness mask.
    """
    X_list = []
    y_list = []
    pt_id_list = []

    for pt in cohort["patients"]:
        p_seq = pt["pain"]
        m_seq = pt["mask"]
        f_seq = pt["fever"]
        s_seq = pt["sleep"]
        a_seq = pt["adherence"]
        
        # Forward-fill missing values only for feature representation, with explicit mask feature
        filled_pain = np.copy(p_seq)
        filled_sleep = np.copy(s_seq)
        
        last_val = 6.0
        for i in range(len(filled_pain)):
            if np.isnan(filled_pain[i]):
                filled_pain[i] = last_val
            else:
                last_val = filled_pain[i]

        last_sleep = 7.0
        for i in range(len(filled_sleep)):
            if np.isnan(filled_sleep[i]):
                filled_sleep[i] = last_sleep
            else:
                last_sleep = filled_sleep[i]

        for t in range(window_size, len(p_seq)):
            # Target is actual next-day pain
            target = p_seq[t]
            if np.isnan(target):
                continue # Cannot evaluate on missing ground truth
            
            # Features over window [t - window_size, t]
            window_pain = filled_pain[t - window_size:t] / 10.0
            window_fever = f_seq[t - window_size:t]
            window_sleep = filled_sleep[t - window_size:t] / 10.0
            window_adh = a_seq[t - window_size:t] / 100.0
            window_mask = m_seq[t - window_size:t]
            
            # Shape: (window_size, 5)
            feat_matrix = np.column_stack([window_pain, window_fever, window_sleep, window_adh, window_mask])
            
            X_list.append(feat_matrix)
            y_list.append(target)
            pt_id_list.append(pt["patient_id"])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), pt_id_list


def split_by_patient(
    X: np.ndarray,
    y: np.ndarray,
    pt_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    STRICT PATIENT-LEVEL SPLIT: Prevents data leakage between train, validation, and test.
    """
    unique_patients = np.array(sorted(list(set(pt_ids))))
    np.random.seed(random_seed)
    np.random.shuffle(unique_patients)
    
    n_total = len(unique_patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_pts = set(unique_patients[:n_train])
    val_pts = set(unique_patients[n_train:n_train + n_val])
    test_pts = set(unique_patients[n_train + n_val:])
    
    pt_arr = np.array(pt_ids)
    train_idx = np.isin(pt_arr, list(train_pts))
    val_idx = np.isin(pt_arr, list(val_pts))
    test_idx = np.isin(pt_arr, list(test_pts))
    
    return {
        "train": (X[train_idx], y[train_idx]),
        "val": (X[val_idx], y[val_idx]),
        "test": (X[test_idx], y[test_idx]),
        "splits_info": {
            "num_train_patients": len(train_pts),
            "num_val_patients": len(val_pts),
            "num_test_patients": len(test_pts),
            "train_samples": int(np.sum(train_idx)),
            "val_samples": int(np.sum(val_idx)),
            "test_samples": int(np.sum(test_idx)),
        }
    }
