



# Streamlined attack simulation for PQC-protected biometric templates.
# Includes: Template inversion, correlation attacks, brute force, hill climbing.


import matplotlib
matplotlib.use('Agg')

import numpy as np
import random
import sys
sys.path.append('.')  


from main import Config, DataPipeline, FeatureExtractor, CancelableBiometricTemplate
# Imports classical protection system
from main_classical import generate_biohash_template
# Import classical verification function
try:
    from main_classical import verify_biohash_template
except ImportError:
    def verify_biohash_template(query, stored, threshold=0.85):
        # Computes cosine similarity between query and stored template
        query = query.astype(np.float32)
        stored = stored.astype(np.float32)
        norm_q = np.linalg.norm(query)
        norm_s = np.linalg.norm(stored)
        if norm_q > 1e-6 and norm_s > 1e-6:
            cosine_sim = float(np.dot(query, stored) / (norm_q * norm_s))
        else:
            cosine_sim = 0.0
        match = cosine_sim >= threshold
        return match, cosine_sim

# Loads real biometric data and generate PQC-protected (cancelable) templates
config = Config()
data_pipeline = DataPipeline(config)
feature_extractor = FeatureExtractor(config)

cancelable_system = CancelableBiometricTemplate(config)


# Loads fingerprint and face data
fingerprint_data = data_pipeline.load_fingerprint_data()
face_data = data_pipeline.load_face_data()


def get_templates_for_attack():
    # This function prepares a list of templates for attack simulation.
    # Both PQC-protected and classical-protected templates are included for fingerprint and face.
    templates = []
    # PQC-protected fingerprint
    if fingerprint_data:
        fp_sample = fingerprint_data[0]
        fp_processed = data_pipeline.preprocess_image(fp_sample['image'])
        fp_features = feature_extractor.extract_fingerprint_features(fp_processed)
        fp_cancelable, _ = cancelable_system.generate_cancelable_template(fp_features, 'user_fp_001', 'app_fp')
        templates.append(('pqc_fingerprint', fp_cancelable))
        # Classical protected fingerprint
        fp_classical = generate_biohash_template(fp_features, 'user_fp_001', 'app_fp')
        templates.append(('classical_fingerprint', fp_classical))
    # PQC-protected face
    if face_data:
        face_sample = face_data[0]
        face_processed = data_pipeline.preprocess_image(face_sample['image'])
        face_features = feature_extractor.extract_face_features(face_processed)
        face_cancelable, _ = cancelable_system.generate_cancelable_template(face_features, 'user_face_001', 'app_face')
        templates.append(('pqc_face', face_cancelable))
        # Classical protected face
        face_classical = generate_biohash_template(face_features, 'user_face_001', 'app_face')
        templates.append(('classical_face', face_classical))
    return templates

cancelable_templates = get_templates_for_attack()

# 1. Template Inversion Testing
def template_inversion(template):
    # Simulates template inversion by random guessing, but tests via cancelable verification
    guess = np.random.normal(0, 1, size=template.shape)
    user_id = 'attacker'
    app_id = 'attack_app'
    # Suppress internal logs for cancelable system
    import contextlib
    import io
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        cancelable_guess, params = cancelable_system.generate_cancelable_template(guess, user_id, app_id)
        match, score = cancelable_system.verify_cancelable_template(guess, template, params)
    print(f"[RESULT] Template Inversion: {'SUCCESS' if match else 'FAIL'} | Score: {score:.3f}")
    return score

# Classical attack: inversion for biohash
def classical_template_inversion(template):
    # Simulates inversion by random guessing, tests via classical verification
    guess = np.random.normal(0, 1, size=template.shape).astype(np.float32)
    norm = np.linalg.norm(guess)
    if norm > 1e-6:
        guess = guess / norm
    match, score = verify_biohash_template(guess, template)
    print(f"[RESULT] Classical Template Inversion: {'SUCCESS' if match else 'FAIL'} | Score: {score:.3f}")
    return score

# 2. Correlation Attacks
def correlation_attack(template):
    # Computes cross-application correlation for unlinkability
    user_id = 'user_test'
    app_id1 = 'app1'
    app_id2 = 'app2'
    import contextlib
    import io
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        template1, _ = cancelable_system.generate_cancelable_template(template, user_id, app_id1)
        template2, _ = cancelable_system.generate_cancelable_template(template, user_id, app_id2)
    norm1 = np.linalg.norm(template1)
    norm2 = np.linalg.norm(template2)
    if norm1 > 0 and norm2 > 0:
        cosine_sim = np.dot(template1, template2) / (norm1 * norm2)
    else:
        cosine_sim = 0.0
    print(f"[RESULT] Cross-App Correlation (Unlinkability): {cosine_sim:.3f}")
    return cosine_sim

# Classical attack: correlation for biohash
def classical_correlation_attack(template):
    # Simulates cross-app correlation by generating two biohashes for same user, different apps
    user_id = 'user_test'
    app_id1 = 'app1'
    app_id2 = 'app2'
    template1 = generate_biohash_template(template, user_id, app_id1)
    template2 = generate_biohash_template(template, user_id, app_id2)
    # Cosine similarity
    norm1 = np.linalg.norm(template1)
    norm2 = np.linalg.norm(template2)
    if norm1 > 0 and norm2 > 0:
        cosine_sim = np.dot(template1, template2) / (norm1 * norm2)
    else:
        cosine_sim = 0.0
    print(f"[RESULT] Classical Cross-App Correlation (Unlinkability): {cosine_sim:.3f}")
    return cosine_sim

# 3. Brute Force Analysis
def brute_force_attack(template, max_attempts=1000):
    # Brute force attack using cancelable verification (black-box)
    matches = 0
    import contextlib
    import io
    f = io.StringIO()
    for _ in range(max_attempts):
        guess = np.random.normal(0, 1, size=template.shape)
        user_id = 'attacker'
        app_id = 'attack_app'
        with contextlib.redirect_stdout(f):
            cancelable_guess, params = cancelable_system.generate_cancelable_template(guess, user_id, app_id)
            match, _ = cancelable_system.verify_cancelable_template(guess, template, params)
        if match:
            matches += 1
    print(f"[RESULT] Brute Force: Matches in {max_attempts} attempts: {matches}")
    return matches

# Classical attack: brute force for biohash
def classical_brute_force_attack(template, max_attempts=1000):
    matches = 0
    for _ in range(max_attempts):
        guess = np.random.normal(0, 1, size=template.shape).astype(np.float32)
        norm = np.linalg.norm(guess)
        if norm > 1e-6:
            guess = guess / norm
        match, _ = verify_biohash_template(guess, template)
        if match:
            matches += 1
    print(f"[RESULT] Classical Brute Force: Matches in {max_attempts} attempts: {matches}")
    return matches

# 4. Hill Climbing Attacks
def hill_climbing_attack(template, steps=100):
    # Black-box hill climbing using only match score/decision
    guess = np.random.normal(0, 1, size=template.shape)
    user_id = 'attacker'
    app_id = 'attack_app'
    best_score = -float('inf')
    best_match = False
    import contextlib
    import io
    f = io.StringIO()
    for step in range(steps):
        idx = random.randint(0, len(template)-1)
        guess[idx] += np.random.normal(0, 0.5)
        with contextlib.redirect_stdout(f):
            cancelable_guess, params = cancelable_system.generate_cancelable_template(guess, user_id, app_id)
            match, score = cancelable_system.verify_cancelable_template(guess, template, params)
        if score > best_score:
            best_score = score
            best_match = match
    print(f"[RESULT] Hill Climbing: Best score after {steps} steps: {best_score:.3f} | Match: {'SUCCESS' if best_match else 'FAIL'}")
    return best_score

# Classical attack: hill climbing for biohash
def classical_hill_climbing_attack(template, steps=100):
    guess = np.random.normal(0, 1, size=template.shape).astype(np.float32)
    norm = np.linalg.norm(guess)
    if norm > 1e-6:
        guess = guess / norm
    best_score = -float('inf')
    best_match = False
    for step in range(steps):
        idx = random.randint(0, len(template)-1)
        # Randomly perturb guess in float space
        guess[idx] += np.random.normal(0, 0.2)
        norm = np.linalg.norm(guess)
        if norm > 1e-6:
            guess = guess / norm
        match, score = verify_biohash_template(guess, template)
        if score > best_score:
            best_score = score
            best_match = match
    print(f"[RESULT] Classical Hill Climbing: Best score after {steps} steps: {best_score:.3f} | Match: {'SUCCESS' if best_match else 'FAIL'}")
    return best_score

if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu
    import datetime


    # Attack parameters
    hill_steps_list = [20, 50, 100]
    brute_force_attempts = 650

    # Sample 30 random fingerprints and 30 random faces from our datasets
    def sample_subjects(subjects, n):
        import random
        if len(subjects) <= n:
            return subjects
        return random.sample(subjects, n)

    fp_subjects = sample_subjects(fingerprint_data if fingerprint_data else [], 30)
    face_subjects = sample_subjects(face_data if face_data else [], 30)

    # Results storage
    results = []

    # Helper for Wilson CI
    def wilson_ci(successes, n, alpha=0.05):
        from math import sqrt
        if n == 0:
            return (0, 0)
        z = 1.96  # 95%
        phat = successes / n
        denom = 1 + z**2/n
        centre = (phat + z**2/(2*n)) / denom
        margin = z * sqrt(phat*(1-phat)/n + z**2/(4*n**2)) / denom
        return max(0, centre - margin), min(1, centre + margin)

    # Loop over all subjects and modalities
    for modality, subjects in [('fingerprint', fp_subjects), ('face', face_subjects)]:
        for i, sample in enumerate(subjects):
            user_id = sample.get('user_id', f'user_{i:03d}')
            img = sample['image']
            processed = data_pipeline.preprocess_image(img)
            if modality == 'fingerprint':
                features = feature_extractor.extract_fingerprint_features(processed)
            else:
                features = feature_extractor.extract_face_features(processed)

            # PQC-protected
            pqc_template, _ = cancelable_system.generate_cancelable_template(features, user_id, f'app_{modality}')
            # Classical-protected
            classical_template = generate_biohash_template(features, user_id, f'app_{modality}')

            for label, template in [
                (f'pqc_{modality}', pqc_template),
                (f'classical_{modality}', classical_template)
            ]:
                label_print = label.replace('classical_', 'Classical-Protected ').replace('pqc_', 'PQC-Protected ').replace('_', ' ').title()
                # Inversion
                inv_score = classical_template_inversion(template) if label.startswith('classical_') else template_inversion(template)
                # Cross-app correlation (genuine)
                corr_score = classical_correlation_attack(template) if label.startswith('classical_') else correlation_attack(template)
                # Brute force
                brute_matches = classical_brute_force_attack(template, brute_force_attempts) if label.startswith('classical_') else brute_force_attack(template, brute_force_attempts)
                brute_phat = brute_matches / brute_force_attempts
                brute_ci_low, brute_ci_high = wilson_ci(brute_matches, brute_force_attempts)
                # Hill climbing for each S
                hill_scores = []
                hill_successes = []
                for S in hill_steps_list:
                    if label.startswith('classical_'):
                        score = classical_hill_climbing_attack(template, S)
                        # Success if score >= 0.85 (classical threshold)
                        success = score >= 0.85
                    else:
                        score = hill_climbing_attack(template, S)
                        # Success if score >= 1.5 (PQC threshold)
                        success = score >= 1.5
                    hill_scores.append(score)
                    hill_successes.append(success)

                # Store per subject
                results.append({
                    'label': label_print,
                    'subject_id': user_id,
                    'inv_score': inv_score,
                    'corr_score': corr_score,
                    'brute_matches': brute_matches,
                    'brute_phat': brute_phat,
                    'brute_ci_low': brute_ci_low,
                    'brute_ci_high': brute_ci_high,
                    'hill_scores': hill_scores,
                    'hill_successes': hill_successes
                })

    # Unlinkability: genuine vs impostor cross-app cosine
    genuine_cosines = []
    impostor_cosines = []
    # For each modality, compare all pairs
    for modality, subjects in [('fingerprint', fp_subjects), ('face', face_subjects)]:
        n = len(subjects)
        # Get features for all
        features_list = []
        for i, sample in enumerate(subjects):
            img = sample['image']
            processed = data_pipeline.preprocess_image(img)
            if modality == 'fingerprint':
                features = feature_extractor.extract_fingerprint_features(processed)
            else:
                features = feature_extractor.extract_face_features(processed)
            features_list.append(features)
        # Genuine: same user, different app
        for i in range(n):
            f = features_list[i]
            t1 = generate_biohash_template(f, f'user_{i:03d}', 'app1')
            t2 = generate_biohash_template(f, f'user_{i:03d}', 'app2')
            norm1 = np.linalg.norm(t1)
            norm2 = np.linalg.norm(t2)
            cosine = np.dot(t1, t2) / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
            genuine_cosines.append(cosine)
        # Impostor: different users
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                f1 = features_list[i]
                f2 = features_list[j]
                t1 = generate_biohash_template(f1, f'user_{i:03d}', 'app1')
                t2 = generate_biohash_template(f2, f'user_{j:03d}', 'app2')
                norm1 = np.linalg.norm(t1)
                norm2 = np.linalg.norm(t2)
                cosine = np.dot(t1, t2) / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
                impostor_cosines.append(cosine)

    # Mann-Whitney U test
    if genuine_cosines and impostor_cosines:
        mw_stat, mw_p = mannwhitneyu(genuine_cosines, impostor_cosines, alternative='less')
        print(f"Unlinkability Mann-Whitney p-value: {mw_p:.4g}")

    # Export CSV
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    df = pd.DataFrame(results)
    # Expand hill scores/successes into columns
    for idx, S in enumerate(hill_steps_list):
        df[f'hill_score_{S}'] = df['hill_scores'].apply(lambda x: x[idx])
        df[f'hill_success_{S}'] = df['hill_successes'].apply(lambda x: x[idx])
    df.drop(['hill_scores', 'hill_successes'], axis=1, inplace=True)
    csv_path = f'attacks_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Raw attack results exported to {csv_path}")

    # Plot summary stats (median/IQR)
    import numpy as np
    labels = sorted(set(df['label']))
    medians = {k: [] for k in ['inv_score', 'corr_score', 'brute_phat']}
    iqrs = {k: [] for k in ['inv_score', 'corr_score', 'brute_phat']}
    for label in labels:
        sub = df[df['label'] == label]
        for k in medians:
            medians[k].append(np.median(sub[k]))
            iqrs[k].append(np.percentile(sub[k], 75) - np.percentile(sub[k], 25))

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width, medians['inv_score'], width, label='Inversion Median', color='tan', yerr=iqrs['inv_score'])
    ax.bar(x, medians['corr_score'], width, label='Correlation Median', color='cadetblue', yerr=iqrs['corr_score'])
    ax.bar(x + width, medians['brute_phat'], width, label='Brute Force Median', color='silver', yerr=iqrs['brute_phat'])
    ax.set_ylabel('Attack Score / Success Probability')
    ax.set_title('Attack Results: Median and IQR Across Subjects')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.legend()
    plt.tight_layout()
    fig.savefig('attack_comparison.png')
    print("\nComparative attack results graph saved as 'attack_comparison.png'.")

    # Plot unlinkability histograms
    plt.figure(figsize=(10,5))
    plt.hist(genuine_cosines, bins=30, alpha=0.7, label='Genuine (same user)', color='green')
    plt.hist(impostor_cosines, bins=30, alpha=0.7, label='Impostor (diff user)', color='red')
    plt.xlabel('Cross-App Cosine Similarity')
    plt.ylabel('Count')
    plt.title('Unlinkability: Genuine vs Impostor Cosine Distributions')
    plt.legend()
    plt.tight_layout()
    plt.savefig('unlinkability_histograms.png')
    print("Unlinkability histograms saved as 'unlinkability_histograms.png'.")
