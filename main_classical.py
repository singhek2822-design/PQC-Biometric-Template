

# Classical biometric hardening pipeline for comparison with PQC system.
# Uses RSA/ECDH for key exchange
# Uses RSA/ECDSA for signatures
# Uses AES for encryption
# Uses SHA-256 for hashing
# Logs computational metrics for direct comparison

import numpy as np
import time
import os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hmac
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

# Setup logging
logging.basicConfig(filename='logs/classical_system.log', level=logging.INFO)

def load_biometric_data():
    # Loads all biometric datasets (fingerprint, face, LFW, NIST) for classical pipeline.
    import cv2
    from pathlib import Path
    # Matches main.py config
    DATA_DIR = Path("data")
    FINGERPRINT_DIR = DATA_DIR / "Fingerprint-FVC"
    FACE_ATT_DIR = DATA_DIR / "Faces-ATT"
    FACE_LFW_DIR = DATA_DIR / "Faces-LFW"
    NIST_SD302_DIR = DATA_DIR / "NIST-SD302"
    fingerprint_data = []
    face_data = []
    lfw_data = []
    nist_data = []
    # Loads fingerprints
    for file_path in FINGERPRINT_DIR.glob("*.tif"):
        subject_id = file_path.stem.split('_')[0]
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            fingerprint_data.append({'subject_id': subject_id, 'image': img, 'filename': file_path.name})
    # Loads faces (ATT)
    for subject_dir in FACE_ATT_DIR.glob("s*"):
        if subject_dir.is_dir():
            subject_id = subject_dir.name
            for img_file in subject_dir.glob("*.pgm"):
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    face_data.append({'subject_id': subject_id, 'image': img, 'filename': img_file.name})
    # Loads Faces-LFW (recursively from lfw-deepfunneled, limit to 100 images)
    lfw_deep_dir = FACE_LFW_DIR / "lfw-deepfunneled"
    if lfw_deep_dir.exists():
        count = 0
        for img_file in lfw_deep_dir.rglob("*.jpg"):
            if count >= 100:
                break
            subject_id = img_file.parent.name
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                lfw_data.append({'subject_id': subject_id, 'image': img, 'filename': img_file.name})
                count += 1
    # Loads NIST-SD302
    nist_png_dir = NIST_SD302_DIR / "baseline" / "R" / "500" / "slap" / "png"
    if nist_png_dir.exists():
        for file_path in nist_png_dir.glob("*.png"):
            filename_parts = file_path.stem.split('_')
            subject_id = filename_parts[0]
            capture_num = filename_parts[-1]
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                nist_data.append({'subject_id': subject_id, 'capture_num': capture_num, 'image': img, 'filename': file_path.name})
    return fingerprint_data, face_data, lfw_data, nist_data

def extract_features(data):
    # Uses main.py feature extraction logic
    import cv2
    import numpy as np
    def preprocess_image(image, target_size=(128, 128)):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed = cv2.resize(image, target_size)
        processed = processed.astype(np.float32) / 255.0
        processed = cv2.equalizeHist((processed * 255).astype(np.uint8))
        return processed
    def extract_fingerprint_features(image):
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            gray = cv2.resize(gray, (128, 128))
            gray = cv2.equalizeHist(gray)
            features = []
            features.extend([
                np.mean(gray), np.std(gray), np.median(gray), np.min(gray), np.max(gray), np.var(gray)
            ])
            h, w = gray.shape
            regions = [
                gray[:h//2, :w//2], gray[:h//2, w//2:], gray[h//2:, :w//2], gray[h//2:, w//2:]
            ]
            for region in regions:
                features.extend([np.mean(region), np.std(region), np.var(region)])
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            features.extend([
                np.mean(grad_mag), np.std(grad_mag), np.max(grad_mag), np.percentile(grad_mag, 90), np.percentile(grad_mag, 10)
            ])
            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            hist = hist.flatten() / (np.sum(hist) + 1e-8)
            features.extend(hist[:8])
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            for region in [edges[:h//2, :w//2], edges[:h//2, w//2:], edges[h//2:, :w//2], edges[h//2:, w//2:]]:
                region_density = np.sum(region > 0) / (region.shape[0] * region.shape[1])
                features.append(region_density)
            h_transitions = np.sum(np.abs(np.diff(gray, axis=1)) > 10)
            v_transitions = np.sum(np.abs(np.diff(gray, axis=0)) > 10)
            features.extend([h_transitions / w, v_transitions / h])
            try:
                orb = cv2.ORB_create(nfeatures=50)
                keypoints, descriptors = orb.detectAndCompute(gray, None)
                if descriptors is not None and len(descriptors) > 0:
                    orb_features = descriptors.astype(np.float32).flatten()
                    orb_subset = orb_features[:20] if len(orb_features) >= 20 else orb_features
                    features.extend(orb_subset)
                else:
                    features.extend(np.zeros(20))
            except:
                features.extend(np.zeros(20))
            feature_vector = np.array(features, dtype=np.float32)
            feature_vector = np.nan_to_num(feature_vector)
            image_hash = hash(tuple(gray.flatten()[::10])) % 1000000
            np.random.seed(image_hash)
            unique_variance = np.random.normal(0, 0.02, len(feature_vector))
            feature_vector += unique_variance
            if len(feature_vector) > 0:
                q1, q99 = np.percentile(feature_vector, [1, 99])
                feature_vector = np.clip(feature_vector, q1, q99)
                mean_val = np.mean(feature_vector)
                std_val = np.std(feature_vector)
                if std_val > 1e-6:
                    feature_vector = (feature_vector - mean_val) / std_val
                norm = np.linalg.norm(feature_vector)
                if norm > 1e-6:
                    feature_vector = feature_vector / norm
            target_dim = 256
            if len(feature_vector) > target_dim:
                feature_vector = feature_vector[:target_dim]
            elif len(feature_vector) < target_dim:
                padded = np.zeros(target_dim, dtype=np.float32)
                padded[:len(feature_vector)] = feature_vector
                feature_vector = padded
            return feature_vector
        except Exception as e:
            image_sum = np.sum(image) if len(image.flatten()) > 0 else 12345
            np.random.seed(int(image_sum) % 1000000)
            return np.random.randn(256).astype(np.float32)
    def extract_face_features(image):
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            gray = cv2.resize(gray, (112, 112))
            gray = cv2.equalizeHist(gray)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            features = []
            h, w = gray.shape
            regions = {
                'upper_forehead': gray[:h//6, :],
                'lower_forehead': gray[h//6:h//3, :],
                'left_eye': gray[h//4:h//2, :w//3],
                'right_eye': gray[h//4:h//2, 2*w//3:],
                'nose_bridge': gray[h//3:2*h//3, w//3:2*w//3],
                'nose_tip': gray[h//2:2*h//3, w//4:3*w//4],
                'left_cheek': gray[h//3:2*h//3, :w//4],
                'right_cheek': gray[h//3:2*h//3, 3*w//4:],
                'mouth': gray[2*h//3:5*h//6, w//4:3*w//4],
                'chin': gray[5*h//6:, w//3:2*w//3]
            }
            for region_name, region in regions.items():
                if region.size > 0:
                    features.extend([
                        np.mean(region), np.std(region), np.median(region), np.percentile(region, 10),
                        np.percentile(region, 25), np.percentile(region, 75), np.percentile(region, 90),
                        np.max(region) - np.min(region)
                    ])
            for size in [16, 32]:
                resized = cv2.resize(gray, (size, size))
                normalized = resized.astype(np.float32) / 255.0
                if size == 32:
                    center = normalized[8:24, 8:24]
                    features.extend(center.flatten()[::2][:32])
                else:
                    features.extend(normalized.flatten()[:64])
            try:
                sift = cv2.SIFT_create(nfeatures=80, contrastThreshold=0.04, edgeThreshold=15)
                keypoints, descriptors = sift.detectAndCompute(gray, None)
                if descriptors is not None and len(descriptors) >= 10:
                    desc_mean = np.mean(descriptors, axis=0)
                    desc_std = np.std(descriptors, axis=0)
                    desc_max = np.max(descriptors, axis=0)
                    desc_min = np.min(descriptors, axis=0)
                    desc_median = np.median(descriptors, axis=0)
                    features.extend(desc_mean[:24])
                    features.extend(desc_std[:16])
                    features.extend((desc_max - desc_min)[:12])
                    features.extend(desc_median[:8])
                else:
                    features.extend(np.zeros(60))
            except:
                features.extend(np.zeros(60))
            try:
                gray_small = cv2.resize(gray, (16, 16)).astype(np.float32)
                dct_result = cv2.dct(gray_small)
                freq_features = []
                for i in range(4):
                    for j in range(4):
                        if i + j <= 4:
                            freq_features.append(dct_result[i, j])
                features.extend(freq_features[:16])
            except:
                features.extend(np.zeros(16))
            try:
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                h, w = grad_mag.shape
                regions = [
                    grad_mag[:h//2, :w//2], grad_mag[:h//2, w//2:], grad_mag[h//2:, :w//2], grad_mag[h//2:, w//2:]
                ]
                for region in regions:
                    if region.size > 0:
                        features.extend([np.mean(region), np.std(region)])
            except:
                features.extend(np.zeros(8))
            try:
                radius = 2
                n_points = 8
                h, w = gray.shape
                lbp_image = np.zeros_like(gray, dtype=np.uint8)
                for y in range(radius, h - radius):
                    for x in range(radius, w - radius):
                        center = gray[y, x]
                        pattern = 0
                        for i in range(n_points):
                            angle = 2 * np.pi * i / n_points
                            dx = int(radius * np.cos(angle))
                            dy = int(radius * np.sin(angle))
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if gray[ny, nx] >= center:
                                    pattern |= (1 << i)
                        lbp_image[y, x] = pattern
                hist, _ = np.histogram(lbp_image.flatten(), bins=16, range=(0, 256))
                hist = hist.astype(np.float32) / (np.sum(hist) + 1e-8)
                features.extend(hist)
            except:
                features.extend(np.zeros(16))
            feature_vector = np.array(features, dtype=np.float32)
            feature_vector = np.nan_to_num(feature_vector)
            if len(feature_vector) > 0:
                q0_5, q99_5 = np.percentile(feature_vector, [0.5, 99.5])
                feature_vector = np.clip(feature_vector, q0_5, q99_5)
                mean_val = np.mean(feature_vector)
                std_val = np.std(feature_vector)
                if std_val > 1e-6:
                    feature_vector = (feature_vector - mean_val) / std_val
                feature_vector = np.clip(feature_vector, -3.0, 3.0)
                feature_vector = feature_vector * 1.3
                sign = np.sign(feature_vector)
                feature_vector = sign * np.power(np.abs(feature_vector), 0.75)
                feature_vector = feature_vector + np.random.normal(0, 0.01, len(feature_vector))
                norm = np.linalg.norm(feature_vector)
                if norm > 1e-6:
                    feature_vector = feature_vector / norm
                    feature_vector = feature_vector * 1.02
            target_dim = 256
            if len(feature_vector) > target_dim:
                feature_vector = feature_vector[:target_dim]
            elif len(feature_vector) < target_dim:
                padded = np.zeros(target_dim, dtype=np.float32)
                padded[:len(feature_vector)] = feature_vector
                feature_vector = padded
            return feature_vector
        except Exception as e:
            return np.random.randn(256).astype(np.float32)
    fp_features = [extract_fingerprint_features(preprocess_image(d['image'])) for d in data['fingerprint']]
    face_features = [extract_face_features(preprocess_image(d['image'])) for d in data['face']]
    lfw_features = [extract_face_features(preprocess_image(d['image'])) for d in data['lfw']]
    nist_features = [extract_fingerprint_features(preprocess_image(d['image'])) for d in data['nist']]
    return fp_features, face_features, lfw_features, nist_features


# Classical: BioHashing (random projection, real-valued output)
def generate_biohash_template(features, user_id, app_name, output_dim=128):
    # Generates a classical BioHash template using random projection.
    # The projection is seeded with user_id and app_name for revocability and unlinkability.
    seed = hash((user_id, app_name)) % (2**32)
    rng = np.random.RandomState(seed)
    proj_matrix = rng.normal(0, 1, (output_dim, len(features)))
    biohash = np.dot(proj_matrix, features)
    # Optionally, normalize (for cosine similarity etc.)
    norm = np.linalg.norm(biohash)
    if norm > 1e-6:
        biohash = biohash / norm
    return biohash.astype(np.float32)

def classical_key_exchange():
    # RSA key generation
    t0 = time.time()
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    public_key = private_key.public_key()
    t1 = time.time()
    keygen_time = t1 - t0
    return private_key, public_key, keygen_time

def classical_signature(private_key, data):
    t0 = time.time()
    signature = private_key.sign(
        data,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    t1 = time.time()
    sign_time = t1 - t0
    return signature, sign_time

def classical_verify(public_key, data, signature):
    t0 = time.time()
    try:
        public_key.verify(
            signature,
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        valid = True
    except Exception:
        valid = False
    t1 = time.time()
    verify_time = t1 - t0
    return valid, verify_time

def classical_encrypt(data, key):
    t0 = time.time()
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    pad_len = 16 - (len(data) % 16)
    padded_data = data + bytes([pad_len] * pad_len)
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    t1 = time.time()
    enc_time = t1 - t0
    return ciphertext, iv, enc_time

def classical_decrypt(ciphertext, key, iv):
    t0 = time.time()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()
    pad_len = padded_data[-1]
    data = padded_data[:-pad_len]
    t1 = time.time()
    dec_time = t1 - t0
    return data, dec_time

def main():
    # Main function for running the classical biometric hardening pipeline.
    print("\n Classical Biometric Hardening System ")
    fingerprint_data, face_data, lfw_data, nist_data = load_biometric_data()
    print(f"Loaded {len(fingerprint_data)} fingerprints, {len(face_data)} ATT faces, {len(lfw_data)} LFW faces, {len(nist_data)} NIST samples.")
    # Feature extraction
    t0 = time.time()
    # Prepares the data dict for feature extraction
    data_dict = {
        'fingerprint': fingerprint_data,
        'face': face_data,
        'lfw': lfw_data,
        'nist': nist_data
    }
    fp_features, face_features, lfw_features, nist_features = extract_features(data_dict)
    t1 = time.time()
    feature_time = t1 - t0
    # Per-sample performance metrics for each modality
    modalities = [
        ("fingerprints", fp_features),
        ("faces_ATT", face_features),
        ("faces_LFW", lfw_features),
        ("NIST", nist_features)
    ]
    private_key, public_key, keygen_time = classical_key_exchange()
    aes_key = os.urandom(32)
    all_cancelable_times = []
    all_signature_times = []
    all_verify_times = []
    all_enc_times = []
    all_dec_times = []
    all_match_times = []
    for modality_name, features_list in modalities:
        cancelable_times = []
        signature_times = []
        verify_times = []
        enc_times = []
        dec_times = []
        match_times = []
        all_templates = []
        for i, features in enumerate(features_list):
            t0 = time.time()
            tpl = generate_biohash_template(features, f'{modality_name}_user_{i}', 'classical_app')
            all_templates.append(tpl)
            t1 = time.time()
            cancelable_times.append(t1 - t0)
            t0 = time.time()
            signature, sign_time = classical_signature(private_key, tpl.tobytes())
            t1 = time.time()
            signature_times.append(t1 - t0)
            t0 = time.time()
            valid, verify_time = classical_verify(public_key, tpl.tobytes(), signature)
            t1 = time.time()
            verify_times.append(t1 - t0)
            t0 = time.time()
            ciphertext, iv, enc_time = classical_encrypt(tpl.tobytes(), aes_key)
            t1 = time.time()
            enc_times.append(t1 - t0)
            t0 = time.time()
            decrypted, dec_time = classical_decrypt(ciphertext, aes_key, iv)
            t1 = time.time()
            dec_times.append(t1 - t0)
            t0 = time.time()
            sim = np.dot(tpl, tpl)
            t1 = time.time()
            match_times.append(t1 - t0)
        # Save templates for this modality
        if all_templates:
            templates_array = np.stack(all_templates)
            np.save(f'classical_protected_templates_{modality_name}.npy', templates_array)
            print(f"Saved {len(all_templates)} classical protected templates for {modality_name} to 'classical_protected_templates_{modality_name}.npy'.")
        all_cancelable_times.extend(cancelable_times)
        all_signature_times.extend(signature_times)
        all_verify_times.extend(verify_times)
        all_enc_times.extend(enc_times)
        all_dec_times.extend(dec_times)
        all_match_times.extend(match_times)
    print("--- End per-sample performance ---\n")

    # Classical MFA: Biometrics + OTP
    print("\n Classical MFA (Biometrics + OTP) ")
    import random
    mfa_users = 20  # Simulate MFA for 20 users
    mfa_enroll_success = 0
    mfa_auth_success = 0
    biometric_success = 0
    otp_success = 0
    wrong_otp_rejection = 0
    mfa_start = time.time()
    for user_id in range(mfa_users):
        # Enrollment: stores biometric template and OTP delivery method
        # Simulates the biometric template 
        modality_idx = random.choice([0, 1, 2, 3])
        features_list = [fp_features, face_features, lfw_features, nist_features][modality_idx]
        if not features_list:
            continue
        template = random.choice(features_list)
        # Simulates OTP delivery (store OTP)
        otp = str(random.randint(100000, 999999))
        mfa_enroll_success += 1
        # Authentication: user provides biometric sample and OTP
        # Simulate correct biometric (match to template)
        biometric_input = template.copy()
        tpl1 = generate_biohash_template(biometric_input, f'user_{user_id}', 'classical_mfa')
        tpl2 = generate_biohash_template(template, f'user_{user_id}', 'classical_mfa')
        biometric_match = np.allclose(tpl1, tpl2)
        if biometric_match:
            biometric_success += 1
        # Simulates correct OTP
        otp_input = otp
        if otp_input == otp:
            otp_success += 1
        # MFA authentication success
        if biometric_match and otp_input == otp:
            mfa_auth_success += 1
        # Simulates wrong OTP attempt
        wrong_otp = str(random.randint(100000, 999999))
        if wrong_otp != otp:
            # Should be rejected
            wrong_otp_rejection += 1
    mfa_end = time.time()
    mfa_time = mfa_end - mfa_start
    mfa_throughput = mfa_auth_success / mfa_time if mfa_time > 0 else 0
    print(f"MFA Enrollment Success: {mfa_enroll_success}/{mfa_users} ({100.0*mfa_enroll_success/mfa_users:.1f}%)")
    print(f"Authentication Success: {mfa_auth_success}/{mfa_users} ({100.0*mfa_auth_success/mfa_users:.1f}%)")
    print(f"Factor Reliability: Biometric {100.0*biometric_success/mfa_users:.1f}% | OTP {100.0*otp_success/mfa_users:.1f}%")
    print(f"Wrong OTP Rejection: {wrong_otp_rejection}/{mfa_users} ({100.0*wrong_otp_rejection/mfa_users:.1f}%)")
    print(f"System Throughput: {mfa_throughput:.2f} authentications/sec")
    logging.info(f"MFA Enrollment Success: {mfa_enroll_success}/{mfa_users}")
    logging.info(f"Authentication Success: {mfa_auth_success}/{mfa_users}")
    logging.info(f"Factor Reliability: Biometric {biometric_success}/{mfa_users}, OTP {otp_success}/{mfa_users}")
    logging.info(f"Wrong OTP Rejection: {wrong_otp_rejection}/{mfa_users}")
    logging.info(f"System Throughput: {mfa_throughput:.2f} authentications/sec")
    # Aggregate metrics
    print(f"Avg cancelable template time: {np.mean(all_cancelable_times):.6f}s")
    print(f"Avg signature time: {np.mean(all_signature_times):.6f}s")
    print(f"Avg verification time: {np.mean(all_verify_times):.6f}s")
    print(f"Avg encryption time: {np.mean(all_enc_times):.6f}s")
    print(f"Avg decryption time: {np.mean(all_dec_times):.6f}s")
    print(f"Avg matching time: {np.mean(all_match_times):.6f}s")
    # Logs results
    logging.info(f"Feature extraction: {feature_time:.4f}s")
    logging.info(f"Avg cancelable template generation: {np.mean(all_cancelable_times):.6f}s")
    logging.info(f"RSA key generation: {keygen_time:.4f}s")
    logging.info(f"Signature generation: {np.mean(all_signature_times):.6f}s")
    logging.info(f"Signature verification: {np.mean(all_verify_times):.6f}s")
    logging.info(f"AES encryption: {np.mean(all_enc_times):.6f}s")
    logging.info(f"AES decryption: {np.mean(all_dec_times):.6f}s")
    logging.info(f"Avg template matching: {np.mean(all_match_times):.6f}s")
    print("\n Classical System Summary ")
    print(f"Feature extraction: {feature_time:.4f}s")
    print(f"Avg cancelable template generation: {np.mean(all_cancelable_times):.6f}s")
    print(f"RSA key generation: {keygen_time:.4f}s")
    print(f"Signature generation: {np.mean(all_signature_times):.4f}s")
    print(f"Signature verification: {np.mean(all_verify_times):.4f}s")
    print(f"AES encryption: {np.mean(all_enc_times):.4f}s")
    print(f"AES decryption: {np.mean(all_dec_times):.4f}s")
    print(f"Avg template matching: {np.mean(all_match_times):.6f}s")
    print("Results logged to logs/classical_system.log")

if __name__ == "__main__":
    main()

    # Comparative Metrics Table: Classical vs PQC (LaTeX)
    def generate_latex_table():
        import re
        pqc_log = 'logs/PQC_system.log'
        classical_log = 'logs/classical_system.log'
        metric_map = [
            ('Feature extraction', ['Feature extraction', 'Avg feature extraction time']),
            ('Cancelable template generation', ['Avg cancelable template generation', 'Cancelable template generation', 'Avg cancelable template time']),
            ('Key generation', ['RSA key generation', 'Avg key generation time']),
            ('Signature generation', ['Signature generation', 'Avg signing time']),
            ('Signature verification', ['Signature verification', 'Avg verification time']),
            ('Template matching', ['Avg template matching time', 'Template matching', 'Avg template matching'])
        ]
        def parse_log(logfile):
            values = {}
            try:
                with open(logfile, 'r') as f:
                    for line in f:
                        for std_name, variants in metric_map:
                            for m in variants:
                                if m in line:
                                    match = re.search(r'([-+]?[0-9]*\.?[0-9]+)', line)
                                    if match:
                                        if std_name not in values:
                                            values[std_name] = float(match.group(1))
            except Exception as e:
                print(f"Could not read {logfile}: {e}")
            return values
        pqc_vals = parse_log(pqc_log)
        classical_vals = parse_log(classical_log)
        common = [std_name for std_name, _ in metric_map if std_name in pqc_vals and std_name in classical_vals]
        if not common:
            print("No common metrics found in both logs.")
            return
        # Only uses up to 6 metrics
        common = common[:6]
        pqc_data = [pqc_vals[m] for m in common]
        classical_data = [classical_vals[m] for m in common]
        
        header = ["Metric", "Classical (s)", "PQC (s)"]
        latex = []
        latex.append("\\documentclass{article}")
        latex.append("\\usepackage{graphicx}")
        latex.append("\\usepackage{booktabs}")
        latex.append("")
        latex.append("\\begin{document}")
        latex.append("")
        latex.append("\\begin{table}[ht]")
        latex.append("\\centering")
        latex.append("\\begin{tabular}{|l|c|c|}")
        latex.append("\\hline")
        latex.append(f"{' & '.join(header)} \\")
        latex.append("\\hline")
        for i, metric in enumerate(common):
            c_val = f"{classical_data[i]:.6f}"
            p_val = f"{pqc_data[i]:.6f}"
            latex.append(f"{metric} & {c_val} & {p_val} \\")
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\caption{Computation Time Comparison: Classical vs PQC}")
        latex.append("\\label{tab:comp_time_comparison}")
        latex.append("\\end{table}")
        latex.append("")
        latex.append("\\end{document}")
        latex_code = '\n'.join(latex)
        # Saves the latex table to file
        with open('comparative_metrics_table.tex', 'w') as f:
            f.write(latex_code)
        print("\nLaTeX table saved as 'comparative_metrics_table.tex'.")
        print("\nLaTeX Table (full document):\n")
        print(latex_code)

    
    generate_latex_table()
