"""Train a LightGBM URL classifier, evaluate at realistic base rates, export ONNX."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from phishguard_model.constants import BRANDS, FEATURE_NAMES  # noqa: E402
from phishguard_model.download import (  # noqa: E402
    load_openphish,
    load_phiusiil,
    load_phreshphish_urls,
    load_tranco_benign,
    load_urlhaus_hosts,
)
from phishguard_model.features import extract_vector, get_etld1, coerce_parse  # noqa: E402

DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
EXT_MODEL = ROOT.parent / "extension" / "public" / "model"


LOOKALIKE_PHISH = [
    "https://paypa1.com/signin",
    "https://paypal-secure-login.net/verify",
    "https://paypal.com.account-reset.biz/login",
    "https://www.paypaI.com/cgi-bin/webscr",
    "https://appleid-apple.com-verify.live/unlock",
    "https://apple.com.secure-id.net/account",
    "https://icl0ud.com/signin",
    "https://login-microsoftonline.com/common/oauth2",
    "https://microsoft-support-alert.com/verify",
    "https://office365-login.net/session",
    "https://amaz0n.com/ap/signin",
    "https://amazon-account-update.net/prime",
    "https://amazon.com.security-check.info/login",
    "https://faceb00k.com/login",
    "https://facebook.com.recover-account.xyz/login",
    "https://instagram-verify-badge.com/claim",
    "https://netfl1x.com/login",
    "https://netflix-billing-update.com/restart",
    "https://chase-online-banking.net/verify",
    "https://wellsfargo-securelogin.com/auth",
    "https://bankofamerica-alerts.net/secure",
    "https://citi-onlineverify.com/signin",
    "https://usps-package-hold.com/track",
    "https://fedex-delivery-fee.net/pay",
    "https://dhl-customs-invoice.com/pay",
    "https://irs-refund-status.net/claim",
    "https://coinbase-wallet-restore.com/seed",
    "https://binance-security-alert.com/login",
    "https://steamcommunity.com.login-auth.ru/signin",
    "https://adobe-id-confirm.com/acrobat",
    "https://dropbox-sharedfile.com/view",
    "https://docusign-docs.net/sign",
    "https://linkedin-security.net/checkpoint",
    "https://192.168.8.12/paypal/login",
    "http://185.22.11.9/verify-account",
    "https://secure-login.paypa1-support.com/update",
    "https://accounts.g00gle.com/signin",
    "https://google.com-verify-user.net/alert",
    "https://whatsapp-web-login.net/qr",
    "https://icloud.com.account-locked.net/find",
]

LEGIT_GOLD = [
    "https://www.paypal.com/signin",
    "https://www.apple.com/icloud/",
    "https://appleid.apple.com/",
    "https://accounts.google.com/ServiceLogin",
    "https://mail.google.com/mail/",
    "https://login.microsoftonline.com/",
    "https://outlook.live.com/mail/",
    "https://www.amazon.com/gp/css/homepage.html",
    "https://www.facebook.com/login",
    "https://www.instagram.com/accounts/login/",
    "https://www.netflix.com/login",
    "https://www.chase.com/",
    "https://www.wellsfargo.com/",
    "https://www.bankofamerica.com/",
    "https://www.usps.com/manage/welcome.htm",
    "https://www.fedex.com/en-us/tracking.html",
    "https://www.dhl.com/en/express/tracking.html",
    "https://www.irs.gov/",
    "https://www.coinbase.com/signin",
    "https://www.binance.com/",
    "https://store.steampowered.com/login/",
    "https://auth.adobe.com/",
    "https://www.dropbox.com/login",
    "https://account.docusign.com/",
    "https://www.linkedin.com/login",
    "https://github.com/login",
    "https://www.wikipedia.org/",
    "https://news.ycombinator.com/",
    "https://www.nytimes.com/",
    "https://www.bbc.com/news",
    "https://www.reddit.com/",
    "https://stackoverflow.com/questions",
    "https://www.cloudflare.com/",
    "https://vercel.com/dashboard",
    "https://www.shopify.com/",
    "https://www.spotify.com/account/",
    "https://zoom.us/signin",
    "https://slack.com/signin",
    "https://www.intuit.com/",
    "https://www.walmart.com/",
]


SYNTH_TLDS = ["net", "xyz", "top", "info", "live", "club", "shop", "online", "site"]
SYNTH_PATHS = ["/login", "/signin", "/verify", "/account", "/secure", "/update", "/confirm"]
LEGIT_PATHS = ["/", "/about", "/help", "/blog", "/pricing", "/news", "/privacy", "/login"]


def generate_synthetic(n_phish: int = 400, n_legit: int = 400) -> pd.DataFrame:
    rows: list[tuple[str, int]] = []
    i = 0
    for brand in BRANDS:
        token = brand["tokens"][0]
        official = brand["domains"][0]
        rows.append((f"https://{official}{LEGIT_PATHS[i % len(LEGIT_PATHS)]}", 0))
        rows.append((f"https://www.{official}/", 0))
        typo = token.replace("o", "0").replace("l", "1").replace("e", "3") or (token[:-1] + "1")
        tld = SYNTH_TLDS[i % len(SYNTH_TLDS)]
        path = SYNTH_PATHS[i % len(SYNTH_PATHS)]
        rows.append((f"https://{typo}.{tld}{path}", 1))
        rows.append((f"https://{token}-secure-login.{tld}{path}", 1))
        rows.append((f"https://{token}.com.account-reset.{tld}{path}", 1))
        i += 1
    while len([r for r in rows if r[1] == 1]) < n_phish:
        rows.append((f"http://185.22.{i % 250}.{10 + i % 80}/login", 1))
        i += 1
    return pd.DataFrame(rows, columns=["url", "label"]).drop_duplicates(subset=["url"])


def host_of(url: str) -> str:
    parsed = coerce_parse(url)
    if not parsed:
        return url.lower()
    return get_etld1(parsed[1])


def vectors_from_urls(urls: list[str]) -> np.ndarray:
    return np.asarray([extract_vector(u) for u in urls], dtype=np.float32)


def assemble_dataset(include_phresh: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    try:
        print("Loading PhiUSIIL…")
        frames.append(load_phiusiil(DATA_DIR))
        print(f"  PhiUSIIL rows: {len(frames[-1])}")
    except Exception as exc:
        print(f"  PhiUSIIL unavailable ({exc})")

    extra = load_urlhaus_hosts(DATA_DIR)
    if len(extra):
        print(f"  URLhaus hosts: {len(extra)}")
        frames.append(extra)
    extra = load_openphish(DATA_DIR)
    if len(extra):
        print(f"  OpenPhish urls: {len(extra)}")
        frames.append(extra)
    extra = load_tranco_benign(DATA_DIR, limit=25000)
    if len(extra):
        print(f"  Tranco benign: {len(extra)}")
        frames.append(extra)
    if include_phresh:
        extra = load_phreshphish_urls(limit=40000)
        if len(extra):
            print(f"  PhreshPhish urls: {len(extra)}")
            frames.append(extra)

    gold_phish = pd.DataFrame({"url": LOOKALIKE_PHISH, "label": 1})
    gold_legit = pd.DataFrame({"url": LEGIT_GOLD, "label": 0})
    synthetic = generate_synthetic()
    print(f"  Synthetic lookalike/legit: {len(synthetic)}")
    frames.extend([gold_phish, gold_legit, synthetic])

    if not frames:
        raise SystemExit("No training data could be downloaded.")

    df = pd.concat(frames, ignore_index=True)
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"].str.len() > 5].drop_duplicates(subset=["url"])
    df["host"] = df["url"].map(host_of)
    print(f"Combined unique URLs: {len(df)}  (phish={int(df.label.sum())} legit={int((df.label==0).sum())})")
    return df


def split_by_host(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    hosts = df["host"].drop_duplicates()
    train_hosts, test_hosts = train_test_split(hosts, test_size=test_size, random_state=seed)
    train = df[df["host"].isin(set(train_hosts))]
    test = df[df["host"].isin(set(test_hosts))]
    return train, test


def featurize(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = vectors_from_urls(df["url"].tolist())
    y = df["label"].to_numpy(dtype=np.int32)
    return X, y


def eval_prevalence(y_true: np.ndarray, scores: np.ndarray, prevalence: float, n: int = 40000, seed: int = 0):
    rng = np.random.default_rng(seed)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return {"precision": 0.0, "recall": 0.0, "n": 0}
    n_pos = max(1, int(n * prevalence))
    n_neg = max(1, n - n_pos)
    sp = rng.choice(pos, size=n_pos, replace=len(pos) < n_pos)
    sn = rng.choice(neg, size=n_neg, replace=len(neg) < n_neg)
    s = np.concatenate([sp, sn])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    return s, y


def metrics_at_threshold(y: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    pred = scores >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": thr, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def choose_threshold(y: np.ndarray, scores: np.ndarray, min_precision: float = 0.99) -> float:
    mix_scores, mix_y = eval_prevalence(y, scores, prevalence=0.01)
    precision, recall, thresholds = precision_recall_curve(mix_y, mix_scores)
    best_thr = 0.85
    best_rec = -1.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= min_precision and r > best_rec:
            best_rec = r
            best_thr = float(t)
    if best_rec < 0:
        fbeta = (1.25 * precision[:-1] * recall[:-1]) / np.clip(0.25 * precision[:-1] + recall[:-1], 1e-9, None)
        idx = int(np.nanargmax(fbeta))
        best_thr = float(thresholds[min(idx, len(thresholds) - 1)])
    return float(np.clip(best_thr, 0.05, 0.99))


def export_onnx(model, path: Path) -> None:
    from onnxmltools.convert.common.data_types import FloatTensorType

    n = len(FEATURE_NAMES)
    if hasattr(model, "booster_"):
        from onnxmltools.convert import convert_lightgbm

        onnx_model = convert_lightgbm(
            model.booster_,
            initial_types=[("input", FloatTensorType([None, n]))],
            target_opset=15,
            zipmap=False,
        )
    else:
        from skl2onnx import convert_sklearn

        onnx_model = convert_sklearn(
            model,
            initial_types=[("input", FloatTensorType([None, n]))],
            target_opset=15,
            options={id(model): {"zipmap": False}},
        )
    path.write_bytes(onnx_model.SerializeToString())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phreshphish", action="store_true")
    parser.add_argument("--max-rows", type=int, default=120000)
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    EXT_MODEL.mkdir(parents=True, exist_ok=True)

    df = assemble_dataset(include_phresh=args.phreshphish)
    if len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=42)

    train_df, test_df = split_by_host(df)
    print(f"Train {len(train_df)} / test {len(test_df)}", flush=True)

    print("Extracting features…", flush=True)
    X_train, y_train = featurize(train_df)
    X_test, y_test = featurize(test_df)

    try:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=63,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=40,
            objective="binary",
            n_jobs=-1,
            random_state=42,
        )
        print("Training LightGBM…", flush=True)
    except OSError as exc:
        from sklearn.ensemble import HistGradientBoostingClassifier

        print(f"LightGBM unavailable ({exc}); using HistGradientBoosting", flush=True)
        model = HistGradientBoostingClassifier(
            max_depth=10,
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=63,
            random_state=42,
        )

    model.fit(X_train, y_train)

    raw_test = model.predict_proba(X_test)[:, 1]
    calibrator = LogisticRegression(max_iter=200)
    calibrator.fit(raw_test.reshape(-1, 1), y_test)
    scores = calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]

    auc = float(roc_auc_score(y_test, scores)) if len(np.unique(y_test)) > 1 else 0.0
    ap = float(average_precision_score(y_test, scores)) if len(np.unique(y_test)) > 1 else 0.0
    thr = choose_threshold(y_test, scores, min_precision=0.99)

    report = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "roc_auc": auc,
        "pr_auc": ap,
        "threshold": thr,
        "features": FEATURE_NAMES,
        "prevalence": {},
        "gold": {},
    }

    for prev in (0.001, 0.01):
        mix_scores, mix_y = eval_prevalence(y_test, scores, prevalence=prev)
        report["prevalence"][str(prev)] = metrics_at_threshold(mix_y, mix_scores, thr)

    gold_urls = LOOKALIKE_PHISH + LEGIT_GOLD
    gold_raw = model.predict_proba(vectors_from_urls(gold_urls))[:, 1]
    gold_s = calibrator.predict_proba(gold_raw.reshape(-1, 1))[:, 1]
    gold_pred = gold_s >= thr
    report["gold"] = {
        "lookalike_recall": float(gold_pred[: len(LOOKALIKE_PHISH)].mean()),
        "legit_false_positive_rate": float(gold_pred[len(LOOKALIKE_PHISH) :].mean()),
        "n_lookalike": len(LOOKALIKE_PHISH),
        "n_legit": len(LEGIT_GOLD),
    }
    print(json.dumps(report, indent=2), flush=True)

    import joblib

    joblib.dump({"model": model, "calibrator": calibrator}, ARTIFACTS / "model.joblib")
    onnx_path = ARTIFACTS / "model.onnx"
    print(f"Exporting ONNX → {onnx_path}", flush=True)
    export_onnx(model, onnx_path)

    calib = {"intercept": float(calibrator.intercept_[0]), "coef": float(calibrator.coef_[0][0])}
    meta = {
        "features": FEATURE_NAMES,
        "threshold": thr,
        "danger": max(thr, 0.85),
        "suspicious": min(0.45, max(0.35, thr * 0.6)),
        "calibration": calib,
        "metrics": report,
    }
    (ARTIFACTS / "model_meta.json").write_text(json.dumps(meta, indent=2))
    (ARTIFACTS / "features.json").write_text(json.dumps(FEATURE_NAMES, indent=2))

    target_onnx = EXT_MODEL / "model.onnx"
    target_meta = EXT_MODEL / "model_meta.json"
    target_onnx.write_bytes(onnx_path.read_bytes())
    target_meta.write_text(json.dumps(meta, indent=2))
    print(f"Copied model to {target_onnx}", flush=True)


if __name__ == "__main__":
    main()
