#!/usr/bin/env python
"""R2 S3 预签名 URL 生成器(纯标准库 SigV4,只在本地 Mac 运行)。

背景:spark 侧 workers.dev 被 DNS 污染+SNI 阻断,而 S3 端点
<account>.r2.cloudflarestorage.com 实测可达(HTTP 400 匿名拒绝)。回程通道
改用预签名 URL:Mac 持有 S3 密钥并生成限时单对象 URL,spark 只见 URL——
无凭据落节点、无常驻端点、URL 到期自动失效。

凭据(只入本地 .env,永不进 git/spark):
  R2_ACCOUNT_ID=<cloudflare account id>
  R2_ACCESS_KEY_ID=<r2 api token access key>
  R2_SECRET_ACCESS_KEY=<r2 api token secret>

用法(每行一个 key,stdout 对应每行一条 URL):
  printf '%s\n' k1 k2 | python scripts/r2_presign.py put --bucket B --expires 3600
  printf '%s\n' k1 k2 | python scripts/r2_presign.py get --bucket B --expires 3600
  python scripts/r2_presign.py put --bucket B --key single-key
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

REGION = "auto"
SERVICE = "s3"


def _load_env() -> dict[str, str]:
    values = {}
    keys = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    env_file = Path(__file__).resolve().parent.parent / ".env"
    file_vals: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                file_vals[k.strip()] = v.strip().strip('"').strip("'")
    for key in keys:
        values[key] = os.environ.get(key) or file_vals.get(key, "")
        if not values[key]:
            sys.exit(
                f"missing {key} — 在本地仓库根 .env 写入 R2_ACCOUNT_ID/"
                "R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY(已 gitignore,勿上 spark)"
            )
    return values


def presign(
    method: str,
    bucket: str,
    key: str,
    expires: int,
    creds: dict[str, str],
    now: datetime | None = None,
) -> str:
    host = f"{creds['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    now = now or datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.strftime("%Y%m%d")
    scope = f"{datestamp}/{REGION}/{SERVICE}/aws4_request"
    path = "/" + urllib.parse.quote(f"{bucket}/{key}", safe="/-_.~")
    query = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{creds['R2_ACCESS_KEY_ID']}/{scope}",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": str(expires),
        "X-Amz-SignedHeaders": "host",
    }
    canonical_query = "&".join(
        f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}"
        for k, v in sorted(query.items())
    )
    canonical_request = "\n".join(
        [
            method,
            path,
            canonical_query,
            f"host:{host}\n",
            "host",
            "UNSIGNED-PAYLOAD",
        ]
    )
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        ]
    )

    def _hmac(key_bytes: bytes, msg: str) -> bytes:
        return hmac.new(key_bytes, msg.encode(), hashlib.sha256).digest()

    signing_key = _hmac(
        _hmac(
            _hmac(
                _hmac(("AWS4" + creds["R2_SECRET_ACCESS_KEY"]).encode(), datestamp),
                REGION,
            ),
            SERVICE,
        ),
        "aws4_request",
    )
    signature = hmac.new(
        signing_key, string_to_sign.encode(), hashlib.sha256
    ).hexdigest()
    return (
        f"https://{host}{path}?{canonical_query}"
        f"&X-Amz-Signature={signature}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("method", choices=["put", "get", "delete"])
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--key", default=None, help="单 key;缺省从 stdin 逐行读")
    ap.add_argument("--expires", type=int, default=3600)
    args = ap.parse_args()
    creds = _load_env()
    keys = [args.key] if args.key else [
        line.strip() for line in sys.stdin if line.strip()
    ]
    if not keys:
        sys.exit("no keys given")
    now = datetime.now(timezone.utc)
    for key in keys:
        print(presign(args.method.upper(), args.bucket, key, args.expires, creds, now))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
