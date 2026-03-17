#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hmac
import json
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_REQUEST_SIZE = MAX_FILE_SIZE + 1024 * 1024
MAX_PREDICT_REQUEST_SIZE = MAX_FILE_SIZE + 1024 * 1024
MAX_AUTH_PAYLOAD_SIZE = 16 * 1024
MAX_TRAIN_PAYLOAD_SIZE = 32 * 1024
MAX_RECORD_OP_PAYLOAD_SIZE = 32 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
ADMIN_PASSWORD = "ustcl00AnaExpt"
TRAINING_COOLDOWN_SECONDS = 20
HEIF_BRANDS = {
    b"heic",
    b"heix",
    b"hevc",
    b"hevx",
    b"heim",
    b"heis",
    b"heif",
    b"hefs",
    b"mif1",
    b"msf1",
}
TRAINING_FILE_HEADER = "H,S,V,R,G,B,pH"
TRAINING_DATA_LOCK = Lock()
TRAINING_RUN_LOCK = Lock()


def json_response(handler: SimpleHTTPRequestHandler, status: int, payload: dict) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def text_response(
    handler: SimpleHTTPRequestHandler,
    status: int,
    payload: str,
    extra_headers: dict[str, str] | None = None,
) -> None:
    raw = payload.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    if extra_headers:
        for key, value in extra_headers.items():
            handler.send_header(key, value)
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def sanitize_filename(filename: str) -> str:
    basename = Path(filename).name
    basename = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    return basename or "upload"


def detect_image_type(content: bytes) -> str | None:
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return ".webp"
    if len(content) >= 12 and content[4:8] == b"ftyp":
        brand = content[8:12]
        if brand in HEIF_BRANDS:
            if brand.startswith(b"hei"):
                return ".heic"
            return ".heif"
    return None


def parse_text_part(part) -> str:
    raw = part.get_payload(decode=True) or b""
    charset = part.get_content_charset() or "utf-8"
    try:
        return raw.decode(charset, errors="replace").strip()
    except LookupError:
        return raw.decode("utf-8", errors="replace").strip()


def normalize_training_key(
    h: float,
    s: float,
    v: float,
    r: float,
    g: float,
    b: float,
    ph: float,
) -> tuple[str, str, str, str, str, str, str]:
    values = [f"{value:.4f}" for value in (h, s, v, r, g, b, ph)]
    return values[0], values[1], values[2], values[3], values[4], values[5], values[6]


def append_unique_training_data(
    file_path: Path,
    h: float,
    s: float,
    v: float,
    r: float,
    g: float,
    b: float,
    ph: float,
) -> bool:
    key = normalize_training_key(h, s, v, r, g, b, ph)
    existing_keys: set[tuple[str, str, str, str, str, str, str]] = set()

    if file_path.exists():
        raw_lines = file_path.read_text(encoding="utf-8").splitlines()
        non_empty_lines = [line.strip() for line in raw_lines if line.strip()]

        # 不兼容旧格式：如果文件头不是新格式，直接重置为新表头。
        if non_empty_lines:
            first_line = non_empty_lines[0].lower().replace(" ", "")
            expected = TRAINING_FILE_HEADER.lower().replace(" ", "")
            if first_line != expected:
                file_path.write_text(TRAINING_FILE_HEADER + "\n", encoding="utf-8")
                raw_lines = [TRAINING_FILE_HEADER]

        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            if line.lower().replace(" ", "") == TRAINING_FILE_HEADER.lower():
                continue

            parts = [piece.strip() for piece in line.split(",")]
            if len(parts) != 7:
                continue
            try:
                existing_keys.add(
                    normalize_training_key(
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                    )
                )
            except ValueError:
                continue

    if key in existing_keys:
        return False

    needs_header = (not file_path.exists()) or file_path.stat().st_size == 0
    with file_path.open("a", encoding="utf-8") as fp:
        if needs_header:
            fp.write(TRAINING_FILE_HEADER + "\n")
        fp.write(",".join(key) + "\n")

    return True


def run_comp_vision(
    comp_vision_path: Path,
    image_path: Path,
) -> tuple[float, float, float, float, float, float, int]:
    proc = subprocess.run(
        [sys.executable, str(comp_vision_path), str(image_path), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "comp_vision 执行失败"
        raise RuntimeError(detail)

    output_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("comp_vision 未输出结果")

    try:
        payload = json.loads(output_lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("comp_vision 输出格式错误") from exc

    try:
        h = float(payload["h"])
        s = float(payload["s"])
        v = float(payload["v"])
        r = float(payload["r"])
        g = float(payload["g"])
        b = float(payload["b"])
        picked_count = int(payload.get("picked_count", 0))
    except (TypeError, ValueError, KeyError) as exc:
        raise RuntimeError("comp_vision 缺少有效 HSV/RGB 字段") from exc

    return h, s, v, r, g, b, picked_count


def read_timestamp_file(file_path: Path) -> datetime | None:
    if not file_path.exists():
        return None

    raw = file_path.read_text(encoding="utf-8").strip()
    if not raw:
        return None

    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def write_timestamp_file(file_path: Path, value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    text = value.astimezone(timezone.utc).isoformat()
    file_path.write_text(text, encoding="utf-8")
    return text


def parse_training_entries(file_path: Path) -> list[tuple[float, float, float, float, float, float, float]]:
    entries: list[tuple[float, float, float, float, float, float, float]] = []
    if not file_path.exists():
        return entries

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().replace(" ", "") == TRAINING_FILE_HEADER.lower():
            continue

        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            h = float(parts[0])
            s = float(parts[1])
            v = float(parts[2])
            r = float(parts[3])
            g = float(parts[4])
            b = float(parts[5])
            ph = float(parts[6])
        except ValueError:
            continue
        entries.append((h, s, v, r, g, b, ph))

    return entries


def write_training_entries(file_path: Path, entries: list[tuple[float, float, float, float, float, float, float]]) -> None:
    lines = [TRAINING_FILE_HEADER]
    for h, s, v, r, g, b, ph in entries:
        lines.append(f"{h:.4f},{s:.4f},{v:.4f},{r:.4f},{g:.4f},{b:.4f},{ph:.4f}")
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_feature_name(value: str) -> str:
    feature = value.strip().lower()
    if feature not in {"hsv", "rgb"}:
        raise ValueError("feature 只能是 hsv 或 rgb")
    return feature


def list_latest_models(
    model_output_directory: Path,
    limit: int = 10,
    feature: str | None = None,
) -> list[dict[str, str]]:
    if not model_output_directory.exists():
        return []

    model_paths = [path for path in model_output_directory.glob("*.pkl") if path.is_file()]
    if feature is not None:
        feature_prefix = f"model-{feature.lower()}-"
        model_paths = [path for path in model_paths if path.name.lower().startswith(feature_prefix)]
    model_paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    rows: list[dict[str, str]] = []
    for model_path in model_paths[:limit]:
        modified_at = datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc).isoformat()
        rows.append(
            {
                "name": model_path.name,
                "modifiedAt": modified_at,
                "feature": "rgb" if "-rgb-" in model_path.name.lower() else "hsv",
            }
        )
    return rows


def resolve_model_path(model_output_directory: Path, model_name: str) -> Path:
    safe_name = Path(model_name).name
    if safe_name != model_name:
        raise ValueError("模型名不合法")
    if not safe_name.lower().endswith(".pkl"):
        raise ValueError("模型文件必须是 .pkl")

    model_path = model_output_directory / safe_name
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError("模型不存在")
    return model_path


def run_use_predict(
    use_script_path: Path,
    model_path: Path,
    feature: str,
    h: float,
    s: float,
    v: float,
    r: float,
    g: float,
    b: float,
) -> float:
    proc = subprocess.run(
        [
            sys.executable,
            str(use_script_path),
            "--model",
            str(model_path),
            "--feature",
            feature,
            "--h",
            str(h),
            "--s",
            str(s),
            "--v",
            str(v),
            "--r",
            str(r),
            "--g",
            str(g),
            "--b",
            str(b),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "use.py 执行失败"
        raise RuntimeError(detail)

    output_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("use.py 未输出结果")

    try:
        payload = json.loads(output_lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("use.py 输出格式错误") from exc

    try:
        predicted_ph = float(payload["ph"])
    except (TypeError, ValueError, KeyError) as exc:
        raise RuntimeError("use.py 缺少有效 ph 字段") from exc

    return predicted_ph


class UploadHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args,
        directory: str,
        upload_directory: Path,
        training_data_path: Path,
        comp_vision_path: Path,
        train_script_path: Path,
        use_script_path: Path,
        model_output_directory: Path,
        predict_buffer_directory: Path,
        manual_refresh_timestamp_path: Path,
        last_training_timestamp_path: Path,
        **kwargs,
    ):
        self.upload_directory = upload_directory
        self.training_data_path = training_data_path
        self.comp_vision_path = comp_vision_path
        self.train_script_path = train_script_path
        self.use_script_path = use_script_path
        self.model_output_directory = model_output_directory
        self.predict_buffer_directory = predict_buffer_directory
        self.manual_refresh_timestamp_path = manual_refresh_timestamp_path
        self.last_training_timestamp_path = last_training_timestamp_path
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/training-data":
            self.handle_training_data(parse_qs(parsed.query))
            return
        if parsed.path == "/api/models/latest":
            self.handle_latest_models(parse_qs(parsed.query))
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/upload":
            self.handle_upload()
            return
        if path == "/api/predict-ph":
            self.handle_predict_ph()
            return
        if path == "/api/verify-password":
            self.handle_verify_password()
            return
        if path == "/api/train-model":
            self.handle_train_model()
            return
        if path == "/api/training-records/list":
            self.handle_list_training_records()
            return
        if path == "/api/training-records/delete":
            self.handle_delete_training_record()
            return
        if path == "/api/training-records/clear":
            self.handle_clear_training_records()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def read_json_payload(self, max_size: int) -> dict[str, object] | None:
        raw_length = self.headers.get("Content-Length", "")
        try:
            content_length = int(raw_length)
        except ValueError:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "缺少合法 Content-Length"})
            return None

        if content_length <= 0:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体为空"})
            return None
        if content_length > max_size:
            json_response(self, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "请求体过大"})
            return None

        body = self.rfile.read(content_length)
        if len(body) != content_length:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体读取不完整"})
            return None

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体必须是 JSON"})
            return None

        if not isinstance(payload, dict):
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体格式错误"})
            return None

        return payload

    def extract_password(self, payload: dict[str, object]) -> str | None:
        password = payload.get("password")
        if not isinstance(password, str):
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "password 字段必须是字符串"})
            return None
        return password

    def handle_latest_models(self, query_params: dict[str, list[str]]) -> None:
        feature_raw = query_params.get("feature", [""])[0]
        feature_filter: str | None = None
        if feature_raw.strip():
            try:
                feature_filter = normalize_feature_name(feature_raw)
            except ValueError as exc:
                json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

        models = list_latest_models(self.model_output_directory, limit=10, feature=feature_filter)
        json_response(
            self,
            HTTPStatus.OK,
            {
                "ok": True,
                "models": models,
            },
        )

    def handle_training_data(self, query_params: dict[str, list[str]]) -> None:
        is_manual_refresh = False
        for value in query_params.get("manual", []):
            if value.strip().lower() in {"1", "true", "yes", "y"}:
                is_manual_refresh = True
                break

        manual_refresh_at = ""
        if is_manual_refresh:
            manual_refresh_at = write_timestamp_file(
                self.manual_refresh_timestamp_path,
                datetime.now(timezone.utc),
            )
        else:
            previous = read_timestamp_file(self.manual_refresh_timestamp_path)
            if previous is not None:
                manual_refresh_at = previous.isoformat()

        with TRAINING_DATA_LOCK:
            if not self.training_data_path.exists():
                headers = {"X-Manual-Refresh-At": manual_refresh_at} if manual_refresh_at else None
                text_response(self, HTTPStatus.OK, "", extra_headers=headers)
                return
            content = self.training_data_path.read_text(encoding="utf-8")
        headers = {"X-Manual-Refresh-At": manual_refresh_at} if manual_refresh_at else None
        text_response(self, HTTPStatus.OK, content, extra_headers=headers)

    def handle_predict_ph(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            json_response(
                self,
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "请求必须为 multipart/form-data"},
            )
            return

        raw_length = self.headers.get("Content-Length", "")
        try:
            content_length = int(raw_length)
        except ValueError:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "缺少合法 Content-Length"})
            return

        if content_length <= 0:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体为空"})
            return
        if content_length > MAX_PREDICT_REQUEST_SIZE:
            json_response(
                self,
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "请求体过大，单文件不能超过 5MB"},
            )
            return

        body = self.rfile.read(content_length)
        if len(body) != content_length:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体读取不完整"})
            return

        fake_header = (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n"
            "\r\n"
        ).encode("utf-8")
        message = BytesParser(policy=default).parsebytes(fake_header + body)
        if not message.is_multipart():
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "上传数据格式错误"})
            return

        file_part = None
        model_raw = ""
        feature_raw = ""
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            field_name = part.get_param("name", header="content-disposition")
            if field_name == "image" and part.get_filename():
                file_part = part
            if field_name == "model":
                model_raw = parse_text_part(part)
            if field_name == "feature":
                feature_raw = parse_text_part(part)

        if file_part is None:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "未找到 image 文件字段"})
            return
        if not model_raw:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "未指定模型"})
            return
        if not feature_raw:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "未指定 feature"})
            return

        try:
            feature = normalize_feature_name(feature_raw)
        except ValueError as exc:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        try:
            model_path = resolve_model_path(self.model_output_directory, model_raw)
        except ValueError as exc:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except FileNotFoundError as exc:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        expected_prefix = f"model-{feature}-"
        if not model_path.name.lower().startswith(expected_prefix):
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "所选模型与特征类型不匹配"})
            return

        original_name = sanitize_filename(file_part.get_filename() or "")
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            json_response(
                self,
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "仅支持 jpg/jpeg/png/heic/heif/webp"},
            )
            return

        content = file_part.get_payload(decode=True) or b""
        if not content:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "文件为空"})
            return
        if len(content) > MAX_FILE_SIZE:
            json_response(
                self,
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "文件超过 5MB 限制"},
            )
            return

        detected = detect_image_type(content)
        if detected is None:
            json_response(self, HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "文件内容不是有效图片"})
            return

        if suffix in {".jpg", ".jpeg"} and detected != ".jpg":
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "文件扩展名与内容不匹配"})
            return
        if suffix == ".png" and detected != ".png":
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "文件扩展名与内容不匹配"})
            return
        if suffix == ".webp" and detected != ".webp":
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "文件扩展名与内容不匹配"})
            return
        if suffix in {".heic", ".heif"} and detected not in {".heic", ".heif"}:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "文件扩展名与内容不匹配"})
            return

        self.predict_buffer_directory.mkdir(parents=True, exist_ok=True)
        temp_name = f"predict-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}{suffix}"
        buffer_path = self.predict_buffer_directory / temp_name
        buffer_path.write_bytes(content)

        try:
            h, s, v, r, g, b, picked_count = run_comp_vision(self.comp_vision_path, buffer_path)
            predicted_ph = run_use_predict(
                self.use_script_path,
                model_path,
                feature,
                h,
                s,
                v,
                r,
                g,
                b,
            )
        except Exception as exc:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": f"预测失败: {exc}"})
            return
        finally:
            try:
                buffer_path.unlink(missing_ok=True)
            except OSError:
                pass

        json_response(
            self,
            HTTPStatus.OK,
            {
                "ok": True,
                "model": model_path.name,
                "feature": feature,
                "h": round(h, 4),
                "s": round(s, 4),
                "v": round(v, 4),
                "r": round(r, 4),
                "g": round(g, 4),
                "b": round(b, 4),
                "pickedCount": picked_count,
                "predictedPh": round(predicted_ph, 4),
            },
        )

    def handle_list_training_records(self) -> None:
        payload = self.read_json_payload(MAX_RECORD_OP_PAYLOAD_SIZE)
        if payload is None:
            return

        password = self.extract_password(payload)
        if password is None:
            return
        if not hmac.compare_digest(password, ADMIN_PASSWORD):
            json_response(self, HTTPStatus.FORBIDDEN, {"error": "密码错误"})
            return

        with TRAINING_DATA_LOCK:
            entries = parse_training_entries(self.training_data_path)

        rows = [
            {
                "id": idx,
                "h": round(entry[0], 4),
                "s": round(entry[1], 4),
                "v": round(entry[2], 4),
                "r": round(entry[3], 4),
                "g": round(entry[4], 4),
                "b": round(entry[5], 4),
                "ph": round(entry[6], 4),
                "line": (
                    f"{entry[0]:.4f},{entry[1]:.4f},{entry[2]:.4f},"
                    f"{entry[3]:.4f},{entry[4]:.4f},{entry[5]:.4f},{entry[6]:.4f}"
                ),
            }
            for idx, entry in enumerate(entries, start=1)
        ]

        json_response(self, HTTPStatus.OK, {"ok": True, "rows": rows, "count": len(rows)})

    def handle_delete_training_record(self) -> None:
        payload = self.read_json_payload(MAX_RECORD_OP_PAYLOAD_SIZE)
        if payload is None:
            return

        password = self.extract_password(payload)
        if password is None:
            return
        if not hmac.compare_digest(password, ADMIN_PASSWORD):
            json_response(self, HTTPStatus.FORBIDDEN, {"error": "密码错误"})
            return

        record_id_raw = payload.get("id")
        try:
            record_id = int(record_id_raw)
        except (TypeError, ValueError):
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "id 必须是整数"})
            return
        if record_id <= 0:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "id 必须大于 0"})
            return

        with TRAINING_DATA_LOCK:
            entries = parse_training_entries(self.training_data_path)
            if not entries:
                json_response(self, HTTPStatus.BAD_REQUEST, {"error": "没有可删除的数据"})
                return
            if record_id > len(entries):
                json_response(self, HTTPStatus.BAD_REQUEST, {"error": "id 超出范围"})
                return

            deleted = entries.pop(record_id - 1)
            write_training_entries(self.training_data_path, entries)

        json_response(
            self,
            HTTPStatus.OK,
            {
                "ok": True,
                "deleted": {
                    "h": round(deleted[0], 4),
                    "s": round(deleted[1], 4),
                    "v": round(deleted[2], 4),
                    "r": round(deleted[3], 4),
                    "g": round(deleted[4], 4),
                    "b": round(deleted[5], 4),
                    "ph": round(deleted[6], 4),
                    "line": (
                        f"{deleted[0]:.4f},{deleted[1]:.4f},{deleted[2]:.4f},"
                        f"{deleted[3]:.4f},{deleted[4]:.4f},{deleted[5]:.4f},{deleted[6]:.4f}"
                    ),
                },
                "remaining": len(entries),
            },
        )

    def handle_clear_training_records(self) -> None:
        payload = self.read_json_payload(MAX_RECORD_OP_PAYLOAD_SIZE)
        if payload is None:
            return

        password = self.extract_password(payload)
        if password is None:
            return
        if not hmac.compare_digest(password, ADMIN_PASSWORD):
            json_response(self, HTTPStatus.FORBIDDEN, {"error": "密码错误"})
            return

        with TRAINING_DATA_LOCK:
            previous_count = len(parse_training_entries(self.training_data_path))
            write_training_entries(self.training_data_path, [])

        json_response(
            self,
            HTTPStatus.OK,
            {
                "ok": True,
                "cleared": previous_count,
                "message": "training_data.dat 已清空",
            },
        )

    def handle_verify_password(self) -> None:
        raw_length = self.headers.get("Content-Length", "")
        try:
            content_length = int(raw_length)
        except ValueError:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "缺少合法 Content-Length"})
            return

        if content_length <= 0:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体为空"})
            return
        if content_length > MAX_AUTH_PAYLOAD_SIZE:
            json_response(self, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "请求体过大"})
            return

        body = self.rfile.read(content_length)
        if len(body) != content_length:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体读取不完整"})
            return

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体必须是 JSON"})
            return

        if not isinstance(payload, dict):
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体格式错误"})
            return

        password = payload.get("password")
        if not isinstance(password, str):
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "password 字段必须是字符串"})
            return

        ok = hmac.compare_digest(password, ADMIN_PASSWORD)
        last_training_at = read_timestamp_file(self.last_training_timestamp_path)
        json_response(
            self,
            HTTPStatus.OK,
            {
                "ok": ok,
                "lastTrainingAt": last_training_at.isoformat() if last_training_at else "",
                "cooldownSeconds": TRAINING_COOLDOWN_SECONDS,
            },
        )

    def handle_train_model(self) -> None:
        raw_length = self.headers.get("Content-Length", "")
        try:
            content_length = int(raw_length)
        except ValueError:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "缺少合法 Content-Length"})
            return

        if content_length <= 0:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体为空"})
            return
        if content_length > MAX_TRAIN_PAYLOAD_SIZE:
            json_response(self, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "请求体过大"})
            return

        body = self.rfile.read(content_length)
        if len(body) != content_length:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体读取不完整"})
            return

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体必须是 JSON"})
            return

        if not isinstance(payload, dict):
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体格式错误"})
            return

        password = payload.get("password")
        if not isinstance(password, str):
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "password 字段必须是字符串"})
            return

        if not hmac.compare_digest(password, ADMIN_PASSWORD):
            json_response(self, HTTPStatus.FORBIDDEN, {"error": "密码错误"})
            return

        auto_tune = payload.get("autoTune", True)
        if not isinstance(auto_tune, bool):
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "autoTune 字段必须是布尔值"})
            return

        c_value = 10.0
        epsilon_value = 0.05
        gamma_value = "scale"
        if not auto_tune:
            try:
                c_value = float(payload.get("c"))
                epsilon_value = float(payload.get("epsilon"))
            except (TypeError, ValueError):
                json_response(self, HTTPStatus.BAD_REQUEST, {"error": "c 和 epsilon 必须为数字"})
                return

            if c_value <= 0:
                json_response(self, HTTPStatus.BAD_REQUEST, {"error": "c 必须大于 0"})
                return
            if epsilon_value < 0:
                json_response(self, HTTPStatus.BAD_REQUEST, {"error": "epsilon 不能为负数"})
                return

            gamma_raw = payload.get("gamma")
            if gamma_raw is None:
                json_response(self, HTTPStatus.BAD_REQUEST, {"error": "gamma 不能为空"})
                return
            gamma_text = str(gamma_raw).strip().lower()
            if not gamma_text:
                json_response(self, HTTPStatus.BAD_REQUEST, {"error": "gamma 不能为空"})
                return

            if gamma_text not in {"scale", "auto"}:
                try:
                    float(gamma_text)
                except ValueError:
                    json_response(
                        self,
                        HTTPStatus.BAD_REQUEST,
                        {"error": "gamma 需为 scale、auto 或合法数字"},
                    )
                    return
            gamma_value = gamma_text

        if not self.training_data_path.exists():
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "training_data.dat 不存在"})
            return

        with TRAINING_DATA_LOCK:
            sample_entries = parse_training_entries(self.training_data_path)
        if not sample_entries:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "training_data.dat 中没有可训练样本"})
            return

        with TRAINING_RUN_LOCK:
            now_utc = datetime.now(timezone.utc)
            last_training = read_timestamp_file(self.last_training_timestamp_path)
            if last_training is not None:
                elapsed = (now_utc - last_training).total_seconds()
                if elapsed < TRAINING_COOLDOWN_SECONDS:
                    wait_seconds = int(TRAINING_COOLDOWN_SECONDS - elapsed + 0.999)
                    json_response(
                        self,
                        HTTPStatus.TOO_MANY_REQUESTS,
                        {
                            "error": f"两次训练间隔不得小于 {TRAINING_COOLDOWN_SECONDS}s",
                            "lastTrainingAt": last_training.isoformat(),
                            "waitSeconds": wait_seconds,
                        },
                    )
                    return

            self.model_output_directory.mkdir(parents=True, exist_ok=True)
            model_stamp = now_utc.strftime("%Y%m%d-%H%M%S")

            feature_to_model_path: dict[str, Path] = {}
            for feature in ("hsv", "rgb"):
                candidate = self.model_output_directory / f"model-{feature}-{model_stamp}.pkl"
                if candidate.exists():
                    candidate = self.model_output_directory / f"model-{feature}-{model_stamp}-{uuid.uuid4().hex[:6]}.pkl"
                feature_to_model_path[feature] = candidate

            feature_outputs: dict[str, str] = {}
            for feature in ("hsv", "rgb"):
                model_path = feature_to_model_path[feature]
                command = [
                    sys.executable,
                    str(self.train_script_path),
                    "--data",
                    str(self.training_data_path),
                    "--feature",
                    feature,
                    "--save",
                    str(model_path),
                ]
                if auto_tune:
                    command.append("--auto-tune")
                else:
                    command.extend(
                        [
                            "--c",
                            str(c_value),
                            "--epsilon",
                            str(epsilon_value),
                            "--gamma",
                            gamma_value,
                        ]
                    )

                proc = subprocess.run(command, capture_output=True, text=True, check=False)
                if proc.returncode != 0:
                    detail = proc.stderr.strip() or proc.stdout.strip() or f"{feature} 训练失败"
                    json_response(
                        self,
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {
                            "error": f"{feature.upper()} 训练失败: {detail}",
                            "feature": feature,
                        },
                    )
                    return

                stdout = (proc.stdout or "").strip()
                feature_outputs[feature] = "\n".join(stdout.splitlines()[-20:]) if stdout else ""

            trained_at = write_timestamp_file(self.last_training_timestamp_path, now_utc)

        json_response(
            self,
            HTTPStatus.OK,
            {
                "ok": True,
                "trainedAt": trained_at,
                "autoTune": auto_tune,
                "results": {
                    "hsv": {
                        "modelFile": feature_to_model_path["hsv"].name,
                        "modelPath": f"{self.model_output_directory.name}/{feature_to_model_path['hsv'].name}",
                        "output": feature_outputs.get("hsv", ""),
                    },
                    "rgb": {
                        "modelFile": feature_to_model_path["rgb"].name,
                        "modelPath": f"{self.model_output_directory.name}/{feature_to_model_path['rgb'].name}",
                        "output": feature_outputs.get("rgb", ""),
                    },
                },
            },
        )

    def handle_upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            json_response(
                self,
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "请求必须为 multipart/form-data"},
            )
            return

        raw_length = self.headers.get("Content-Length", "")
        try:
            content_length = int(raw_length)
        except ValueError:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "缺少合法 Content-Length"})
            return

        if content_length <= 0:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体为空"})
            return
        if content_length > MAX_REQUEST_SIZE:
            json_response(
                self,
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "请求体过大，单文件不能超过 5MB"},
            )
            return

        body = self.rfile.read(content_length)
        if len(body) != content_length:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体读取不完整"})
            return

        fake_header = (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n"
            "\r\n"
        ).encode("utf-8")
        message = BytesParser(policy=default).parsebytes(fake_header + body)
        if not message.is_multipart():
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "上传数据格式错误"})
            return

        file_part = None
        ph_raw = ""
        password_raw = ""
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            field_name = part.get_param("name", header="content-disposition")
            if field_name == "image" and part.get_filename():
                file_part = part
            if field_name == "ph":
                ph_raw = parse_text_part(part)
            if field_name == "password":
                password_raw = parse_text_part(part)

        if file_part is None:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "未找到 image 文件字段"})
            return

        if not ph_raw:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "未找到 ph 字段或 ph 为空"})
            return

        if not password_raw:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "未找到 password 字段或 password 为空"})
            return

        if not hmac.compare_digest(password_raw, ADMIN_PASSWORD):
            json_response(self, HTTPStatus.FORBIDDEN, {"error": "上传密码错误"})
            return

        try:
            ph_value = float(ph_raw)
        except ValueError:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "ph 必须是数字"})
            return

        if not 0.0 <= ph_value <= 14.0:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "ph 范围应在 0 到 14 之间"})
            return

        original_name = sanitize_filename(file_part.get_filename() or "")
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            json_response(
                self,
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "仅支持 jpg/jpeg/png/heic/heif/webp"},
            )
            return

        content = file_part.get_payload(decode=True) or b""
        if not content:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "文件为空"})
            return
        if len(content) > MAX_FILE_SIZE:
            json_response(
                self,
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "文件超过 5MB 限制"},
            )
            return

        detected = detect_image_type(content)
        if detected is None:
            json_response(self, HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "文件内容不是有效图片"})
            return

        if suffix in {".jpg", ".jpeg"} and detected != ".jpg":
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "文件扩展名与内容不匹配"})
            return
        if suffix == ".png" and detected != ".png":
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "文件扩展名与内容不匹配"})
            return
        if suffix == ".webp" and detected != ".webp":
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "文件扩展名与内容不匹配"})
            return
        if suffix in {".heic", ".heif"} and detected not in {".heic", ".heif"}:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "文件扩展名与内容不匹配"})
            return

        stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        unique_name = f"{stamp}-{uuid.uuid4().hex[:8]}{suffix}"
        target = self.upload_directory / unique_name
        target.write_bytes(content)

        try:
            h, s, v, r, g, b, picked_count = run_comp_vision(self.comp_vision_path, target)
        except Exception as exc:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": f"图片分析失败: {exc}"})
            return

        with TRAINING_DATA_LOCK:
            inserted = append_unique_training_data(
                self.training_data_path,
                h=h,
                s=s,
                v=v,
                r=r,
                g=g,
                b=b,
                ph=ph_value,
            )

        json_response(
            self,
            HTTPStatus.OK,
            {
                "ok": True,
                "originalName": original_name,
                "filename": unique_name,
                "size": len(content),
                "url": f"/uploads/{unique_name}",
                "training": {
                    "inserted": inserted,
                    "h": round(h, 4),
                    "s": round(s, 4),
                    "v": round(v, 4),
                    "r": round(r, 4),
                    "g": round(g, 4),
                    "b": round(b, 4),
                    "ph": round(ph_value, 4),
                    "pickedCount": picked_count,
                    "dataFile": self.training_data_path.name,
                },
            },
        )


def make_handler(
    web_root: Path,
    upload_directory: Path,
    training_data_path: Path,
    comp_vision_path: Path,
    train_script_path: Path,
    use_script_path: Path,
    model_output_directory: Path,
    predict_buffer_directory: Path,
    manual_refresh_timestamp_path: Path,
    last_training_timestamp_path: Path,
):
    def factory(*args, **kwargs):
        return UploadHandler(
            *args,
            directory=str(web_root),
            upload_directory=upload_directory,
            training_data_path=training_data_path,
            comp_vision_path=comp_vision_path,
            train_script_path=train_script_path,
            use_script_path=use_script_path,
            model_output_directory=model_output_directory,
            predict_buffer_directory=predict_buffer_directory,
            manual_refresh_timestamp_path=manual_refresh_timestamp_path,
            last_training_timestamp_path=last_training_timestamp_path,
            **kwargs,
        )

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple image upload web service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument(
        "--web-root",
        default=str(Path(__file__).resolve().parent),
        help="Directory to serve static files",
    )
    args = parser.parse_args()

    web_root = Path(args.web_root).resolve()
    upload_directory = web_root / "uploads"
    training_data_path = web_root / "training_data.dat"
    comp_vision_path = web_root / "comp_vision.py"
    train_script_path = web_root / "train.py"
    use_script_path = web_root / "use.py"
    model_output_directory = web_root / "models"
    predict_buffer_directory = web_root / ".predict_buffer"
    manual_refresh_timestamp_path = web_root / ".manual_refresh.timestamp"
    last_training_timestamp_path = web_root / ".last_training.timestamp"
    upload_directory.mkdir(parents=True, exist_ok=True)
    model_output_directory.mkdir(parents=True, exist_ok=True)
    predict_buffer_directory.mkdir(parents=True, exist_ok=True)

    if not comp_vision_path.exists():
        raise FileNotFoundError(f"comp_vision.py 不存在: {comp_vision_path}")
    if not train_script_path.exists():
        raise FileNotFoundError(f"train.py 不存在: {train_script_path}")
    if not use_script_path.exists():
        raise FileNotFoundError(f"use.py 不存在: {use_script_path}")

    handler = make_handler(
        web_root,
        upload_directory,
        training_data_path,
        comp_vision_path,
        train_script_path,
        use_script_path,
        model_output_directory,
        predict_buffer_directory,
        manual_refresh_timestamp_path,
        last_training_timestamp_path,
    )
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {web_root} on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()