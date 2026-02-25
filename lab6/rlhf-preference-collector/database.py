import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from supabase import Client, create_client
except ModuleNotFoundError:
    Client = Any  # type: ignore[assignment]
    create_client = None  # type: ignore[assignment]


class DatabaseManager:
    def __init__(
        self,
        supabase_url: Optional[str],
        supabase_key: Optional[str],
        local_file: Path,
    ):
        self.local_file = local_file
        self.local_file.parent.mkdir(parents=True, exist_ok=True)

        self.supabase_client: Optional[Client] = None
        self.supabase_ready = False

        if supabase_url and supabase_key and create_client is not None:
            try:
                self.supabase_client = create_client(supabase_url, supabase_key)
                # Lightweight check to validate credentials/connectivity.
                self.supabase_client.table("preference_data").select("id").limit(1).execute()
                self.supabase_ready = True
            except Exception:
                self.supabase_client = None
                self.supabase_ready = False

    def insert_record(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        if self.supabase_ready and self.supabase_client is not None:
            try:
                self.supabase_client.table("preference_data").insert(record).execute()
                return True, "Saved to Supabase."
            except Exception as exc:
                self._append_local(record)
                return (
                    True,
                    f"Supabase insert failed ({exc}). Saved locally instead.",
                )

        self._append_local(record)
        return True, "Saved locally (Supabase not configured or unavailable)."

    def get_all_records(self) -> List[Dict[str, Any]]:
        if self.supabase_ready and self.supabase_client is not None:
            try:
                response = self.supabase_client.table("preference_data").select("*").execute()
                data = response.data or []
                if isinstance(data, list):
                    return data
                return []
            except Exception:
                return self._read_local()

        return self._read_local()

    def _append_local(self, record: Dict[str, Any]) -> None:
        with self.local_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _read_local(self) -> List[Dict[str, Any]]:
        if not self.local_file.exists():
            return []

        rows: List[Dict[str, Any]] = []
        with self.local_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows
