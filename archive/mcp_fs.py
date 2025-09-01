# mcp_fs.py
from mcp.server.fastmcp import FastMCP
from pathlib import Path
import os, shutil, glob
import sys, traceback, functools

mcp = FastMCP("kiro_full_fs")
ROOT = Path(__file__).resolve().parent

# ---------- 工具 ----------
def _inside_root(p: Path) -> Path:
    resolved = p.resolve()
    if not str(resolved).startswith(str(ROOT)):
        raise ValueError("Path outside allowed directories")
    return resolved

# ---------- 装饰器 ----------
def log_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise
    return wrapper

# ---------- 工具定义 ----------
@log_errors
@mcp.tool()
def read_file(path: str) -> str:
    return _inside_root(Path(path)).read_text(encoding="utf-8")

@log_errors
@mcp.tool()
def read_multiple_files(paths: list[str]) -> dict[str, str]:
    return {p: read_file(p) for p in paths}

@log_errors
@mcp.tool()
def write_file(path: str, content: str, create_dirs: bool = True) -> dict:
    p = _inside_root(Path(path))
    if create_dirs:
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"status": "success", "bytes": len(content)}

@log_errors
@mcp.tool()
def create_file(path: str, content: str = "") -> dict:
    p = _inside_root(Path(path))
    if p.exists():
        return {"status": "error", "message": "File already exists"}
    return write_file(path, content)

@log_errors
@mcp.tool()
def list_directory(path: str = ".") -> list[str]:
    return [str(item) for item in _inside_root(Path(path)).iterdir()]

@log_errors
@mcp.tool()
def create_directory(path: str, parents: bool = False) -> dict:
    _inside_root(Path(path)).mkdir(parents=parents, exist_ok=True)
    return {"status": "created", "path": str(_inside_root(Path(path)))}

@log_errors
@mcp.tool()
def delete_file(path: str) -> dict:
    _inside_root(Path(path)).unlink(missing_ok=True)
    return {"status": "deleted"}

@log_errors
@mcp.tool()
def delete_directory(path: str, recursive: bool = True) -> dict:
    p = _inside_root(Path(path))
    if recursive and p.is_dir():
        shutil.rmtree(p)
    else:
        p.rmdir()
    return {"status": "deleted"}

@log_errors
@mcp.tool()
def move_file(src: str, dst: str) -> dict:
    shutil.move(_inside_root(Path(src)), _inside_root(Path(dst)))
    return {"status": "moved"}

@log_errors
@mcp.tool()
def copy_file(src: str, dst: str) -> dict:
    shutil.copy2(_inside_root(Path(src)), _inside_root(Path(dst)))
    return {"status": "copied"}

@log_errors
@mcp.tool()
def get_file_info(path: str) -> dict:
    p = _inside_root(Path(path))
    stat = p.stat()
    return {
        "path": str(p),
        "size": stat.st_size,
        "is_dir": p.is_dir(),
        "modified": stat.st_mtime,
        "created": stat.st_ctime,
    }

@log_errors
@mcp.tool()
def search_files(
    path: str = ".", pattern: str = "*", max_results: int = 100
) -> list[str]:
    matches = glob.glob(pattern, root_dir=_inside_root(Path(path)), recursive=True)
    return matches[:max_results]

@log_errors
@mcp.tool()
def edit_file(
    path: str,
    operation: str = "replace",
    target=None,
    new_text: str = "",
) -> dict:
    try:
        p = _inside_root(Path(path))
        if not p.exists():
            return {"status": "error", "message": "File not found"}
        lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
        if operation == "append":
            lines.append(new_text if new_text.endswith("\n") else new_text + "\n")
        elif operation == "prepend":
            lines.insert(0, new_text if new_text.endswith("\n") else new_text + "\n")
        elif operation == "replace":
            if isinstance(target, int):
                if 1 <= target <= len(lines):
                    lines[target - 1] = new_text + ("\n" if not new_text.endswith("\n") else "")
                else:
                    return {"status": "error", "message": f"Line {target} out of range"}
            elif isinstance(target, str):
                text = "".join(lines)
                text = text.replace(target, new_text, 1)
                lines = [text] if text else []
        elif operation == "delete_lines":
            start = int(target) - 1
            n = int(new_text) if new_text else 1
            del lines[start:start + n]
        else:
            return {"status": "error", "message": "Unknown operation"}
        p.write_text("".join(lines), encoding="utf-8")
        return {"status": "success", "lines": len(lines)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ---------- 入口 ----------
if __name__ == "__main__":
    mcp.run()