"""Update an existing installer archive with this checkout's release sources."""
import argparse
from pathlib import Path
import zipfile


def package_release(base_zip, output):
    root = Path(__file__).resolve().parents[1]
    base_zip, output = Path(base_zip).resolve(), Path(output).resolve()
    if base_zip == output:
        raise ValueError("Keep the base release archive; choose a different output path.")
    replacements = {p.name: p for p in (root / "installers").iterdir() if p.suffix in (".bat", ".sh")}
    replacements["requirements.txt"] = root / "requirements.txt"
    with zipfile.ZipFile(base_zip) as source:
        missing = replacements.keys() - set(source.namelist())
        if missing:
            raise ValueError(f"Base release is missing installer files: {sorted(missing)}")
        # Exclusive creation preserves any previously built artifact.
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as target:
            for info in source.infolist():
                data = replacements[info.filename].read_bytes() if info.filename in replacements else source.read(info)
                target.writestr(info, data)
    with zipfile.ZipFile(output) as result:
        if result.testzip() is not None:
            raise ValueError("Release archive failed its CRC check.")
        for name, path in replacements.items():
            if result.read(name) != path.read_bytes():
                raise ValueError(f"Packaged file does not match source: {name}")
    print(f"Verified release archive: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-zip", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    package_release(args.base_zip, args.output)
