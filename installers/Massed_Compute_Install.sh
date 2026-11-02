#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! cd "$SCRIPT_DIR"; then
    echo "Failed to enter the installer directory. Exiting..."
    exit 1
fi

pip install requests
pip install tqdm
sudo apt update
sudo apt install software-properties-common --yes
sudo apt install git-lfs --yes
sudo apt install xz-utils curl --yes
sudo apt install zram-config --yes
git lfs install
df -h /dev/shm

FFMPEG_URL="https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/ffmpeg-n9.0-latest-linux64-gpl-9.0.tar.xz"
echo "Installing FFmpeg n9.0 as the system default..."
FFMPEG_TMP="$(mktemp -d)"
if ! curl -fL -o "$FFMPEG_TMP/ffmpeg.tar.xz" "$FFMPEG_URL" \
    || ! tar -xJf "$FFMPEG_TMP/ffmpeg.tar.xz" -C "$FFMPEG_TMP" \
    || ! sudo install -m 755 "$FFMPEG_TMP"/ffmpeg-*/bin/ffmpeg "$FFMPEG_TMP"/ffmpeg-*/bin/ffprobe /usr/local/bin/ \
    || ! sudo ln -sf /usr/local/bin/ffmpeg /usr/local/bin/ffprobe /usr/bin/; then
    echo "Failed to install FFmpeg. Exiting..."
    exit 1
fi
rm -rf "$FFMPEG_TMP"
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo "FFmpeg or ffprobe installation could not be verified. Exiting..."
    exit 1
fi
echo "Using $(ffmpeg -version | head -n 1)"

export UV_SKIP_WHEEL_FILENAME_CHECK=1
export UV_LINK_MODE=copy
# Compile the libraries to bytecode during install; otherwise the first app start does it and looks frozen
export UV_COMPILE_BYTECODE=1

git clone https://github.com/FurkanGozukara/Wan2.1 Wan2.1

cd Wan2.1 || exit 1

git remote set-url origin https://github.com/FurkanGozukara/Wan2.1

git reset --hard

git pull || exit 1

REQUIRED_PY_VERSION="3.12.13"
REQUIRED_PY_TUPLE="(3, 12, 13)"

VENV_DIR="venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_READY=0
CAN_INSTALL_SYSTEM_PACKAGES=1
SUDO=()

python_is_current() {
    [ -n "$1" ] || return 1
    "$1" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.version_info.releaselevel == 'final' and sys.version_info[:3] >= $REQUIRED_PY_TUPLE else 1)" 2>/dev/null
}

if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO=(sudo)
    else
        CAN_INSTALL_SYSTEM_PACKAGES=0
    fi
fi

echo "Checking for an existing Python $REQUIRED_PY_VERSION (or newer 3.12.x) virtual environment..."

if [ -x "$VENV_PYTHON" ] && python_is_current "$VENV_PYTHON"; then
    echo "Reusing existing $("$VENV_PYTHON" --version 2>&1) virtual environment."
    VENV_READY=1
else
    if [ -x "$VENV_PYTHON" ]; then
        echo "Existing virtual environment is $("$VENV_PYTHON" --version 2>&1); it will be rebuilt on Python $REQUIRED_PY_VERSION or newer."
    else
        echo "No existing Python 3.12 virtual environment was found."
    fi

    if [ "$CAN_INSTALL_SYSTEM_PACKAGES" -ne 1 ]; then
        echo "Root access or sudo is required to install Python $REQUIRED_PY_VERSION. Exiting..."
        exit 1
    fi

    if command -v python3.12 >/dev/null 2>&1 && ! python3.12 -c 'import sys; raise SystemExit(0 if sys.version_info.releaselevel == "final" else 1)' 2>/dev/null; then
        echo "Removing the existing prerelease Python 3.12 packages..."
        if ! "${SUDO[@]}" apt-get remove -y python3.12 python3.12-venv python3.12-dev python3.12-tk; then
            echo "Failed to remove the prerelease Python 3.12 packages. Exiting..."
            exit 1
        fi
        if ! "${SUDO[@]}" apt-get autoremove -y; then
            echo "Failed to clean obsolete Python packages. Exiting..."
            exit 1
        fi
    fi

    PYTHON_BIN=""

    echo "Adding deadsnakes PPA..."
    if "${SUDO[@]}" apt-get install -y software-properties-common \
        && "${SUDO[@]}" add-apt-repository -y ppa:deadsnakes/ppa; then

        "${SUDO[@]}" tee /etc/apt/preferences.d/deadsnakes-python312.pref >/dev/null <<'EOF'
Package: python3.12 python3.12-*
Pin: release o=LP-PPA-deadsnakes
Pin-Priority: 1000
EOF

        if ! "${SUDO[@]}" apt-get update; then
            echo "Warning: failed to refresh APT package information; using the cached package lists."
        fi

        echo "Available Python 3.12 versions:"
        apt-cache policy python3.12

        echo "Installing the latest stable Python 3.12 from APT..."
        if ! "${SUDO[@]}" apt-get install -y python3.12 python3.12-venv python3.12-dev python3.12-tk; then
            echo "Warning: the APT Python 3.12 packages could not be installed."
        fi
    else
        echo "Warning: the deadsnakes PPA could not be added."
    fi

    if python_is_current python3.12; then
        PYTHON_BIN="python3.12"
        echo "APT provided $(python3.12 --version 2>&1)."
    else
        if command -v python3.12 >/dev/null 2>&1; then
            echo "APT only offers $(python3.12 --version 2>&1); Python $REQUIRED_PY_VERSION is not packaged for this Ubuntu release."
        fi

        echo "Falling back to a uv-managed standalone CPython $REQUIRED_PY_VERSION..."
        if ! command -v uv >/dev/null 2>&1; then
            if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
                echo "Failed to install uv. Exiting..."
                exit 1
            fi
        fi
        export PATH="$HOME/.local/bin:$PATH"

        if command -v uv >/dev/null 2>&1 && uv python install "$REQUIRED_PY_VERSION"; then
            PYTHON_BIN="$(uv python find "$REQUIRED_PY_VERSION" 2>/dev/null)"
        fi

        if ! python_is_current "$PYTHON_BIN"; then
            echo "Failed to obtain Python $REQUIRED_PY_VERSION. Exiting..."
            exit 1
        fi
        echo "Using uv-managed $("$PYTHON_BIN" --version 2>&1)."
    fi

    echo "Recreating the virtual environment with $("$PYTHON_BIN" --version 2>&1)..."
    if ! "$PYTHON_BIN" -m venv --clear "$VENV_DIR"; then
        echo "Failed to recreate the Python 3.12 virtual environment. Exiting..."
        exit 1
    fi

    if [ ! -x "$VENV_PYTHON" ] || ! python_is_current "$VENV_PYTHON"; then
        echo "Failed to create the Python $REQUIRED_PY_VERSION virtual environment. Exiting..."
        exit 1
    fi

    VENV_READY=1
fi

if [ "$VENV_READY" -ne 1 ]; then
    echo "Python $REQUIRED_PY_VERSION virtual environment setup failed. Exiting..."
    exit 1
fi

source ./venv/bin/activate || exit 1

python3 -m pip install --upgrade pip || exit 1

pip install uv || exit 1

echo "Installing requirements"

cd "$SCRIPT_DIR" || exit 1

uv pip install -r requirements.txt --index-strategy unsafe-best-match || exit 1

cd Wan2.1 || exit 1

git clone https://github.com/SECourses/DiffSynth-Studio

cd DiffSynth-Studio || exit 1

git remote set-url origin https://github.com/SECourses/DiffSynth-Studio

git reset --hard

git pull || exit 1

# Its dependencies are in the unified requirements file; its own pins (old transformers) must not be installed
uv pip install -e . --no-deps --build-constraints "$SCRIPT_DIR/uv_build_constraints.txt" || exit 1

cd ..

git clone https://github.com/FurkanGozukara/Practical-RIFE Practical-RIFE

cd Practical-RIFE || exit 1

git remote set-url origin https://github.com/FurkanGozukara/Practical-RIFE

git reset --hard

git pull || exit 1

cd "$SCRIPT_DIR" || exit 1

python3 Download_RIFE.py || exit 1

python3 "Wan2.1/verify_install.py" || exit 1

# Show completion message
echo "Virtual environment made and installed properly"
echo "Download the models with ./Download_Ubuntu.sh and start the app with ./Linux_Start_App.sh"

# Keep the terminal open
read -p "Press Enter to continue..."
