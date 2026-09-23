:<<"::CMDLITERAL"
@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo PhysAI Solver Environment Router ^& Installer
echo ==================================================

:: 1. Check if WSL is installed and available
where wsl.exe >nul 2>nul
if %errorlevel% equ 0 (
    echo [!] Windows environment detected, but WSL is available.
    echo The optional PhysAI solver stack must run in a Linux environment.
    set /p "run_wsl=Would you like to jump into WSL and continue automatically? (y/n): "
    if /i "!run_wsl!"=="y" (
        echo [*] Passing execution into your default WSL Linux distribution...
        for /f "delims=" %%i in ('wsl wslpath -a "%~f0"') do set "WSL_PATH=%%i"
        if not defined WSL_PATH (
            echo [-] Error: could not translate this file's path for WSL.
            pause
            exit /b 1
        )
        wsl bash -c "sed -i 's/\r$//' '!WSL_PATH!' 2>/dev/null; exec bash '!WSL_PATH!'"
        exit /b %errorlevel%
    )
    echo [-] WSL execution declined by user.
    echo You can run it yourself later with:
    echo     wsl bash "$(wslpath -a (WSL_PATH)'!WSL_PATH!')"
    pause
    exit /b 1
)

:: 2. Check for other running virtualization systems (e.g., Docker Desktop)
tasklist /FI "IMAGENAME eq Docker Desktop.exe" 2>nul | findstr /I "Docker Desktop.exe" >nul
if %errorlevel% equ 0 (
    echo.
    echo [!] Docker Desktop detected on your Windows host.
    echo [!] The solver stack cannot be deployed directly on native Windows CMD/PowerShell.
    echo Please start your container/VM manually, copy this file inside it, and
    echo run it there with: bash installer.cmd
    echo.
    pause
    exit /b 1
)

:: 3. Pure Windows fallback
echo.
echo [!] System Compatibility Error:
echo The complete optional solver stack is not supported in native Windows CMD.
echo Please install the Windows Subsystem for Linux (WSL) to proceed.
echo.
echo To set up WSL, open PowerShell as Administrator and run:
echo    wsl --install
echo.
echo Once installed, restart your machine and re-run this script.
echo.
pause
exit /b 1
::CMDLITERAL

#!/usr/bin/env bash
#
# PhysAI Optional Solver Stack Installer for Linux / WSL / macOS
# (the batch block above is only ever seen/run by Windows cmd.exe;
#  bash treats it as a no-op heredoc and skips straight to here)
#
set -euo pipefail

echo "=================================================="
echo "   PhysAI Solver Stack Installer for Linux / WSL / macOS   "
echo "=================================================="

echo -e "\n[1/3] Checking for Conda installation..."
CONDA_CMD=""

# 1. Check if 'conda' is already available in PATH or as a shell function
if command -v conda &> /dev/null || [ "$(type -t conda 2>/dev/null || true)" = "function" ]; then
    CONDA_CMD="conda"
else
    # 2. Scan standard installation directories if it's not in PATH
    SEARCH_PATHS=(
        "$HOME/miniconda3"
        "$HOME/anaconda3"
        "$HOME/mambaforge"
        "$HOME/opt/miniconda3"
        "$HOME/opt/anaconda3"
        "/opt/miniconda3"
        "/opt/anaconda3"
        "/usr/local/miniconda3"
        "/usr/local/anaconda3"
    )

    for path in "${SEARCH_PATHS[@]}"; do
        if [ -x "$path/bin/conda" ]; then
            CONDA_CMD="$path/bin/conda"
            break
        fi
    done
fi

# 3. If still not found, present the interactive menu
if [ -z "$CONDA_CMD" ]; then
    echo "[!] No active Conda environment (Anaconda/Miniconda/Mambaforge) detected."
    echo "--------------------------------------------------"
    echo "The optional native solver packages require a Conda environment."
    echo "Please choose an option:"
    echo "  1) Automatically download and install Miniconda (lightweight, recommended)"
    echo "  2) Automatically download and install Anaconda (full data-science suite)"
    echo "  3) Abort installation"
    echo "--------------------------------------------------"

    while true; do
        read -r -p "Enter your choice [1-3]: " user_choice
        case "$user_choice" in
            1) INSTALL_TYPE="miniconda"; break ;;
            2) INSTALL_TYPE="anaconda"; break ;;
            3) echo "[-] Installation cancelled. The solver stack requires Conda."; exit 1 ;;
            *) echo "[!] Invalid option. Please enter 1, 2, or 3." ;;
        esac
    done

    # Detect Operating System and Architecture
    OS_TYPE="Linux"
    ARCH_TYPE="x86_64"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS_TYPE="MacOSX"
        if [[ "$(uname -m)" == "arm64" ]]; then
            ARCH_TYPE="arm64"
        fi
    else
        if [[ "$(uname -m)" == "aarch64" ]]; then
            ARCH_TYPE="aarch64"
        fi
    fi

    # Set up installer filename, correct download URL, and default path
    if [ "$INSTALL_TYPE" = "miniconda" ]; then
        INSTALLER_NAME="Miniconda3-latest-${OS_TYPE}-${ARCH_TYPE}.sh"
        DOWNLOAD_URL="https://repo.anaconda.com/miniconda/${INSTALLER_NAME}"
        DEFAULT_DIR="$HOME/miniconda3"
    else
        INSTALLER_NAME="Anaconda3-latest-${OS_TYPE}-${ARCH_TYPE}.sh"
        DOWNLOAD_URL="https://repo.anaconda.com/archive/${INSTALLER_NAME}"
        DEFAULT_DIR="$HOME/anaconda3"
    fi

    # Ask user for a custom installation path
    echo -e "\n[?] Where would you like to install ${INSTALL_TYPE^}?"
    read -r -p "Enter path [Press Enter for default: ${DEFAULT_DIR}]: " TARGET_DIR

    if [ -z "$TARGET_DIR" ]; then
        TARGET_DIR="$DEFAULT_DIR"
    fi

    # Expand a leading ~ manually if the user typed it
    TARGET_DIR="${TARGET_DIR/#\~/$HOME}"

    if [ -e "$TARGET_DIR" ]; then
        echo "[-] Error: target path '$TARGET_DIR' already exists. Choose a different path or remove it first."
        exit 1
    fi

    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "$TMP_DIR"' EXIT
    INSTALLER_PATH="$TMP_DIR/$INSTALLER_NAME"

    echo -e "\n[*] Downloading ${INSTALL_TYPE^} for ${OS_TYPE} (${ARCH_TYPE})..."
    if command -v curl &> /dev/null; then
        curl -fL -o "$INSTALLER_PATH" "$DOWNLOAD_URL"
    elif command -v wget &> /dev/null; then
        wget -O "$INSTALLER_PATH" "$DOWNLOAD_URL"
    else
        echo "[-] Error: neither 'curl' nor 'wget' was found. Please install one and re-run this script."
        exit 1
    fi

    echo "[*] Installing ${INSTALL_TYPE^} silently to ${TARGET_DIR}..."
    mkdir -p "$(dirname "$TARGET_DIR")"
    bash "$INSTALLER_PATH" -b -p "$TARGET_DIR"

    echo "[*] Initializing Conda shell integration..."
    "$TARGET_DIR/bin/conda" init bash &> /dev/null || true
    "$TARGET_DIR/bin/conda" init zsh &> /dev/null || true
    CONDA_CMD="$TARGET_DIR/bin/conda"
    echo "[+] ${INSTALL_TYPE^} installed successfully."
else
    echo "[+] Conda detected: $CONDA_CMD"
fi

# User confirmation to deploy the optional solver environment
echo -e "\n[2/3] Preparing the PhysAI solver environment."
echo "This creates a Conda environment named 'physai-solvers' with Python 3.11."
echo "Packages: Dedalus, FiPy, classic FEniCS, FEniCSx, Meep, and CuPy."
read -r -p "Do you want to proceed? (y/n): " proceed
if [[ "$proceed" != "y" && "$proceed" != "Y" ]]; then
    echo "[!] Installation canceled by user."
    exit 0
fi

# Install the optional solver stack from conda-forge. These compiled stacks
# share MPI/PETSc dependencies, so let Conda resolve a compatible set.
echo -e "\n[3/3] Installing the solver packages... this may take a while."
"$CONDA_CMD" create -n physai-solvers -c conda-forge \
    python=3.11 pip dedalus fipy fenics-dolfin fenics-dolfinx pymeep cupy -y

echo -e "\n=================================================="
echo "[+] The PhysAI solver environment was installed successfully!"
echo "=================================================="
echo "To start using it, open a new terminal window and run:"
echo "    conda activate physai-solvers"
echo "    python -m pip install physai"
echo "Then run PhysAI from this environment to use the installed solver adapters."
