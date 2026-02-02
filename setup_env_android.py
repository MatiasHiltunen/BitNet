import subprocess
import signal
import sys
import os
import platform
import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("setup_env_android")

# Only ARM64 (aarch64) on Android/Termux
SUPPORTED_HF_MODELS = {
    "1bitLLM/bitnet_b1_58-large": {"model_name": "bitnet_b1_58-large"},
    "1bitLLM/bitnet_b1_58-3B":   {"model_name": "bitnet_b1_58-3B"},
    "HF1BitLLM/Llama3-8B-1.58-100B-tokens": {"model_name": "Llama3-8B-1.58-100B-tokens"},
    "tiiuae/Falcon3-7B-Instruct-1.58bit":  {"model_name": "Falcon3-7B-Instruct-1.58bit"},
    "tiiuae/Falcon3-7B-1.58bit":          {"model_name": "Falcon3-7B-1.58bit"},
    "tiiuae/Falcon3-10B-Instruct-1.58bit": {"model_name": "Falcon3-10B-Instruct-1.58bit"},
    "tiiuae/Falcon3-10B-1.58bit":         {"model_name": "Falcon3-10B-1.58bit"},
    "tiiuae/Falcon3-3B-Instruct-1.58bit":  {"model_name": "Falcon3-3B-Instruct-1.58bit"},
    "tiiuae/Falcon3-3B-1.58bit":          {"model_name": "Falcon3-3B-1.58bit"},
    "tiiuae/Falcon3-1B-Instruct-1.58bit":  {"model_name": "Falcon3-1B-Instruct-1.58bit"},
    "microsoft/BitNet-b1.58-2B-4T":       {"model_name": "BitNet-b1.58-2B-4T"},
}
SUPPORTED_QUANT_TYPES = {"arm64": ["i2_s", "tl1"]}
COMPILER_EXTRA_ARGS = {"arm64": ["-DBITNET_ARM_TL1=ON"]}


def system_info():
    arch = platform.machine().lower()
    if arch in ("aarch64", "arm64"):
        return platform.system(), "arm64"
    logging.error(f"Unsupported arch: {arch}. Only aarch64 supported.")
    sys.exit(1)


def run_command(cmd, shell=False, log_step=None):
    if log_step:
        log_file = Path(args.log_dir) / f"{log_step}.log"
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        with open(log_file, "w") as f:
            try:
                subprocess.run(cmd_str, shell=True, check=True, stdout=f, stderr=f)
            except subprocess.CalledProcessError as e:
                logging.error(f"Command failed: {e}. See {log_file}")
                sys.exit(1)
    else:
        subprocess.run(cmd, shell=shell, check=True)


def get_model_name():
    return SUPPORTED_HF_MODELS[args.hf_repo]["model_name"] if args.hf_repo else Path(args.model_dir).name


def prepare_model():
    model_name = get_model_name()
    model_dir = Path(args.model_dir) / model_name if args.hf_repo else Path(args.model_dir)
    if args.hf_repo:
        model_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading {args.hf_repo} to {model_dir}...")
        run_command(["huggingface-cli", "download", args.hf_repo, "--local-dir", str(model_dir)], log_step="download_model")
    else:
        if not model_dir.exists():
            logging.error(f"Model directory not found: {model_dir}")
            sys.exit(1)
    config_file = model_dir / "config.json"
    if not config_file.is_file():
        logging.error(
            f"config.json missing in {model_dir}. "
            "Use --hf-repo to download or place config.json there."
        )
        sys.exit(1)
    gguf = model_dir / f"ggml-model-{args.quant_type}.gguf"
    if not gguf.exists() or gguf.stat().st_size == 0:
        logging.info("Converting to GGUF...")
        cmd = [sys.executable, "utils/convert-hf-to-gguf-bitnet.py", str(model_dir), "--outtype", args.quant_type]
        if args.quant_embd:
            cmd.append("--quant-embd")
        run_command(cmd, log_step="convert_model")
        logging.info(f"Saved GGUF at {gguf}")
    else:
        logging.info(f"GGUF exists at {gguf}")


def setup_gguf():
    run_command([sys.executable, "-m", "pip", "install", "3rdparty/llama.cpp/gguf-py"], log_step="install_gguf")


def gen_code():
    model = get_model_name()
    if model == "bitnet_b1_58-large":
        bm, bk, bms = "256,128,256", "128,64,128", "32,64,32"
    elif model.startswith(("Llama3", "Falcon3")):
        bm, bk, bms = "256,128,256,128", "128,64,128,64", "32,64,32,64"
    else:
        bm, bk, bms = "160,320,320", "64,128,64", "32,64,32"
    codegen_model = "bitnet_b1_58-3B" if model == "BitNet-b1.58-2B-4T" else model
    logging.info(f"Generating code for {codegen_model} (BM={bm}, BK={bk}, bm={bms})")
    run_command([sys.executable, "utils/codegen_tl1.py", "--model", codegen_model,
                 "--BM", bm, "--BK", bk, "--bm", bms], log_step="codegen")


def compile_code():
    # Ensure CMake & Ninja installed
    try:
        subprocess.run(["cmake", "--version"], check=True)
        subprocess.run(["ninja", "--version"], check=True)
    except subprocess.CalledProcessError:
        logging.error("cmake or ninja not found. Install via pkg install cmake ninja.")
        sys.exit(1)
    # Clean old build dir
    build_dir = Path("build")
    if build_dir.exists():
        logging.info("Removing old build directory.")
        shutil.rmtree(build_dir)
    _, arch = system_info()
    # Configure with Ninja generator
    cmake_cmd = ["cmake", "-G", "Ninja", "-B", "build"] + COMPILER_EXTRA_ARGS[arch]
    logging.info(f"Configuring: {' '.join(cmake_cmd)}")
    run_command(' '.join(cmake_cmd) + f" | tee {args.log_dir}/cmake_configure.log", shell=True)
    # Build with Ninja (verbose)
    ninja_cmd = f"ninja -C build -v | tee {args.log_dir}/cmake_build.log"
    logging.info("Building with Ninja...")
    run_command(ninja_cmd, shell=True)


def main():
    logging.basicConfig(level=logging.INFO)
    setup_gguf()
    gen_code()
    compile_code()
    prepare_model()


def parse_args():
    parser = argparse.ArgumentParser(description="Setup BitNet on termux aarch64")
    parser.add_argument("-hr", "--hf-repo", choices=SUPPORTED_HF_MODELS.keys(), help="HuggingFace repo to fetch")
    parser.add_argument("-md", "--model-dir", default="models", help="Local model directory")
    parser.add_argument("-ld", "--log-dir", default="logs", help="Directory for logs")
    parser.add_argument("-q",  "--quant-type", choices=SUPPORTED_QUANT_TYPES["arm64"], default="i2_s", help="Quantization type")
    parser.add_argument("--quant-embd", action="store_true", help="Quantize embeddings to f16")
    return parser.parse_args()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    main()
