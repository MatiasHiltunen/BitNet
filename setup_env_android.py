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
BASE_ANDROID_CMAKE_ARGS = [
    "-DGGML_LLAMAFILE=OFF",
    "-DLLAMA_BUILD_SERVER=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
]
BUILD_PROFILES = {
    "default": {
        "cmake_args": [],
        "cflags": "",
        "cxxflags": "",
    },
    "tuned-aarch64": {
        "cmake_args": ["-DGGML_OPENMP=OFF"],
        "cflags": "-march=armv8.2-a+dotprod+i8mm+bf16 -mtune=cortex-a710",
        "cxxflags": "-march=armv8.2-a+dotprod+i8mm+bf16 -mtune=cortex-a710",
    },
}

MODEL_NAME_ALIASES = {
    "bitnet-b1.58-2b-4t": "BitNet-b1.58-2B-4T",
}


def system_info():
    arch = platform.machine().lower()
    if arch in ("aarch64", "arm64"):
        return platform.system(), "arm64"
    logging.error(f"Unsupported arch: {arch}. Only aarch64 supported.")
    sys.exit(1)


def cpu_supports_tuned_profile():
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return False
    text = cpuinfo.read_text(errors="ignore").lower()
    # Require dot-product and int8 matrix multiply for the tuned profile.
    return "asimddp" in text and "i8mm" in text


def get_build_profile():
    if args.profile != "auto":
        return args.profile
    return "tuned-aarch64" if cpu_supports_tuned_profile() else "default"


def run_command(cmd, shell=False, log_step=None):
    if log_step:
        log_file = Path(args.log_dir) / f"{log_step}.log"
        with open(log_file, "w") as f:
            try:
                if shell:
                    cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
                    subprocess.run(cmd_str, shell=True, check=True, stdout=f, stderr=f)
                else:
                    subprocess.run(cmd, shell=False, check=True, stdout=f, stderr=f)
            except subprocess.CalledProcessError as e:
                logging.error(f"Command failed: {e}. See {log_file}")
                sys.exit(1)
    else:
        subprocess.run(cmd, shell=shell, check=True)


def get_model_name():
    if args.hf_repo:
        return SUPPORTED_HF_MODELS[args.hf_repo]["model_name"]
    local_name = Path(args.model_dir).name
    return MODEL_NAME_ALIASES.get(local_name.lower(), local_name)


def prepare_model():
    model_name = get_model_name()
    model_dir = Path(args.model_dir) / model_name if args.hf_repo else Path(args.model_dir)
    gguf = model_dir / f"ggml-model-{args.quant_type}.gguf"
    if args.hf_repo:
        model_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading {args.hf_repo} to {model_dir}...")
        run_command(["huggingface-cli", "download", args.hf_repo, "--local-dir", str(model_dir)], log_step="download_model")
    else:
        if not model_dir.exists():
            logging.error(f"Model directory not found: {model_dir}")
            sys.exit(1)
    if gguf.exists() and gguf.stat().st_size > 0:
        logging.info(f"GGUF exists at {gguf}")
        return

    # Reuse an already-quantized sibling model directory when available.
    sibling_gguf = next(
        (
            path / f"ggml-model-{args.quant_type}.gguf"
            for path in model_dir.parent.iterdir()
            if path.is_dir() and (path / f"ggml-model-{args.quant_type}.gguf").is_file()
        ),
        None,
    )
    if sibling_gguf is not None and sibling_gguf.stat().st_size > 0:
        model_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Reusing GGUF from {sibling_gguf}")
        shutil.copy2(sibling_gguf, gguf)
        logging.info(f"Saved GGUF at {gguf}")
        return

    config_file = model_dir / "config.json"
    if not config_file.is_file():
        logging.error(
            f"config.json missing in {model_dir} and no existing GGUF found in sibling model directories. "
            "Use --hf-repo to download a full HF model or place a GGUF file in the model directory."
        )
        sys.exit(1)

    if not gguf.exists() or gguf.stat().st_size == 0:
        logging.info("Converting to GGUF...")
        if args.quant_type == "i2_s":
            f32_gguf = model_dir / "ggml-model-f32.gguf"
            if not f32_gguf.exists() or f32_gguf.stat().st_size == 0:
                run_command(
                    [sys.executable, "utils/convert-hf-to-gguf-bitnet.py", str(model_dir), "--outtype", "f32"],
                    log_step="convert_model_f32",
                )
            quant_cmd = ["./build/bin/llama-quantize"]
            if args.quant_embd:
                quant_cmd.extend(["--token-embedding-type", "f16"])
            quant_cmd.extend([str(f32_gguf), str(gguf), "I2_S", "1"])
            if args.quant_embd:
                quant_cmd.append("1")
            run_command(quant_cmd, log_step="convert_model")
        else:
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
    model_l = model.lower()
    if model_l in ("bitnet_b1_58-large", "bitnet-b1.58-large"):
        bm, bk, bms = "256,128,256", "128,64,128", "32,64,32"
        codegen_model = "bitnet_b1_58-large"
    elif model_l.startswith(("llama3", "falcon3")):
        bm, bk, bms = "256,128,256,128", "128,64,128,64", "32,64,32,64"
        codegen_model = "Llama3-8B-1.58-100B-tokens"
    elif model_l in ("bitnet_b1_58-3b", "bitnet-b1.58-3b"):
        bm, bk, bms = "160,320,320", "64,128,64", "32,64,32"
        codegen_model = "bitnet_b1_58-3B"
    elif model_l in ("bitnet-b1.58-2b-4t", "bitnet_b1_58-2b-4t"):
        bm, bk, bms = "160,320,320", "64,128,64", "32,64,32"
        codegen_model = "bitnet_b1_58-3B"
    else:
        logging.error(
            "Unsupported model '%s' for TL1 codegen. "
            "Use one of: bitnet_b1_58-large, bitnet_b1_58-3B, "
            "BitNet-b1.58-2B-4T, Llama3/Falcon3 families.",
            model,
        )
        sys.exit(1)
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
    _, arch = system_info()
    profile = get_build_profile()
    profile_cfg = BUILD_PROFILES[profile]
    logging.info("Using build profile: %s", profile)
    build_dir = Path("build")
    cache_file = build_dir / "CMakeCache.txt"
    if cache_file.exists():
        cache_text = cache_file.read_text(errors="ignore")
        if (
            "CMAKE_GENERATOR:INTERNAL=Ninja" not in cache_text
            or f"GGML_OPENMP:BOOL={'OFF' if '-DGGML_OPENMP=OFF' in profile_cfg['cmake_args'] else 'ON'}" not in cache_text
            or (profile_cfg["cflags"] and profile_cfg["cflags"] not in cache_text)
        ):
            logging.info("Build directory uses a different generator. Recreating build directory.")
            shutil.rmtree(build_dir)
    # Configure with Ninja generator
    cmake_cmd = [
        "cmake",
        "-G",
        "Ninja",
        "-B",
        "build",
        "-DCMAKE_BUILD_TYPE=Release",
        *BASE_ANDROID_CMAKE_ARGS,
        *profile_cfg["cmake_args"],
        *COMPILER_EXTRA_ARGS[arch],
    ]
    if profile_cfg["cflags"]:
        cmake_cmd.append(f"-DCMAKE_C_FLAGS={profile_cfg['cflags']}")
    if profile_cfg["cxxflags"]:
        cmake_cmd.append(f"-DCMAKE_CXX_FLAGS={profile_cfg['cxxflags']}")
    logging.info(f"Configuring: {' '.join(cmake_cmd)}")
    run_command(cmake_cmd, log_step="cmake_configure")
    # Build only targets required for local inference and conversion.
    jobs = max(1, args.jobs)
    targets = ["llama-cli"]
    if args.quant_type == "i2_s":
        targets.append("llama-quantize")
    ninja_cmd = ["ninja", "-C", "build", "-v", f"-j{jobs}", *targets]
    logging.info(f"Building with Ninja (jobs={jobs})...")
    run_command(ninja_cmd, log_step="cmake_build")


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
    parser.add_argument("-j", "--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)), help="Parallel build jobs")
    parser.add_argument(
        "--profile",
        choices=["auto", *BUILD_PROFILES.keys()],
        default="auto",
        help="Build profile for Android/Termux. 'auto' picks tuned-aarch64 when CPU supports it.",
    )
    parser.add_argument("--quant-embd", action="store_true", help="Quantize embeddings to f16")
    return parser.parse_args()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    main()
