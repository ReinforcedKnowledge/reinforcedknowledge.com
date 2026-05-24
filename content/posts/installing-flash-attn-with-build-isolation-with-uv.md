+++
title    = "Install flash-attn without crying while using uv"
date     = "2025-09-23T01:15:24+00:00"
draft    = false
categories = ["Python"]
tags       = ["uv"]
+++

## Introduction



[`uv`’s documentation about build isolation](https://docs.astral.sh/uv/concepts/projects/config/#build-isolation) is already great and has everything you need and I recommend and suggest to read before anything else. Read the whole documentation even, especially if you're just starting with `uv`.  
  
I'm writing this article since these last days I had to either install `flash-attn` as a wheel directly or build it from source and since I struggled different times, I thought of noting down my path so that it stays fresh in my mind. Unfortunately, at the time I didn't read the documentation when going through all of this cause I thought that I could figure it out quickly. Well…  
  
**Note**: too tired to cover no build isolation but I think most of the readers are familiar with it since it was the only method to install `flash-attn` before.   
  
**Another note**: there's obviously too much to say about FA and I'm just covering minimal areas, just what's needed to be known or understood to install `flash-attn` without sweating.



## TL;DR



About FA itself:



**Wheels**



- No nvcc required. CUDA toolkit not strictly needed to install.
- Matching is based on: CUDA major used by your PyTorch build (normalized to 11 or 12 in FA’s setup logic), torch major.minor, cxx11abi flag, CPython tag, platform.
- Wheel names look like: flash\_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp313-cp313-linux\_x86\_64.whl
- FLASH\_ATTENTION\_SKIP\_CUDA\_BUILD=TRUE will skip compile, fail fast if no wheel



**Build from source**



- Requires: nvcc (CUDA >= 11.7), C++17 compiler, CUDA PyTorch, Ampere+ GPU (SM >= 80: 80/90/100/101/110/120 depending on toolkit), CUTLASS bundled via submodule/sdist
- You can narrow targets with FLASH\_ATTN\_CUDA\_ARCHS (e.g. 90 for H100, 100 for Blackwell). Otherwise targets will be added depending on your CUDA version
- Flags that might help:
  - `MAX_JOBS` (from ninja for parallelizing the build) + `NVCC_THREADS`
  - `CUDA_HOME` for cleaner detection (less flaky builds)
  - `FLASH_ATTENTION_FORCE_BUILD=TRUE` if you want to compile even when a wheel exists
  - `FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE` if your base image/toolchain needs C++11 ABI to match PyTorch



About build isolation:



- `flash-attn` fails to install with uv by default because it needs `torch` at build time but doesn’t declare it
- Fix: add `torch` under `[tool.uv.extra-build-dependencies]`
- **Pinning `torch` there only affects the build env but runtime may still resolve to a different version**
- Use `match-runtime = true` if you want build-time and runtime torch to align
- Older versions of `flash-attn` ship metadata `uv` can’t parse so you need to supply it manually with `[[tool.uv.dependency-metadata]]` but only if you're using `match-runtime = true`, if you're using the simple dependency declaration in `[tool.uv.extra-build-dependencies]` then it's okay
- Optional deps work fine once you apply the same build rules



## What do you need for flash-attn?



### For installing the wheel



This one is simple and easy, you need no nvcc, no CUDA toolkit etc. The build process might warn you if those aren’t present but it won't fail. Looking at a wheel's filename and also at [`setup.py`'s logic](https://github.com/Dao-AILab/flash-attention/blob/5c1627a7a1cda9c32cb9b937a053564e663f81bc/setup.py) we figure out that what it cares about is the **CUDA build of your pytorch** (they normalize to 11.8 or 12.3 internally with `torch_cuda_version = parse("11.8") if torch_cuda_version.major == 11 else parse("12.3")`, and then they only extract the major later on with `cuda_version = f"{torch_cuda_version.major}"` so the filename only carries the major), the **pytorch's major and minor versions**, **whether or not your pytorch uses the C++ 11 ABI** (it gets pulled from `torch._C._GLIBCXX_USE_CXX11_ABI`), **CPython version** and finally, **the platform tag**. So wheel names are like `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl` or `flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp313-cp313-linux_x86_64.whl`.



**Note**: remember that installing the wheel doesn't guarantee that the code will work because you can also alter the environment in a way to trick flash attention to install a wheel without that wheel being able to run properly.



### For building from source



There are two paths here, whether you build for CUDA or whether you build for ROCm. I don't know much about ROCm so I'll only talk about CUDA. So this one obviously is more complex. Let's start with some things that will automatically fail your build:



The **compute capability**. This is not explicitly mentioned in setup.py but you need a GPU with compute capability of at least 8.0 (Ampere/Hopper+) because flash attention only compiles kernels for SM 80/90 and newer (100/110/120 when the toolkit supports them). You'll notice that across setup.py where they use sm\_90 or sm\_100 etc.



Whether you have the NVIDIA **CUTLASS library** or not. It's these lines [L204](https://github.com/Dao-AILab/flash-attention/blob/5c1627a7a1cda9c32cb9b937a053564e663f81bc/setup.py#L204) and [L212](https://github.com/Dao-AILab/flash-attention/blob/5c1627a7a1cda9c32cb9b937a053564e663f81bc/setup.py#L212). If you add flash-attn through a package manager you should be good, it should be bundled in the sdist according to that assert message.



**CUDA pytorch** is required obviously otherwise the build will fail.



A **C++ 17 compatible compiler**, it literally has `compiler_c17_flag=["-O3", "-std=c++17"]` at [L264](https://github.com/Dao-AILab/flash-attention/blob/5c1627a7a1cda9c32cb9b937a053564e663f81bc/setup.py#L264) but also the c++17 standard is pased to the `nvcc` flags to compile against.



So the **nvcc compiler** is required as well.



**CUDA Toolkit** of **at least version 11.7**.



There are other requirements that are more or less important like the ninja, packaging and psutil and einops. But those do get installed automatically with a package manager.



### Some interesting flags



`CUDA_HOME`: if you don't set it, it's not a problem per se, you'll just get some warnings but compilation should succeed in most cases. But actually I've noticed that when building images, exposing that env var helped a lot to pass some builds that failed when it was not set up.



`FLASH_ATTENTION_SKIP_CUDA_BUILD` if you want to always skip the build process and fail automatically when there is no appropriate wheel.



`FLASH_ATTN_CUDA_ARCHS` to only choose some of the targets for which to build the package.



`MAX_JOBS`, this is more a ninja flag. You should play with it and find which works best for you. `flash-attn` sets it up automatically but you can get into some weird bugs or the installation might just stale if it takes all the RAM to build the package. It controls how many compilation jobs (translation units) run in parallel via Ninja/setuptools. `flash-attn` auto-picks a safe number based on CPU and free RAM but you can try different values and see what would work best for you.



I found that setting it up with `NVCC_THREADS` which is also used to tune compile parallelism can help as well, it controls internal multithreading within each `nvcc` invocation. The default value is 4 but esentially, **Bigger `MAX_JOBS` = more `.cu` files compiled concurrently** (scales across files) and **Bigger `NVCC_THREADS` = each single file’s compile can use more host threads** (speeds each file).



## extra-build-dependencies



Let's explore what happens when we do `uv add flash-attn` alone, without any other dependency. We get the following messag



```bash
Resolved 29 packages in 859ms
  × Failed to build `flash-attn==2.8.3`
  ├─▶ The build backend returned an error
  ╰─▶ Call to `setuptools.build_meta:__legacy__.build_wheel` failed (exit status: 1)

      [stderr]
      Traceback (most recent call last):
        File "<string>", line 14, in <module>
        File "/dev/shm/ytahtah/vision/.cache/builds-v0/.tmpgAwhoH/lib/python3.10/site-packages/setuptools/build_meta.py", line 331, in get_requires_for_build_wheel
          return self._get_build_requires(config_settings, requirements=[])
        File "/dev/shm/ytahtah/vision/.cache/builds-v0/.tmpgAwhoH/lib/python3.10/site-packages/setuptools/build_meta.py", line 301, in _get_build_requires
          self.run_setup()
        File "/dev/shm/ytahtah/vision/.cache/builds-v0/.tmpgAwhoH/lib/python3.10/site-packages/setuptools/build_meta.py", line 512, in run_setup
          super().run_setup(setup_script=setup_script)
        File "/dev/shm/ytahtah/vision/.cache/builds-v0/.tmpgAwhoH/lib/python3.10/site-packages/setuptools/build_meta.py", line 317, in run_setup
          exec(code, locals())
        File "<string>", line 22, in <module>
      ModuleNotFoundError: No module named 'torch'

      hint: This error likely indicates that `flash-attn@2.8.3` depends on `torch`, but doesn't declare it as a build dependency. If `flash-attn` is a first-party package, consider adding `torch` to its
      `build-system.requires`. Otherwise, either add it to your `pyproject.toml` under:

      [tool.uv.extra-build-dependencies]
      flash-attn = ["torch"]

      or `uv pip install torch` into the environment and re-run with `--no-build-isolation`.
  help: If you want to add the package regardless of the failed resolution, provide the `--frozen` flag to skip locking and syncing.
```



Since `flash-attn` is a third party package that we're installing, we have to add it to `tool.uv.extra-build-dependencies`:



```
[project]
name = "test-fa"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.10"
dependencies = []

[tool.uv.extra-build-dependencies]
flash-attn = ["torch"]
```



Running `uv add flash-attn` correctly installs it:



```
warning: The `extra-build-dependencies` option is experimental and may change without warning. Pass `--preview-features extra-build-dependencies` to disable this warning.
Resolved 29 packages in 18ms
      Built flash-attn==2.8.3
Prepared 20 packages in 36.52s
Installed 27 packages in 55ms
 + einops==0.8.1
 + filelock==3.19.1
 + flash-attn==2.8.3
 + fsspec==2025.9.0
 + jinja2==3.1.6
 + markupsafe==3.0.2
 + mpmath==1.3.0
 + networkx==3.4.2
 + nvidia-cublas-cu12==12.8.4.1
 + nvidia-cuda-cupti-cu12==12.8.90
 + nvidia-cuda-nvrtc-cu12==12.8.93
 + nvidia-cuda-runtime-cu12==12.8.90
 + nvidia-cudnn-cu12==9.10.2.21
 + nvidia-cufft-cu12==11.3.3.83
 + nvidia-cufile-cu12==1.13.1.3
 + nvidia-curand-cu12==10.3.9.90
 + nvidia-cusolver-cu12==11.7.3.90
 + nvidia-cusparse-cu12==12.5.8.93
 + nvidia-cusparselt-cu12==0.7.1
 + nvidia-nccl-cu12==2.27.3
 + nvidia-nvjitlink-cu12==12.8.93
 + nvidia-nvtx-cu12==12.8.90
 + setuptools==80.9.0
 + sympy==1.14.0
 + torch==2.8.0
 + triton==3.4.0
 + typing-extensions==4.15.0
```



And updates the `pyproject.toml` to add `flash-attn>=2.8.3`. We see it having installed `torch==2.8.0` but obviously it's not its role to pin it in the `pyproject.toml` (only to lock it in `uv.lock` file). Now what if *we* pinned the `torch` version?



```
[project]
name = "test-fa"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "flash-attn>=2.8.3",
]

[tool.uv.extra-build-dependencies]
flash-attn = ["torch==2.7.0"]
```



Doing `uv sync` shows that it still installs `torch==2.8.0`. But this is tricky, it's not that pinning the version in `extra-build-dependencies` have no effect, but it's a combination of different things. So what's happening? When doing `extra-build-dependencies`, those declared dependencies will be installed in the **isolated** virtual environment that `flash-attn` gets build in it. And, that `torch` as a build dependency is declared with `torch==2.7.0` and `flash-attn` is built against it. As a matter of fact, if you try `uv sync -v`, you'll see the following debug lines:



```
DEBUG torch.__version__ = 2.7.0+cu126 
DEBUG running bdist_wheel 
DEBUG Guessing wheel URL: https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```



Those lines come from [`flash-attn`'s `setup.py`](https://github.com/Dao-AILab/flash-attention/blob/5c1627a7a1cda9c32cb9b937a053564e663f81bc/setup.py#L528). And as you can see, it uses the wheel with `torch` version 2.7. But if we go investigate what was installed in our env and see the `pkg-info` that you can find at `lib64/python3.12/site-packages/flash_attn-2.8.3.dist-info`, then you find in `METADATA` that `Requires-Dist: torch`. So `flash-attn` doesn't impose any restrictions on the version of `torch` after it's being built. And I guess then `uv` just installs the most recently compatible version of `torch` for your platform. Thus the 2.8.0



So even if we see `Built flash-attn` in the logs, it's not *really* building it. It is in some sense building it by calling `setup.py`, but `setup.py` fetches the an appropriate wheel behind the scenes if there exists one, and fallbacks to building from source if there isn't. Unless you have set the variable `FLASH_ATTENTION_SKIP_CUDA_BUILD` to `TRUE` as show [here in the `setup.py`](https://github.com/Dao-AILab/flash-attention/blob/5c1627a7a1cda9c32cb9b937a053564e663f81bc/setup.py#L62C1-L62C82): `SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"`. In that case, if it doesn't find an appropriate wheel, it'll just fail.



**Conclusion**: pinning `torch` inside `extra-build-dependencies` doesn’t mean that’s the `torch` you’ll end up with at runtime. It only affects the isolated build env.



Now, let's try a different version of `flash-attn`, version 2.7.3



```
[project]
name = "test-fa"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "flash-attn==2.7.3",
]

[tool.uv.extra-build-dependencies]
flash-attn = ["torch==2.7.0"]
```



And this, this will take time because now it'll **build from source**. I don't know why but I had issues building from source on multiple GPUs, so I had to restrict to only one using `CUDA_VISIBLE_DEVICES`. Also since I'm on an H100, I can just restrict the build target using `export FLASH_ATTN_CUDA_ARCHS=90` and `export TORCH_CUDA_ARCH_LIST=90`.



```
nstalled 27 packages in 50ms
 + einops==0.8.1
 + filelock==3.19.1
 + flash-attn==2.7.3
 + fsspec==2025.9.0
 + jinja2==3.1.6
 + markupsafe==3.0.2
 + mpmath==1.3.0
 + networkx==3.5
 + nvidia-cublas-cu12==12.8.4.1
 + nvidia-cuda-cupti-cu12==12.8.90
 + nvidia-cuda-nvrtc-cu12==12.8.93
 + nvidia-cuda-runtime-cu12==12.8.90
 + nvidia-cudnn-cu12==9.10.2.21
 + nvidia-cufft-cu12==11.3.3.83
 + nvidia-cufile-cu12==1.13.1.3
 + nvidia-curand-cu12==10.3.9.90
 + nvidia-cusolver-cu12==11.7.3.90
 + nvidia-cusparse-cu12==12.5.8.93
 + nvidia-cusparselt-cu12==0.7.1
 + nvidia-nccl-cu12==2.27.3
 + nvidia-nvjitlink-cu12==12.8.93
 + nvidia-nvtx-cu12==12.8.90
 + setuptools==80.9.0
 + sympy==1.14.0
 + torch==2.8.0
 + triton==3.4.0
 + typing-extensions==4.15.0
```



## match-runtime = True



Now, what if you want to fix the `torch` version in your repo and have exactly the same version that is installed after `flash-attn` is built and the one used during the build process. Well, there is a way to do that, `match-runtime`: <https://docs.astral.sh/uv/concepts/projects/config/#augmenting-build-dependencies>



So our `pyproject.toml` becomes:



```
[project]
name = "test-fa"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.10"
dependencies = ["flash-attn", "torch"]

[tool.uv.extra-build-dependencies]
flash-attn = [{ requirement = "torch", match-runtime = true }]

[tool.uv.extra-build-variables]
flash-attn = { FLASH_ATTENTION_SKIP_CUDA_BUILD = "TRUE" }
```



Running `uv sync` does what's needed and installs it properly. And for the moment if we pin the following versions we can even remove the `FLASH_ATTENTION_SKIP_CUDA_BUILD` flag because we'll get the wheel instead of having to build from source.



```
warning: The `extra-build-dependencies` option is experimental and may change without warning. Pass `--preview-features extra-build-dependencies` to disable this warning.
Resolved 29 packages in 804ms
      Built flash-attn==2.8.3
Prepared 27 packages in 40.71s
Installed 27 packages in 47ms
 + einops==0.8.1
 + filelock==3.19.1
 + flash-attn==2.8.3
 + fsspec==2025.9.0
 + jinja2==3.1.6
 + markupsafe==3.0.2
 + mpmath==1.3.0
 + networkx==3.4.2
 + nvidia-cublas-cu12==12.8.4.1
 + nvidia-cuda-cupti-cu12==12.8.90
 + nvidia-cuda-nvrtc-cu12==12.8.93
 + nvidia-cuda-runtime-cu12==12.8.90
 + nvidia-cudnn-cu12==9.10.2.21
 + nvidia-cufft-cu12==11.3.3.83
 + nvidia-cufile-cu12==1.13.1.3
 + nvidia-curand-cu12==10.3.9.90
 + nvidia-cusolver-cu12==11.7.3.90
 + nvidia-cusparse-cu12==12.5.8.93
 + nvidia-cusparselt-cu12==0.7.1
 + nvidia-nccl-cu12==2.27.3
 + nvidia-nvjitlink-cu12==12.8.93
 + nvidia-nvtx-cu12==12.8.90
 + setuptools==80.9.0
 + sympy==1.14.0
 + torch==2.8.0
 + triton==3.4.0
 + typing-extensions==4.15.0
```



So we get what we expected to get `flash-attn==2.8.3` and `torch==2.8.0`. Now, let's pin these versions and remove the flag just to ensure it's still working properly :D



```
[project]
name = "test-fa"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.10"
dependencies = ["flash-attn==2.8.3", "torch==2.8.0"]

[tool.uv.extra-build-dependencies]
flash-attn = [{ requirement = "torch", match-runtime = true }]
```



And it works as expected, exactly as above.



So now what if we wanted `flash-attn` to be an optional dependency? Well, let's try it:



```
[project]
name = "test-fa"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.10"
dependencies = ["torch==2.8.0"]

[project.optional-dependencies]
flash = [
    "flash-attn>=2.8.3",
]

[tool.uv.extra-build-dependencies]
flash-attn = [{ requirement = "torch", match-runtime = true }]
```



Now let's try `uv sync`, it works as expected:



```
Resolved 29 packages in 613ms
Prepared 25 packages in 29.03s
Installed 25 packages in 233ms
 + filelock==3.19.1
 + fsspec==2025.9.0
 + jinja2==3.1.6
 + markupsafe==3.0.2
 + mpmath==1.3.0
 + networkx==3.4.2
 + nvidia-cublas-cu12==12.8.4.1
 + nvidia-cuda-cupti-cu12==12.8.90
 + nvidia-cuda-nvrtc-cu12==12.8.93
 + nvidia-cuda-runtime-cu12==12.8.90
 + nvidia-cudnn-cu12==9.10.2.21
 + nvidia-cufft-cu12==11.3.3.83
 + nvidia-cufile-cu12==1.13.1.3
 + nvidia-curand-cu12==10.3.9.90
 + nvidia-cusolver-cu12==11.7.3.90
 + nvidia-cusparse-cu12==12.5.8.93
 + nvidia-cusparselt-cu12==0.7.1
 + nvidia-nccl-cu12==2.27.3
 + nvidia-nvjitlink-cu12==12.8.93
 + nvidia-nvtx-cu12==12.8.90
 + setuptools==80.9.0
 + sympy==1.14.0
 + torch==2.8.0
 + triton==3.4.0
 + typing-extensions==4.15.0
```



And what if we try `uv sync --extra flash`? It works as well:



```
Resolved 29 packages in 4ms
      Built flash-attn==2.8.3
Prepared 2 packages in 8.62s
Installed 2 packages in 1ms
 + einops==0.8.1
 + flash-attn==2.8.3
```



Ok, now let's change the versions of `flash-attn` to 2.7.3 and `torch` to 2.7.0



```
[project]
name = "test-fa"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.10"
dependencies = ["torch==2.7.0"]

[project.optional-dependencies]
flash = [
    "flash-attn==2.7.3",
]

[tool.uv.extra-build-dependencies]
flash-attn = [{ requirement = "torch", match-runtime = true }]
```



Then we get this weird error:



```
warning: The `extra-build-dependencies` option is experimental and may change without warning. Pass `--preview-features extra-build-dependencies` to disable this warning.
  × Failed to download and build `flash-attn==2.7.3`
  ╰─▶ Extra build requirement `torch` was declared with `match-runtime = true`, but `flash-attn` does not declare static metadata, making runtime-matching impossible
  help: `flash-attn` (v2.7.3) was included because `test-fa[flash]` (v0.1.0) depends on `flash-attn==2.7.3`
```



Let's declare some static metadata:



```
[project]
name = "test-fa"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.10"
dependencies = ["torch==2.7.0"]

[project.optional-dependencies]
flash = [
    "flash-attn==2.7.3",
]

[tool.uv.extra-build-dependencies]
flash-attn = [{ requirement = "torch", match-runtime = true }]

[[tool.uv.dependency-metadata]]
name = "flash-attn"
version = "2.7.3"
requires-dist = ["torch", "einops"]
```



Why does this fix it? Well, to know more about this we can run `uv sync -v` on the version with error, you should see something like:



```
...
DEBUG No static `PKG-INFO` available for: flash-attn==2.7.3 (PkgInfo(UnsupportedMetadataVersion("2.1")))
...
```



And we can further verify that by downloading the sdist from <https://pypi.org/project/flash-attn/2.7.3/#files>, then extracting it and opening `PKG-INFO`. We find `Metadata-Version: 2.1`, so though the `PKG-INFO` file has `Requires-Dist: torch` and `Requires-Dist: einops`, since `uv` doesn't support that metadata version, it can't parse it and we have to supply it manually. But, if you download and extract the sdist for version 2.8.3, <https://pypi.org/project/flash-attn/2.8.3/#files>, and open its `PKG-INFO`, you should see `Metadata-Version: 2.2`.



By the way, we'd also encounter this problem if we had `flash-attn` as a base dependency and not an extra.



Now let's try `uv sync --extra flash` and see what will happen:



```
Resolved 29 packages in 433ms
      Built flash-attn==2.7.3
Prepared 27 packages in 8m 29s
Installed 27 packages in 46ms
 + einops==0.8.1
 + filelock==3.19.1
 + flash-attn==2.7.3
 + fsspec==2025.9.0
 + jinja2==3.1.6
 + markupsafe==3.0.2
 + mpmath==1.3.0
 + networkx==3.5
 + nvidia-cublas-cu12==12.6.4.1
 + nvidia-cuda-cupti-cu12==12.6.80
 + nvidia-cuda-nvrtc-cu12==12.6.77
 + nvidia-cuda-runtime-cu12==12.6.77
 + nvidia-cudnn-cu12==9.5.1.17
 + nvidia-cufft-cu12==11.3.0.4
 + nvidia-cufile-cu12==1.11.1.6
 + nvidia-curand-cu12==10.3.7.77
 + nvidia-cusolver-cu12==11.7.1.2
 + nvidia-cusparse-cu12==12.5.4.2
 + nvidia-cusparselt-cu12==0.6.3
 + nvidia-nccl-cu12==2.26.2
 + nvidia-nvjitlink-cu12==12.6.85
 + nvidia-nvtx-cu12==12.6.77
 + setuptools==80.9.0
 + sympy==1.14.0
 + torch==2.7.0
 + triton==3.3.0
 + typing-extensions==4.15.0
```



And it builds properly!



**Note**: At every new step above I cleaned the cache and deleted the venv and the `uv.lock` just to ensure that we're starting from a fresh slate.



**Conclusion**: `match-runtime` is the trick, it makes sure `flash-attn` is built against the exact same `torch` you’ll run with.
